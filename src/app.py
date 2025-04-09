import os
import datetime
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from sqlalchemy import text
from pgvector.sqlalchemy import Vector
from process import calculate_embedding_similarity, process_image

DEFAULT_SIMILARITY = 0.65 # Что возвращать, если это первое фото
THRESHOLD = 0.65 # Если similarity >= THRESHOLD, студент считается собой
COMPARE_WITH_LAST_LIMIT = 5 # Со сколькими последними сравнивать

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Embedding(db.Model):
    __tablename__ = 'embeddings'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.BigInteger, nullable=False)
    datedata = db.Column(db.DateTime, nullable=False)
    embedding = db.Column(Vector(512), nullable=False)

    def __repr__(self):
        return f'<Embedding {self.student_id}/{self.datedata}>'


with app.app_context():
    db.session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    db.session.commit()
    db.create_all()
    db.session.execute(text("""
        CREATE INDEX IF NOT EXISTS embeddings_embedding_idx
        ON embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """))
    db.session.commit()


def get_all_student_ids():
    return [row[0] for row in db.session.query(Embedding.student_id).distinct().all()]


def contains_any_embeddings(student_id):
    return bool(db.session.query(
        db.session.query(Embedding).filter(Embedding.student_id == student_id).exists()
    ).scalar())


def delete_student_embeddings(student_id):
    db.session.query(Embedding) \
        .filter(Embedding.student_id == student_id) \
        .delete(synchronize_session=False)
    db.session.commit()


def calculate_similarity_with_recent_embeddings(student_id, embedding, limit):
    results = db.session.query(Embedding.embedding) \
        .filter(Embedding.student_id == student_id) \
        .order_by(Embedding.datedata.desc()) \
        .limit(limit).all()
    similarity = [calculate_embedding_similarity(np.array(row[0]), embedding) for row in results]
    if similarity:
        return float(sum(similarity) / len(similarity)), True
    return DEFAULT_SIMILARITY, False


def add_embedding_and_clear_overflow(student_id, datedata, embedding, limit):
    new_embedding = Embedding(student_id=student_id, datedata=datedata, embedding=embedding)
    db.session.add(new_embedding)
    subquery = (
        db.session.query(Embedding.id)
        .filter(Embedding.student_id == student_id)
        .order_by(Embedding.datedata.desc())
        .limit(limit)
    )
    db.session.query(Embedding) \
        .filter(Embedding.student_id == student_id) \
        .filter(Embedding.id.notin_(subquery)) \
        .delete(synchronize_session=False)
    db.session.commit()


@app.route('/students', methods=['GET'])
def students_get():
    if 'student_id' not in request.args:
        return  jsonify({'student_ids': get_all_student_ids()}), 200
    if contains_any_embeddings(request.args['student_id']):
        return '', 200
    return '', 404


@app.route('/students', methods=['DELETE'])
def students_delete():
    if 'student_id' not in request.args:
        return jsonify({'error': "Query parameter 'student_id' are required"}), 400
    delete_student_embeddings(request.args['student_id'])
    return '', 200

@app.route('/students', methods=['POST'])
def check():
    if 'student_id' not in request.form:
        return jsonify({'error': "Form field 'student_id' are required"}), 400
    if 'file' not in request.files:
        return jsonify({'error': "Form file 'file' are required"}), 400

    student_id = request.form['student_id']
    file = request.files['file'].read()

    try:
        embedding = process_image(file)
    except ValueError as ve:
        match ve.args[0]:
            case 0:
                return jsonify({'error': "Bad image file"}), 400
            case 404:
                return jsonify({'error': "Face is not recognized"}), 404
            case 409:
                return jsonify({'error': "More than one face"}), 409

    similarity, checked = calculate_similarity_with_recent_embeddings(student_id, embedding, COMPARE_WITH_LAST_LIMIT)
    is_match = bool(similarity >= THRESHOLD)

    if not is_match:
        return jsonify({ "checked": checked, "is_match": is_match, "similarity": similarity })
    
    datedata = datetime.datetime.now(datetime.timezone.utc)
    add_embedding_and_clear_overflow(student_id, datedata, embedding, COMPARE_WITH_LAST_LIMIT)

    return jsonify({ "checked": checked, "is_match": is_match, "similarity": similarity })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
