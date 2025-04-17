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


class CheckLog(db.Model):
    __tablename__ = 'checkLogs'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.BigInteger, nullable=False)
    datedata = db.Column(db.DateTime, nullable=False)
    status_code = db.Column(db.Integer, nullable=False)
    is_match = db.Column(db.Boolean, nullable=False)
    similarity = db.Column(db.Numeric(5, 3), nullable=False)
    
    def __repr__(self):
        return f'<CheckLog {self.student_id}/{self.datedata} {self.status_code}>'


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
        return round(float(sum(similarity) / len(similarity)), 3), True
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


def add_check_log(student_id, datedata, status_code, is_match=False, similarity=0):
    new_log = CheckLog(student_id=student_id, datedata=datedata,
                       status_code=status_code, is_match=is_match, similarity=similarity)
    db.session.add(new_log)
    db.session.commit()


def create_metrics(dateFrom=None, dateTo=None):
    filters = []
    if dateFrom:
        filters.append(CheckLog.datedata >= dateFrom)
    if dateTo:
        filters.append(CheckLog.datedata <= dateTo)

    return db.session.query(
        db.func.count().label('total_count'),
        db.func.count(db.case((CheckLog.status_code == 200, 1))).label('count_status_200'),
        db.func.count(db.case((CheckLog.status_code == 404, 1))).label('count_status_404'),
        db.func.count(db.case((CheckLog.status_code == 409, 1))).label('count_status_409'),
        db.func.count(db.case(((CheckLog.status_code == 200) & (CheckLog.is_match == True), 1))).label('count_is_match_true'),
        db.func.count(db.case(((CheckLog.status_code == 200) & (CheckLog.is_match == False), 1))).label('count_is_match_false'),
        db.func.avg(db.case(((CheckLog.status_code == 200) & (CheckLog.is_match == True), CheckLog.similarity))).label('avg_similarity_is_match_true'),
        db.func.avg(db.case(((CheckLog.status_code == 200) & (CheckLog.is_match == False), CheckLog.similarity))).label('avg_similarity_is_match_false')
    ).filter(*filters).first()._asdict()


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
    datedata = datetime.datetime.now(datetime.timezone.utc)

    try:
        embedding = process_image(file)
    except ValueError as ve:
        match ve.args[0]:
            case 0:
                return jsonify({'error': "Bad image file"}), 400
            case 404:
                add_check_log(student_id, datedata, 404)
                return jsonify({'error': "Face is not recognized"}), 404
            case 409:
                add_check_log(student_id, datedata, 409)
                return jsonify({'error': "More than one face"}), 409

    similarity, checked = calculate_similarity_with_recent_embeddings(student_id, embedding, COMPARE_WITH_LAST_LIMIT)
    is_match = bool(similarity >= THRESHOLD)

    if not checked or is_match:
        add_embedding_and_clear_overflow(student_id, datedata, embedding, COMPARE_WITH_LAST_LIMIT)

    if checked:
        add_check_log(student_id, datedata, 200, is_match, similarity)

    return jsonify({ "checked": checked, "is_match": is_match, "similarity": similarity })


@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(create_metrics())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
