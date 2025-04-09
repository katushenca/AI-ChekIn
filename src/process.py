from identify_face import compare_faces, get_face_embedding

def process_image(file):
    return get_face_embedding(file)

def calculate_embedding_similarity(emb1, emb2):
    return compare_faces(emb1, emb2)[0]
