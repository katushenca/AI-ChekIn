import time
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# Initialize face analysis model
app = FaceAnalysis(name='buffalo_l', providers=[
    'CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=0)  # ctx_id=-1 for CPU, 0 for GPU


def get_face_embedding(file):
    """Extract face embedding from an image"""
    file_array = np.frombuffer(file, dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(0) # проблема с изображением

    faces = app.get(img)

    if len(faces) < 1:
        raise ValueError(404) # нет лица
    if len(faces) > 1:
        raise ValueError(409) # много лиц
    bbox = faces[0].bbox.astype(int)
    x1, y1, x2, y2 = bbox
    cropped_face = img[y1:y2, x1:x2]
    return faces[0].embedding


def get_face_embedding2(file):
    """В отличие от прошлой версии теперь выбирается набольшее по площади и
    наиболее близкое к центру лицо"""
    file_array = np.frombuffer(file, dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(0)  # проблема с изображением

    h, w, _ = img.shape
    center = np.array([w / 2, h / 2])

    def score(face):
        x1, y1, x2, y2 = face.bbox
        face_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        area = (x2 - x1) * (y2 - y1)
        dist_to_center = np.linalg.norm(face_center - center)

        # Нормализация: центр ближе — лучше, площадь больше — лучше
        return area / (1 + dist_to_center)

    faces = app.get(img)
    if len(faces) < 1:
        raise ValueError(404)  # нет лица
    if len(faces) > 1:
        raise ValueError(409)  # много лиц

    return max(faces, key=score).embedding

def compare_faces(emb1, emb2, threshold=0.65):
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold


def print_result(image1_path: str, image2_path: str):
    # Get embeddings
    start = time.time()
    emb1 = get_face_embedding(image1_path)
    emb2 = get_face_embedding(image2_path)

    similarity_score, is_same_person = compare_faces(emb1, emb2)

    return similarity_score
