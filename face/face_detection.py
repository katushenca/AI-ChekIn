import numpy as np
from numpy import ndarray
import cv2

ATTEMPTS = 3

class FaceDetection:
    @staticmethod
    def detect_face(img_bytes: bytes) -> None | ndarray:
        """ Возвращает ndarray обрезанное лицо если нашел, или None если нет лиц или > 1"""
        np_array_image = FaceDetection.get_nparray_from_bytes_image(img_bytes)
        if np_array_image is None:
            return None
        else:
            exception = None
            for i in range(ATTEMPTS):
                try:
                    minx = int(np_array_image.shape[0] * 0.4)
                    miny = int(np_array_image.shape[1] * 0.4)
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray_image = cv2.cvtColor(np_array_image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=5,
                                                          minSize=(minx, miny))
                    if len(faces) == 0:
                        raise ValueError()
                    faces = FaceDetection.filter_duplicates(faces)
                    if len(faces) > 1:
                        raise ValueError()
                    x, y, w, h = faces[0]
                    cropped_face = np_array_image[y:y + h, x:x + w]
                    return cropped_face
                except ValueError as e:
                    exception = e
            if exception is not None:
                return None

    @staticmethod
    def get_nparray_from_bytes_image(img_bytes: bytes) -> None | ndarray:
        if img_bytes is None or len(img_bytes) == 0:
            return None
        np_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        rgb_array = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)
        return rgb_array

    @staticmethod
    def filter_duplicates(faces, iou_threshold=0.5):
        """ Фильтрует дублирующиеся рамки лиц """

        def calculate_iou(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2

            # Преобразуем в формат (x1, y1, x2, y2)
            box1 = (x1, y1, x1 + w1, y1 + h1)
            box2 = (x2, y2, x2 + w2, y2 + h2)

            # Координаты пересечения
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])

            # Если рамки не пересекаются
            if x_right < x_left or y_bottom < y_top:
                return 0.0

            # Площадь пересечения
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # Площади рамок
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

            # Площадь объединения
            union_area = box1_area + box2_area - intersection_area

            # IoU
            return intersection_area / union_area

        # Оставляем только уникальные рамки
        filtered_faces = []
        for i in range(len(faces)):
            keep = True
            for j in range(len(filtered_faces)):
                iou = calculate_iou(faces[i], filtered_faces[j])
                if iou > iou_threshold:
                    keep = False
                    break
            if keep:
                filtered_faces.append(faces[i])

        return filtered_faces
