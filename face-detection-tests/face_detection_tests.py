import unittest
from face import face_detection
import os

def read_images(directory):
    images = []
    for filename in os.listdir(directory):
        input_path = os.path.join(directory, filename)
        with open(input_path, 'rb') as f:
            img_bytes = f.read()
            images.append((img_bytes, filename))
    return images


class TestIdentifyFace(unittest.TestCase):
    def test_identify_face_no_faces(self):
        for image, filename in read_images("TestPhotos/TestIdentifyFace_noFaces"):
            result = face_detection.FaceDetection.detect_face(image)
            self.assertEqual(result, None, msg=f"Failed on file: {filename}")

    def test_identify_face_more_than_one_faces(self):
        for image, filename in read_images("TestPhotos/TestIdentifyFace_More_Than_One_Face"):
            result = face_detection.FaceDetection.detect_face(image)
            self.assertEqual(result, None, msg=f"Failed on file: {filename}")

    def test_identify_face_true(self):
        for image, filename in read_images("TestPhotos/TestIdentifyFace_one_face"):
            result = face_detection.FaceDetection.detect_face(image)
            self.assertIsNotNone(result, msg=f"Face not detected in file: {filename}")