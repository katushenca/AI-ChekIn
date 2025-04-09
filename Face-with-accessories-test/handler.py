from face.face_detection import FaceDetection
import os
import cv2


def images_test(directory):
    good_count = 0
    all_count = 0
    for filename in os.listdir(directory):
        print(filename, end=":\t")
        input_path = os.path.join(directory, filename)
        with open(input_path, 'rb') as f:
            image = f.read()
        result = FaceDetection.detect_face(image)
        print("Detected" if result is not None else "-")
        if result is not None:
            good_count += 1
            cv2.imwrite(os.path.join("Result", filename), result)
        all_count += 1
    print(f"{good_count}/{all_count}")
    print(f"{good_count * 100 / all_count}%")


if __name__ == "__main__":
    images_test("Test")
