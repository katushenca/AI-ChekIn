import time
import os
import process


def get_all_path_pairs(dir: str):
    """По заданному к директории пути возвращает последовательность пар
    лежащих в ней файлов"""
    files = [os.path.join(dir, f) for f in os.listdir(dir) if
             os.path.isfile(os.path.join(dir, f))]
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            yield files[i], files[j]


def get_accuracy_and_av_time(path_to_dataset: str, threshold: float = 0.65):
    """По заданному к датасету пути возвращает accuracy и average time
    У одинаковых людей первые два символа в названии файла одинаковые.
    Если у объекта несколько фото, у него первый символ в имени буква."""
    true_count = 0
    total_count = 0
    total_time = 0
    for file1, file2 in get_all_path_pairs(path_to_dataset):
        start = time.time()
        emb1 = process.get_face_embedding(file1)
        emb2 = process.get_face_embedding(file2)
        score = process.compare_faces(emb1, emb2, threshold)[0]
        end = time.time()
        file1_name = os.path.basename(file1)
        file2_name = os.path.basename(file2)
        if score is None:
            continue
        if file1_name[0].isalpha() and file1_name[:2] == file2_name[:2] and score >= threshold:
            true_count += 1
        elif score < threshold:
            true_count += 1

        total_count += 1
        total_time += end - start

    accuracy = true_count / total_count if total_count > 0 else 0
    average_time = total_time / total_count if total_count > 0 else 0

    return accuracy, average_time
