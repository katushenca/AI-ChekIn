import time
import os
import process


def get_all_path_pairs(dir: str):
    """По заданному пути к директории возвращает последовательность пар
    лежащих в ней файлов"""
    files = [os.path.join(dir, f) for f in os.listdir(dir) if
             os.path.isfile(os.path.join(dir, f))]
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            yield files[i], files[j]


def get_accuracy_and_av_time(path_to_dataset: str, threshold: float = 0.65):
    """По заданному пути к датасету возвращает accuracy и average time"""
    true_count = 0
    total_count = 0
    total_time = 0
    for file1, file2 in get_all_path_pairs(path_to_dataset):
        start = time.time()
        emb1 = process.get_face_embedding(file1)
        emb2 = process.get_face_embedding(file2)
        score = process.compare_faces(emb1, emb2, threshold)[0]
        end = time.time()
        file1_name = file1.split('\\')[-1]
        file2_name = file1.split('\\')[-1]
        if score is None:
            continue
        if file1_name[0] == file2_name[0] and score >= threshold:
            true_count += 1
        elif score < threshold:
            true_count += 1

        total_count += 1
        total_time += end - start

    return true_count / total_count, total_count / total_count
