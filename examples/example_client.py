import requests

endpoint = "http://localhost:8000/students"


def make_get_request(student_id=None):
    query = (endpoint + '?student_id=' + str(student_id)) if student_id != None else endpoint
    response = requests.get(query)
    print(response.status_code, response.text)


def make_post_request(student_id, file_path):
    files = {'file': open(file_path, 'rb')}
    data = {'student_id': student_id}
    response = requests.post(endpoint, files=files, data=data)
    print(response.status_code, response.text)


def make_delete_request(student_id):
    query = (endpoint + '?student_id=' + str(student_id)) if student_id != None else endpoint
    response = requests.delete(query)
    print(response.status_code, response.text)


# curl -X GET "http://localhost:8000/students"
make_get_request()
# 200 {"student_ids":[1,3,2]} - возвращает список id студентов, о которых есть информация

# curl -X GET "http://localhost:8000/students?student_id=1"
make_get_request(student_id=1)
# 200 - если информация есть
# 404 - если информации нет

# curl -X DELETE "http://localhost:8000/students?student_id=1"
make_delete_request(student_id=1)
# 200 - информация о студенте полностью удалена

# curl -X POST -F "student_id=1" -F "file=@images/person1_0.jpg" "http://localhost:8000/students"
make_post_request(student_id=1, file_path='images/person1_0.jpg')
# 200 {"checked":false,"is_match":true,"similarity":0.65}
# checked - показывает, была ли проверка (checked=false бывает только при первом запросе на студента, когда новое фото не с чем сравнивать)
# is_match - показывает, совпадает ли студент на фото с последними для этого student_id
# similarity - показывает, насколько уверена нейронка (значения примерно между -1 и 1), is_match == (similarity >= 0.65)

make_post_request(student_id=1, file_path='images/person1_1.jpg')
# 200 {"checked":true,"is_match":true,"similarity":0.8287}

make_post_request(student_id=1, file_path='images/person1_2.jpg')
# 200 {"checked":true,"is_match":false,"similarity":0.5253}

make_post_request(student_id=1, file_path='images/no_face.jpg')
# 404 - на фото нет лиц

make_post_request(student_id=1, file_path='images/two_face.jpg')
# 409 - на фото несколько лиц
