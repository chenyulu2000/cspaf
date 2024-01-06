import json

file_path = "/home/data/visdial_v1.0_test-std/visdial_1.0_val.json"

with open(file_path, 'r') as file:
    data = json.load(file)['data']
print(type(data))

results = []
questions = data['questions']
dialogs = data['dialogs']
file_path = "attention_map_vis/question.txt"

with open(file_path, 'a') as file:
    for item in dialogs:
        id = item['image_id']
        question_list = [questions[content['question']] for content in item['dialog']]
        question = ''
        file.write(f'{id}\n')
        for content in item['dialog']:
            question = question + questions[content['question']] + '\n'
            file.write(questions[content['question']] + '\n')
        results.append({'id': id, 'ques': question})
        file.write('\n')

file_path = "attention_map_vis/question.json"
with open(file_path, 'w') as file:
    json.dump(results, file)
