
'''
{"Title": "Writing questions that require tracking entity references", "Prompt": "Write a question about the passage.", "Definition": "In this task, you're given passages that contain mentions of names of people, places, or things. Some of these mentions refer to the same person, place, or thing. Your job is to write questions that evaluate one's understanding of such references. Good questions are expected to link pronouns (she, her, him, his, their, etc.) or other mentions to people, places, or things to which they may refer. ", "Things to Avoid": "1. Avoid questions that can be answered correctly without actually understanding the paragraph. 2. Avoid questions that do not link phrases referring to the same entity, 3. Avoid questions that have multiple answers. \n", "Emphasis & Caution": "1. For each of your questions the answer should be one or more phrases in the paragraph, 2. The answer for each question should be unambiguous. ", "Instances": [{"input": "Passage: The ear
'''
import json

f = open('/home/tup51337/dataset/Natural-Instructions/subtask001_quoref_question_generation.json')
data = json.load(f)
print(len(data))
# print(data["Title"])
# print(data["Prompt"])
# print(data["Definition"])
#
# print(data["Things to Avoid"])
# print(data["Emphasis & Caution"])
# print(data["Positive examples"])
# print(data["Negative Examples"])
'''
Definition
Prompt
Title
Emphasis & Caution
Instances
Examples
Things to Avoid
'''
# for key, value in data.items():
#     print(key)
# print(data["Examples"])
for key, value in data["Examples"]["Negative Examples"].items():
    print(key)
f.close()
