

import json

f = open('/home/tup51337/dataset/Natural-Instructions/subtask061_ropes_answer_generation.json')
data = json.load(f)
print(len(data))
# print(data["Instances"])
print(data["Definition"])
print(data["Prompt"])
print(data["Things to Avoid"])
print(data["Emphasis&Caution"])
print(data["Positive Examples"])
print(data["Negative Examples"])
f.close()
