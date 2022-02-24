

import json

f = open('/home/tup51337/dataset/Natural-Instructions/subtask061_ropes_answer_generation.json')
data = json.load(f)
print(len(data))
print(data["Contributors"])
f.close()
