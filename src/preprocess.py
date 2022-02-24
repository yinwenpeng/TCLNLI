
'''
{"Title": "Writing questions that require tracking entity references", "Prompt": "Write a question about the passage.", "Definition": "In this task, you're given passages that contain mentions of names of people, places, or things. Some of these mentions refer to the same person, place, or thing. Your job is to write questions that evaluate one's understanding of such references. Good questions are expected to link pronouns (she, her, him, his, their, etc.) or other mentions to people, places, or things to which they may refer. ", "Things to Avoid": "1. Avoid questions that can be answered correctly without actually understanding the paragraph. 2. Avoid questions that do not link phrases referring to the same entity, 3. Avoid questions that have multiple answers. \n", "Emphasis & Caution": "1. For each of your questions the answer should be one or more phrases in the paragraph, 2. The answer for each question should be unambiguous. ", "Instances": [{"input": "Passage: The ear
'''
import json
import os
import codecs

path = '/home/tup51337/dataset/Natural-Instructions/'
files = set(os.listdir(path))
files.remove('splits.txt')
print(len(files))

for fil in files:
    # f = codecs.open(path+fil, 'r', 'utf-8')
    f = open(path+fil)
    data = json.load(f)
    print(data["Title"])
    print(data["Prompt"])
    print(data["Definition"].encode('utf-8'))

    print(data["Things to Avoid"])
    print(data["Emphasis & Caution"])

    # for ex in enumerate(data["Examples"]["Positive Examples"]):
    #     for key, value in ex.items():
    #         print(key)
    # for ex in enumerate(data["Examples"]["Negative Examples"]):
    #     for key, value in ex.items():
    #         print(key)

    for id, instance in enumerate(data["Instances"]):
        for key, value in instance.items():
            print(key)

    f.close()
