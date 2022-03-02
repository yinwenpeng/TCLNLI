
import json
import os
import codecs

path = '/home/tup51337/dataset/Natural-Instructions/'
files = set(os.listdir(path))
files.remove('splits.txt')
print(len(files))

def load_a_single_file(fil):
    # f = codecs.open(path+fil, 'r', 'utf-8')
    f = open(fil)
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
            print(key, ' --> ', value)
            exit(0)

    f.close()
