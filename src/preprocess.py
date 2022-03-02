
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
    print('Title: ', data["Title"])
    print('Prompt: ', data["Prompt"])
    print('Definition: ', data["Definition"].encode('utf-8'))

    print('Things to Avoid: ', data["Things to Avoid"])
    print('Emphasis & Caution: ', data["Emphasis & Caution"])

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

if __name__ == '__main__':
    load_a_single_file('/home/tup51337/dataset/Natural-Instructions/test_original_paper/subtask052_multirc_identify_bad_question.json')
