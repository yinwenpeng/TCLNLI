
import json
import os
import codecs
import jsonlines
import csv

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

def MNLI_2_csvformat(filename):
    prefix = filename[:filename.find('.jsonl')]
    csvfile_name = prefix+'.csv'
    # with open(csvfile_name, 'w', 'utf-8', newline='') as csvfile:
    csvfile = codecs.open(csvfile_name, 'w', 'utf-8')
    fieldnames = ['premise', 'hypothesis', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()



    with jsonlines.open(filename) as f:
        for line in f.iter():
            writer.writerow({'premise': line['sentence1'].strip(), 'hypothesis': line['sentence2'].strip(), 'label': line['gold_label'].strip()})
    f.close()
    csvfile.close()
    print('write over')

if __name__ == '__main__':
    # load_a_single_file('/home/tup51337/dataset/Natural-Instructions/test_original_paper/subtask052_multirc_identify_bad_question.json')
    MNLI_2_csvformat('/home/tup51337/dataset/MNLI/multinli_1.0_dev_matched.jsonl')
