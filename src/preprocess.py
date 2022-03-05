
import json
import os
import codecs
import jsonlines
import csv





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


def load_a_single_json_file(fil):
    # f = codecs.open(path+fil, 'r', 'utf-8')
    f = open(fil)
    data = json.load(f)

    title_str = data["Title"].strip()
    prompt_str = data["Prompt"].strip()
    def_str = data["Definition"].encode('utf-8').strip()
    avoid_str = data["Things to Avoid"].strip()
    caution_str = data["Emphasis & Caution"].strip()

    for id, ex in enumerate(data["Examples"]["Positive Examples"]):
        for key, value in ex.items():
            print(key)
        exit(0)
    # for ex in enumerate(data["Examples"]["Negative Examples"]):
    #     for key, value in ex.items():
    #         print(key)

    '''instances'''
    for id, instance in enumerate(data["Instances"]):
        for key, value in instance.items():
            print(key, ' --> ', value)
        exit(0)

    f.close()

def convert_data_inito_csv(folder):
    #for each file in the folder, convert to csv file with two columns (input, output)

    file_set = set(os.listdir(folder))
    for fil in file_set:
        prefix = fil[:fil.find('.json')]
        csvfile_name = prefix+'.csv'




if __name__ == '__main__':
    load_a_single_json_file('/home/tup51337/dataset/Natural-Instructions/test_original_paper/subtask052_multirc_identify_bad_question.json')
    # MNLI_2_csvformat('/home/tup51337/dataset/MNLI/multinli_1.0_dev_mismatched.jsonl')
