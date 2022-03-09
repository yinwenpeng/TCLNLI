
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

def load_instruction_from_json_data(fil):

    '''
    return: instruction_text_part, instruction_pos_example_part, pos_examples_as_X&Y_format
    '''
    # f = codecs.open(path+fil, 'r', 'utf-8')
    f = open(fil)
    data = json.load(f)

    title_str = data["Title"].strip()
    # print('title_str:', title_str)
    prompt_str = data["Prompt"].strip()
    # print('prompt_str:', prompt_str)
    def_str = data["Definition"].encode('utf-8').decode('utf-8').strip()
    # print('def_str:', def_str)
    avoid_str = data["Things to Avoid"].strip()
    # print('avoid_str:', avoid_str)
    caution_str = data["Emphasis & Caution"].strip()
    # print('caution_str:', caution_str)
    INSTRUCTION = '[Title] '+title_str+' [Prompt] '+prompt_str+' [Definition] '+def_str+' [Avoid] '+avoid_str+' [Caution] '+caution_str
    INSTRUCTION_text_part = INSTRUCTION.strip().replace('\n', ' ')


    pos_ex_tuple_list = []
    POS = ''
    for id, ex in enumerate(data["Examples"]["Positive Examples"]):
        POS+='[POS'+str(id+1)+'] '
        pos_ex_tuple_list.append((ex['input'].replace('Question:', '[Question]').replace('\n', ' ').strip(), ex['output'].strip()))
        for key, value in ex.items():
            POS+='['+key.strip()+'] '+value.strip()+' '

    INSTRUCTION_pos_example_part = POS.strip().replace('\n', ' ')

    neg_ex_tuple_list = []
    NEG = ''

    # print('data["Examples"]["Negative Examples"]:', data["Examples"]["Negative Examples"])
    if data["Examples"]["Negative Examples"] != ['-']
        for id, ex in enumerate(data["Examples"]["Negative Examples"]):
            NEG+='[NEG'+str(id+1)+'] '
            # print('ex:>>', ex)
            if (type(ex) is dict):

                # print('ex input:>>', ex['input'].encode('utf-8').decode('utf-8'))
                # print('ex output:>>', ex['output'])
                neg_ex_tuple_list.append((ex['input'].replace('Question:', '[Question]').replace('\n', ' ').strip(), ex['output'].strip()))
                for key, value in ex.items():
                    NEG+='['+key.strip()+'] '+value.strip()+' '

    INSTRUCTION_neg_example_part = NEG.strip().replace('\n', ' ')

    f.close()
    return INSTRUCTION_text_part, INSTRUCTION_pos_example_part, pos_ex_tuple_list, INSTRUCTION_neg_example_part, neg_ex_tuple_list

def load_a_single_json_file(fil):
    # f = codecs.open(path+fil, 'r', 'utf-8')
    f = open(fil)
    data = json.load(f)

    INSTRUCTION_text, POS_ex, _, _, _ = load_instruction_from_json_data(fil)

    '''instances'''
    returned_tuple_list = []
    for id, instance in enumerate(data["Instances"]):
        input = ''
        output = ''
        for key, value in instance.items():
            # print('key: ', key)
            # print('value: ', value)
            if key.strip() != 'output':
                input+='['+key+'] '+value.strip()+' '
            else:
                output = ' '.join(value).strip()
        '''put input at the beginning of the long instruction text'''
        input = input.replace('\n', ' ').replace('Question:', '[Question]').strip()
        X = input+' '+INSTRUCTION_text+' '+POS_ex
        Y = output
        # print('X: ', X)
        # print('Y: ', Y)
        # exit(0)
        returned_tuple_list.append((X, Y))

    f.close()
    return returned_tuple_list

def convert_data_into_csv(input_folder, output_folder, overall_output_file_name=None):
    #for each file in the folder, convert to csv file with two columns (input, output)

    if overall_output_file_name is not None:
        overall_csvfile = codecs.open(overall_output_file_name, 'w', 'utf-8')
        fieldnames = ['input', 'output']
        overall_writer = csv.DictWriter(overall_csvfile, fieldnames=fieldnames)
        overall_writer.writeheader()

    file_set = set(os.listdir(input_folder))
    for fil in file_set:
        prefix = fil[:fil.find('.json')]
        csvfile_name = prefix+'.csv'
        write_csv_filename = output_folder+'/'+csvfile_name
        csvfile = codecs.open(write_csv_filename, 'w', 'utf-8')
        fieldnames = ['input', 'output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


        pair_list = load_a_single_json_file(input_folder+'/'+fil)
        for pair in pair_list:
            writer.writerow({'input': pair[0].strip(), 'output': pair[1].strip()})
            if overall_output_file_name is not None:
                overall_writer.writerow({'input': pair[0].strip(), 'output': pair[1].strip()})
        csvfile.close()
        print(fil, ' convert over...')
    if overall_output_file_name is not None:
        overall_csvfile.close()
    print('Done!')


def merge_test_tasks_into_one_category(path, filelist, output_file):
    csvfile = codecs.open(path+output_file, 'w', 'utf-8')
    fieldnames = ['input', 'output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for single_fil in filelist:
        read_fil = codecs.open(path+single_fil, 'r', 'utf-8')
        read_file = csv.DictReader(read_fil)
        for row in read_file:
            dict_row = dict(row)
            writer.writerow({'input': dict_row['input'].strip(), 'output': dict_row['output'].strip()})
        read_fil.close()
    csvfile.close()
    print('merge over')


def generate_training_examples_from_instruction(input_folder, output_folder):
    file_set = set(os.listdir(input_folder))
    for fil in file_set:
        prefix = fil[:fil.find('.json')]
        csvfile_name = prefix+'.csv'
        write_csv_filename = output_folder+'/'+csvfile_name
        csvfile = codecs.open(write_csv_filename, 'w', 'utf-8')
        fieldnames = ['input', 'output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        INSTRUCTION_text, POS_ex_2_text, POS_ex_tuple_list, _, _ = load_instruction_from_json_data(input_folder+'/'+fil)
        for tuple in POS_ex_tuple_list:
            X = tuple[0]+' '+INSTRUCTION_text#+' '+POS_ex_2_text
            Y = tuple[1]
            writer.writerow({'input': X.strip(), 'output': Y.strip()})
        csvfile.close()
        print(fil, ' convert over...')
    print('Done!')



def generate_negative_training_examples_from_instruction(input_folder, output_folder):
    file_set = set(os.listdir(input_folder))
    for fil in file_set:
        prefix = fil[:fil.find('.json')]
        csvfile_name = prefix+'.csv'
        write_csv_filename = output_folder+'/'+csvfile_name
        csvfile = codecs.open(write_csv_filename, 'w', 'utf-8')
        fieldnames = ['input', 'output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        INSTRUCTION_text, _, _, NEG_ex_2_text, NEG_ex_tuple_list = load_instruction_from_json_data(input_folder+'/'+fil)
        for tuple in NEG_ex_tuple_list:
            X = tuple[0]+' '+INSTRUCTION_text#+' '+POS_ex_2_text
            Y = tuple[1]
            writer.writerow({'input': X.strip(), 'output': Y.strip()})
        csvfile.close()
        print(fil, ' convert over...')
    print('Done!')


if __name__ == '__main__':
    # load_a_single_json_file('/home/tup51337/dataset/Natural-Instructions/test_original_paper/subtask052_multirc_identify_bad_question.json')
    # MNLI_2_csvformat('/home/tup51337/dataset/MNLI/multinli_1.0_dev_mismatched.jsonl')
    # convert_data_into_csv('/home/tup51337/dataset/Natural-Instructions/train_original_paper', '/home/tup51337/dataset/Natural-Instructions/train_tasks_csv', '/home/tup51337/dataset/Natural-Instructions/all_training_tasks_in_single_csv.csv')
    # convert_data_into_csv('/home/tup51337/dataset/Natural-Instructions/test_original_paper', '/home/tup51337/dataset/Natural-Instructions/test_tasks_csv', None)

    # test_csv_path = '/home/tup51337/dataset/Natural-Instructions/test_tasks_csv/'
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask002_quoref_answer_generation.csv', 'subtask033_winogrande_answer_generation.csv'], 'AG.csv')
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask003_mctaco_question_generation_event_duration.csv', 'subtask040_qasc_question_generation.csv'], 'QG.csv')
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask005_mctaco_wrong_answer_generation_event_duration.csv', 'subtask008_mctaco_wrong_answer_generation_transient_stationary.csv'], 'IAG.csv')
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask022_cosmosqa_passage_inappropriate_binary.csv', 'subtask052_multirc_identify_bad_question.csv'], 'CF.csv')
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask034_winogrande_question_modification_object.csv', 'subtask045_miscellaneous_sentence_paraphrasing.csv'], 'MM.csv')
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask039_qasc_find_overlapping_words.csv', 'subtask044_essential_terms_identifying_essential_words.csv'], 'VF.csv')

    # generate_training_examples_from_instruction('/home/tup51337/dataset/Natural-Instructions/test_original_paper', '/home/tup51337/dataset/Natural-Instructions/test_tasks_instruction_into_examples_csv')
    #
    # test_csv_path = '/home/tup51337/dataset/Natural-Instructions/test_tasks_instruction_into_examples_csv/'
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask002_quoref_answer_generation.csv', 'subtask033_winogrande_answer_generation.csv'], 'AG.csv')
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask003_mctaco_question_generation_event_duration.csv', 'subtask040_qasc_question_generation.csv'], 'QG.csv')
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask005_mctaco_wrong_answer_generation_event_duration.csv', 'subtask008_mctaco_wrong_answer_generation_transient_stationary.csv'], 'IAG.csv')
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask022_cosmosqa_passage_inappropriate_binary.csv', 'subtask052_multirc_identify_bad_question.csv'], 'CF.csv')
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask034_winogrande_question_modification_object.csv', 'subtask045_miscellaneous_sentence_paraphrasing.csv'], 'MM.csv')
    # merge_test_tasks_into_one_category(test_csv_path, ['subtask039_qasc_find_overlapping_words.csv', 'subtask044_essential_terms_identifying_essential_words.csv'], 'VF.csv')

    generate_negative_training_examples_from_instruction('/home/tup51337/dataset/Natural-Instructions/train_original_paper', '/home/tup51337/dataset/Natural-Instructions/train_tasks_instruction_into_negative_examples_csv')
