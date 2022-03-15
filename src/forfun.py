import os
import statistics
from datasets import load_dataset

values=[5.2949,-2.5,5.3547,-5.4225,25.071199999999997,-1.3740999999999994,4.6,9.833300000000001,10.6846,4.4529,12.027199999999999,-0.30000000000001137,-7.280199999999994,-4.5145,0.1180000000000021,9.084399999999999]
def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


def file_rows(filepath):
    raw_datasets = load_dataset("csv", data_files={'train':filepath})
    print(len(raw_datasets['train']))
if __name__ == "__main__":
    computer_mean_std(values)
    # file_rows('/home/tup51337/dataset/Natural-Instructions/TCLNLI_split/all_task_neg_instruction_examples_in_CSV/subtask017_mctaco_wrong_answer_generation_frequency.neg.csv')
    # file_rows('/home/tup51337/dataset/Natural-Instructions/TCLNLI_split/all_task_neg_instruction_examples_in_CSV/subtask061_ropes_answer_generation.neg.csv')
