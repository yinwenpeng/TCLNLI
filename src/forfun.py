import os
import statistics
from datasets import load_dataset

values=[1.7782,12.9544,0.0,0.6263000000000001,-0.6804999999999986,4.473700000000001,2.7673000000000005,29.4739,-29.861200000000004]
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
