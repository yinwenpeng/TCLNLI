import os
import statistics
from datasets import load_dataset

values=[-8.8512,0.0,0.39839999999999876,6.078799999999999,-6.862800000000002,7.899999999999999,1.7104,71.9,1.1514000000000095,42.0612]
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
