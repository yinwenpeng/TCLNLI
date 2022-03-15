import os
import statistics
from datasets import load_dataset

values=[67.7,12.125300000000001,9.452200000000001,15.14630000000001,28.371199999999998,-0.5423,-2.253099999999999,-4.6477,-5.780799999999999,-19.39479999999999,1.3583000000000007,7.5634999999999994,-22.6223,-5.3564,-5.183400000000001]
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
