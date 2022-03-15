import os
import statistics
from datasets import load_dataset

values=[4.184700000000001, -2.7798000000000003, 7.3564, 4.567100000000001, 13.045299999999997,
8.7909, -5.2681000000000004, -3.3360000000000003, -5.879599999999996, 1.8386000000000005,
1.0866999999999996, -9.242199999999997, 13.6252, -1.0994000000000002, 61.2571, 15.3714,
14.5382, 90.6357, 1.3307000000000002, 17.6553, -5.079999999999998, -1.1705000000000005,
5.313300000000001, 8.0836, 5.9757, 5.6560999999999995, 6.8419, 0.36439999999999984, -9.347199999999999,
29.2845, 1.0986000000000011,-7.0457, -6.6509, -8.835700000000001, -5.147, -12.017199999999999,
-17.297600000000003, -2.6533, 48.105, 6.1835, -9.3921, 21.9757, -9.827300000000005, 3.2064, 24.3153,
60.7346, -2.967099999999995, 44.1239, -8.8108, -3.1669, -4.3369, 43.9, 21.700000000000003, -1.4148000000000005,
0.7176999999999998, -3.1311000000000004]
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
