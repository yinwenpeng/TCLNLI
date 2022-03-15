
import statistics
values=[-0.06559999999999988,-4.429,-5.602799999999999,1.5379,15.2903,-8.097399999999999,4.248400000000001,-37.6847,-6.4183,39.5333,7.766099999999999,-10.3627]
def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


def file_rows(filepath):
    cmd = 'wc -l '+filepath
    print(os.system(cmd))
if __name__ == "__main__":
    # computer_mean_std(values)
    file_rows('/home/tup51337/dataset/Natural-Instructions/TCLNLI_split/all_task_neg_instruction_examples_in_CSV/subtask017_mctaco_wrong_answer_generation_frequency.neg.csv')
