
import statistics
values=[ 69.5279,91.4393,91.3795,90.2958,89.5093,91.3589,91.3906,86.4336,91.5119]

def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


if __name__ == "__main__":
    computer_mean_std(values)
