
import statistics
values=[ 22.3411,25.781,25.7483,26.5305,23.3101,26.2254,24.577]

def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


if __name__ == "__main__":
    computer_mean_std(values)
