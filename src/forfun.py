
import statistics
values=[ 44.6053,51.2612,41.7361,45.2139,47.1491,42.8476,44.2852]

def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


if __name__ == "__main__":
    computer_mean_std(values)
