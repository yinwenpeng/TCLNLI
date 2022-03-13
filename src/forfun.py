
import statistics
values=[ 9.5696,9.6787,9.7954,9.3839,9.5478,9.808,9.7019,9.8682,9.997,9.8404]

def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


if __name__ == "__main__":
    computer_mean_std(values)
