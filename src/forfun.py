
import statistics
values=[-24.5,1.0,-0.23550000000000004,-8.3705,-0.10000000000000053,12.155600000000002,-15.0079,-18.5277,1.0076999999999998,-10.9836]

def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


if __name__ == "__main__":
    computer_mean_std(values)
