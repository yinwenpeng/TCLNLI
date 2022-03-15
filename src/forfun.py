
import statistics
values=[ -0.15050000000000097,-0.2177,1.5497999999999998,-13.3503,-0.25850000000000006,12.013699999999998,-0.4667000000000012,-3.403,-0.7550000000000001,-10.6201,-1.4822,1.2947999999999986,-20.617000000000004,-1.600200000000001,4.178000000000001,10.9512,0.7000000000000028,-1.2999999999999972,3.7773999999999996]

def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


if __name__ == "__main__":
    computer_mean_std(values)
