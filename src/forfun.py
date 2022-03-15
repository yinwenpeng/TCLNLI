
import statistics
values=[ -0.0539000000000005,-2.966699999999996,-2.4633000000000003,-1.0206,-4.253499999999995,-0.10000000000000853,0.0,1.5299,-0.3791999999999973,-2.2299000000000007]

def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


if __name__ == "__main__":
    computer_mean_std(values)
