
import statistics
values=[ -0.06559999999999988,-4.429,-5.602799999999999,1.5379,15.2903,-8.097399999999999,4.248400000000001,-6.4183,7.766099999999999,-10.3627]

def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


if __name__ == "__main__":
    computer_mean_std(values)
