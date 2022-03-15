
import statistics
values=[-10.490200000000002,-5.695399999999999,-6.503,13.5266,0.5804,-73.08260000000001,0.835,-7.649100000000001,0.0,4.897499999999999,-2.4065,-64.5614,-0.5574999999999992,-1.6600000000000001,-10.559199999999999]
def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


if __name__ == "__main__":
    computer_mean_std(values)
