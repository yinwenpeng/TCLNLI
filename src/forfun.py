
import statistics
values=[-13.2596,-7.4849,-21.001,11.033900000000001,3.8212,-67.55730000000001,-7.5925,2.2527,-14.8412,-2.1494999999999997,2.6795999999999998,0.6965000000000003]
def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


if __name__ == "__main__":
    computer_mean_std(values)
