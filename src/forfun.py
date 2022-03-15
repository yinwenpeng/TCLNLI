
import statistics
values=[-5.087600000000002,2.3514000000000017,0.8203999999999994,0.0081,7.7229,-5.852700000000013,1.9043,0.06409999999999982,-0.7000000000000002,7.0036000000000005,3.9014,3.201899999999995,0.25810000000000066,-0.06170000000000009,5.913399999999999]
def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    print( str(average)+'/'+str(res))


if __name__ == "__main__":
    computer_mean_std(values)
