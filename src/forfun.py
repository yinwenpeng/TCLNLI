
import statistics
values=[ 69.5813,69.5813,61.4532,71.3054,65.7635,76.9704,70.3202,71.4286,72.2906,70.4433]

def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    return str(average)+'/'+str(res)


if __name__ == "__main__":
    computer_mean_std(values)
