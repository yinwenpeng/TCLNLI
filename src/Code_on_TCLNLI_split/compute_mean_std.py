import statistics




def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    return str(average)+'/'+str(res)


def compute_for_dict(value_dict):
    new_dict = {}
    for key, value_list in value_dict.items():
        res = computer_mean_std(value_list)
        new_dict[key] = res
    return new_dict
