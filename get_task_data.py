from read_task import *
from cal_path import *

def get_task_data(Task,init_strength,min_strength,init_node,path_number):
    Return_data=[]
    Des_var=[]
    Paths=[]
    for task in Task:
        return_data, des_var = read_task(task, min_strength)
        paths = cal_path(task, init_strength, min_strength, init_node, path_number, des_var)
        Return_data.append(return_data)
        Des_var.append(des_var)
        Paths.append(paths)
    return([Return_data,Des_var,Paths])