from read_xlsx import *
from des_var import desVar
def read_task(Task,min_strength=0.001):
    file_dir = f"./data/task{Task}.xlsx"
    node_data = readData(file_dir)
    return_data = returnData(node_data, node_data.node(), node_data.backnode())
    des_var = desVar(node_data=node_data, min_strength=min_strength)
    return return_data,des_var