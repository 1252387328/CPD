from mtask import mtask
from get_task_data import get_task_data
from concurrent import futures
from openpyxl import Workbook
import statistics

task_list=[1,2,3,4]
init_node = 1
path_number = 20
each_path_size = 10

pop_size = path_number*each_path_size
ce_ratio = 0.00015
risk_ratio=1
init_strength=1
min_strength=0.001
max_iter=100
learn_iter=10
is_Min_Max=True
ts1=1.6
tran_num=int(path_number*each_path_size/20)
db1="./db/cal_max_node.db"
db2='./db/cal_next_node.db'
db3='./db/pf.db'

init_node_list=[1,13,26,28,36,82,87,96]
seed_list=[1251,1240,1253,1279,1265,1311,1268,1263]

task_data=get_task_data(task_list,init_strength,min_strength,init_node,path_number)


m_task=mtask(task_list,task_data,init_node,each_path_size,learn_iter,pop_size,ce_ratio,init_strength,risk_ratio,is_Min_Max,ts1,tran_num,db1,db2)



task_data = get_task_data(task_list, init_strength, min_strength, init_node, path_number)
m_task = mtask(task_list, task_data, init_node, each_path_size, learn_iter, pop_size, ce_ratio, init_strength,
               risk_ratio, is_Min_Max, ts1, tran_num, db1, db2)

task_data = get_task_data(task_list, init_strength, min_strength, init_node, path_number)
m_task = mtask(task_list, task_data, init_node, each_path_size, learn_iter, pop_size, ce_ratio, init_strength,
                   risk_ratio, is_Min_Max, ts1, tran_num, db1, db2)
def nsga2(init_node,max_iter,learn_iter):
    task_data = get_task_data(task_list, init_strength, min_strength, init_node, path_number)
    m_task = mtask(task_list, task_data, init_node, each_path_size, learn_iter, pop_size, ce_ratio, init_strength,
                   risk_ratio, is_Min_Max, ts1, tran_num, db1, db2)

    for i in range(int(max_iter/learn_iter)-1):
        m_task.run()
        m_task.tran()
    pop_list,HV_list,PD_list,IGD_list=m_task.run(cal_HV=True,cal_PD=True,cal_IGD=True)
    print(pop_list)
    return HV_list,PD_list,IGD_list



def mulcalnsga2(Problem,ws_sheet,maxtime,max_iter,learn_iter):
    init_node_list = [12,17,1,47]
    print(f"res of CPD{Problem}")
    init_node = init_node_list[Problem - 1]
    init_node_list = [init_node] * maxtime
    max_iter_list = [max_iter] * maxtime
    learn_iter_list = [learn_iter] * maxtime
    with futures.ProcessPoolExecutor(max_workers=16) as executor:
        res=executor.map(nsga2, init_node_list, max_iter_list, learn_iter_list)
    res = list(res)
    a_res = [x[0] for x in res]
    b_res = [x[1] for x in res]
    c_res = [x[2] for x in res]
    print(a_res)
    a_mean=[]
    b_mean=[]
    a_stdev=[]
    b_stdev=[]
    hv_all=[[] for i in range(len(a_res[0]))]
    pd_all=[[] for i in range(len(b_res[0]))]
    for i in range(len(a_res[0])):
        hv_all[i]=[x[i] for x in a_res]
        pd_all[i]=[x[i] for x in b_res]
    # print(hv_all,pd_all)
    for i in range(len(a_res[0])):
        a_mean.append(statistics.mean(hv_all[i]))
        b_mean.append(statistics.mean(pd_all[i]))
        a_stdev.append(statistics.stdev(hv_all[i]))
        b_stdev.append(statistics.stdev(pd_all[i]))
    igd_all = [[] for i in range(len(c_res[0]))]
    for i in range(len(c_res[0])):
        igd_all[i] = [x[i] for x in c_res]
    igd_mean = []
    igd_stdev = []
    for i in range(len(c_res[0])):
        igd_mean.append(statistics.mean(igd_all[i]))
        igd_stdev.append(statistics.stdev(igd_all[i]))

    base_row = (Problem - 1) * 3 + 1  # 每个Problem占3行，起始行

    ws_sheet[f'C{base_row}'] = str(a_mean)
    ws_sheet[f'C{base_row + 1}'] = str(b_mean)
    ws_sheet[f'C{base_row + 2}'] = str(igd_mean)

    ws_sheet[f'D{base_row}'] = str(a_stdev)
    ws_sheet[f'D{base_row + 1}'] = str(b_stdev)
    ws_sheet[f'D{base_row + 2}'] = str(igd_stdev)

    ws_sheet[f'E{base_row}'] = str(a_res)
    ws_sheet[f'E{base_row + 1}'] = str(b_res)
    ws_sheet[f'E{base_row + 2}'] = str(c_res)
    print(a_res)
    print(b_res)
    print(c_res)
    print(f"hv平均值:{a_mean},PD平均值:{b_mean}\nhv标准差:{a_stdev},PD标准差:{b_stdev}\nigd平均值:{igd_mean},igd标准差:{igd_stdev}")

def save(maxtime,max_iter,learn_iter):
    wb = Workbook()
    ws = wb.active
    ws_sheet = wb.create_sheet('sheet1')
    for i in range(1, 5):
        base_row = (i - 1) * 3 + 1  # 每组3行，起始行
        ws_sheet[f'A{base_row}'] = "CPD" + str(i)
        ws_sheet[f'B{base_row}'] = "HV"
        ws_sheet[f'B{base_row + 1}'] = "PD"
        ws_sheet[f'B{base_row + 2}'] = "IGD"
        mulcalnsga2(i,ws_sheet,maxtime,max_iter,learn_iter)
    wb.save(f"res/res-{maxtime}-{max_iter}-{learn_iter}.xlsx")


if __name__ == '__main__':
    init_node_list=(17, 12, 1, 47)

    pop_list, HV_list, PD_list, IGD_list = m_task.run(cal_HV=True, cal_PD=True, cal_IGD=True)
