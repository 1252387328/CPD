import concurrent.futures

from eva_fun import *
from insgaii import nsgaii
import random
from insgaii import Individual
import sqlite3
import ast
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from concurrent import futures
import json
from scipy.spatial.distance import cdist
class mtask:
    def __init__(self,Task,data,init_node,each_path_size,learn_iter,pop_size,ce_ratio,init_strength,risk_ratio,is_max_min,ts1,tran_num,db1,db2,seed=None):
        self.Task=Task
        self.Return_data,self.Des_var,self.Paths=data
        self.init_node=init_node
        self.each_path_size=each_path_size
        self.learn_iter=learn_iter
        self.ce_ratio=ce_ratio
        self.init_strength=init_strength
        self.risk_ratio=risk_ratio
        self.is_max_min=is_max_min
        self.ts1=ts1
        self.tran_num=tran_num
        self.seed=seed
        self.pop_list=[None for item in Task]
        self.G_list=[self.get_G(item) for item in Task]
        self.db1=db1
        self.db2=db2
        self.conn1,self.conn2,self.conn3=self.init_db()
        self.pop_size=pop_size

    def init_db(self):
        conn1 = sqlite3.connect('./db/cal_max_node.db')
        c = conn1.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS calc_results
                     (task1 TEXT, node1 INTEGER, task2 TEXT, num INTEGER, result_node INTEGER, sis_result REAL)''')
        conn1.commit()

        conn2 = sqlite3.connect('./db/cal_next_node.db')
        c = conn2.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS calc_results
                     (task1 TEXT, nextnode INTEGER, task2 TEXT, lastnode INTEGER, sorted_keys TEXT, sorted_values TEXT)''')
        conn2.commit()
        conn3 = sqlite3.connect('./db/pf.db')
        c = conn3.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS calc_results
                             (task TEXT, initnode INTEGER, pf TEXT)''')
        conn3.commit()
        return conn1,conn2,conn3

    def close_db(self):
        self.conn1.close()
        self.conn2.close()
        self.conn3.close()

    def get_nodenum(self,task):
        return self.Return_data[task-1].len()

    def get_G(self,task):
        return self.Return_data[task-1].get_G()

    def get_node_neighbors(self,task,node):
        return self.Return_data[task-1].get_node_neighbors(node)

    def get_weight(self,task,node1,node2):
        return self.Return_data[task-1].get_weight(node1,node2)

    def get_backnode_num(self,task,node):
        return self.Return_data[task-1].back_node_num(node)

    def get_data(self,task,type,node,back):
        node_data = self.Return_data[task - 1]
        if type=="cost":
            return (node_data.cost(node,back))
        if type=="time":
            return (node_data.time(node,back))
        if type=="perf":
            return (node_data.perf(node,back))
        if type=="ce":
            return (node_data.ce(node,back))

    def cal_sip(self,task1,node1,task2,node2):
        perf1=self.get_data(task1,"perf",node1,0)
        cost1=self.get_data(task1,"cost",node1,0)
        time1=self.get_data(task1,"time",node1,0)
        ce1=self.get_data(task1,"ce",node1,0)*self.ce_ratio
        perf2=self.get_data(task2,"perf",node2,0)
        cost2=self.get_data(task2,"cost",node2,0)
        time2=self.get_data(task2,"time",node2,0)
        ce2=self.get_data(task2,"ce",node2,0)*self.ce_ratio
        data = {
            "variable1": [perf1, cost1, time1, ce1],
            "variable2": [perf2, cost2, time2, ce2],
        }
        x = np.array(data["variable1"])
        y = np.array(data["variable2"])
        sis = np.corrcoef(x, y)[0, 1]
        return sis

    def cal_sis(self,task1,node1,task2,node2):
        # neighbors1=self.get_node_neighbors(task1,node1)
        # neighbors2=self.get_node_neighbors(task2,node2)
        neighbors1=list(self.G_list[task1-1].neighbors(node1))
        neighbors2=list(self.G_list[task2-1].neighbors(node2))
        Sip_max=[]
        # Sip_max_index=[]
        for n1 in neighbors1:
            sip = [0]
            for n2 in neighbors2:
                sip.append(self.cal_sip(task1,n1,task2,n2))
            sip_max=max(sip)
            # sip_max_index=sip.index(sip_max)+1
            Sip_max.append(sip_max)
            # Sip_max_index.append(sip_max_index)
        sis=0
        for i in range(len(Sip_max)):
            sis=sis+Sip_max[i]*self.get_weight(task1,node1,neighbors1[i])
        sis=self.cal_sip(task1,node1,task2,node2)+sis/len(Sip_max)
        return(sis)

    def cal_max_node1(self,task1,node1,task2,num=1):
        sis_list={}
        for i in range(self.get_nodenum(task2)):
            sis_list[str(i+1)]=self.cal_sis(task1, node1, task2, i + 1)
        # print(sis_list)
        sorted_keys = sorted(sis_list, key=sis_list.get, reverse=True)
        return(int(sorted_keys[num-1]),sis_list[sorted_keys[num-1]])
        # return(sis_list_sorted.keys()[:num])

    def cal_max_node(self, task1, node1, task2, num=1):

        c = self.conn1.cursor()

        # 检查数据库中是否已有结果
        c.execute('SELECT result_node, sis_result FROM calc_results WHERE task1=? AND node1=? AND task2=? AND num=?',
                  (task1, node1, task2, num))
        result = c.fetchone()

        if result:
            return result
        else:
            # 如果没有结果，进行计算
            sis_list = {}
            for i in range(self.get_nodenum(task2)):
                sis_list[str(i + 1)] = self.cal_sis(task1, node1, task2, i + 1)

            sorted_keys = sorted(sis_list, key=sis_list.get, reverse=True)
            result_node = int(sorted_keys[num - 1])
            sis_result = sis_list[sorted_keys[num - 1]]

            # 将结果保存到数据库
            c.execute(
                'INSERT INTO calc_results (task1, node1, task2, num, result_node, sis_result) VALUES (?, ?, ?, ?, ?, ?)',
                (task1, node1, task2, num, result_node, sis_result))
            # print(f"{task1}-{node1}-{task2}已保存")
            self.conn1.commit()

            return result_node, sis_result

    def cal_next_node(self,task1,nextnode,task2,lastnode):
        c = self.conn2.cursor()
        c.execute('SELECT sorted_keys, sorted_values FROM calc_results WHERE task1=? AND nextnode=? AND task2=? AND lastnode=?',
                  (task1, nextnode, task2, lastnode))
        result = c.fetchone()
        if result:
            # print(f"{task1}-{nextnode}-{task2}-{lastnode}已存在")
            return ast.literal_eval(result[0]),ast.literal_eval(result[1])
        else:
            neighbors=list(self.G_list[task2-1].neighbors(lastnode))
            sis_list={}
            for node in neighbors:
                sis_list[node]=self.cal_sis(task1,nextnode,task2,node)
            # print(sis_list)
            sorted_keys = sorted(sis_list, key=sis_list.get, reverse=True)
            sorted_values=sorted(sis_list.values(),reverse=True)

            c.execute('INSERT INTO calc_results (task1, nextnode, task2, lastnode, sorted_keys, sorted_values) VALUES (?, ?, ?, ?, ?, ?)',
                (task1, nextnode, task2, lastnode, str(sorted_keys), str(sorted_values)))
            print(f"{task1}-{nextnode}-{task2}-{lastnode}已保存")
            self.conn2.commit()
        return(sorted_keys,sorted_values)

    def cal_next_node1(self,task1,nextnode,task2,lastnode):
        # neighbors=self.get_node_neighbors(task2,lastnode)
        neighbors=list(self.G_list[task2-1].neighbors(lastnode))
        sis_list={}
        for node in neighbors:
            sis_list[node]=self.cal_sis(task1,nextnode,task2,node)
        # print(sis_list)
        sorted_keys = sorted(sis_list, key=sis_list.get, reverse=True)
        sorted_values=sorted(sis_list.values(),reverse=True)
        return(sorted_keys,sorted_values)

    def convert_path(self,task1,path1,task2):
        path2=[]
        sis_list = []
        for i in range(len(path1)):
            if i==0:
                node,sis=self.cal_max_node(task1,path1[i],task2)
                path2.append(node)
                sis_list.append(sis)
            # if i==0:
            #     node=self.init_node
            #     sis=self.cal_sis(task1,path1[0],task2,node)
            #     path2.append(node)
            #     sis_list.append(sis)
            else:
                # print(task1,path1[i],task2,path2[i-1])
                node_list,values_list=self.cal_next_node(task1,path1[i],task2,path2[-1])
                for node in node_list:
                    if node not in path2:
                        path2.append(node)
                        # print(path2)
                        sis_list.append(values_list[node_list.index(node)])
                        break
        return(path2,sis_list)

    def convert_dv(self,task1,dv1,task2):
        try:
            dv2=[]
            stra1 = [dv1[i] for i in range(len(dv1)) if i % 2 != 0]
            path1 = [dv1[i] for i in range(len(dv1)) if i % 2 == 0]
            path2,sis_list=self.convert_path(task1,path1,task2)
            # print(path2,sis_list)
            for i in range(len(path2)):
                dv2.append(path2[i])
                if sis_list[i]>self.ts1:
                    dv2.append(stra1[i])
                else:
                    dv2.append(random.randint(-1, self.get_backnode_num(task2,path2[i])))
            return(dv2)
        except:
            return None

    def cal_pop_div(self,pop):
        def calculate_S(o):
            d_bar = np.mean(o)
            S_A = np.sqrt(np.mean((o - d_bar) ** 2)) / d_bar
            return S_A

        fit=[]
        for i in range(len(pop)):
            fit.append(pop[i].fitness)
        S1= calculate_S([row[0] for row in fit])
        S2= calculate_S([row[1] for row in fit])
        S3= calculate_S([row[2] for row in fit])
        Ssum= 1/S1 + 1/S2 + 1/S3
        p1 = (1/S1) / Ssum
        p2 = (1/S2) / Ssum
        p3 = (1/S3) / Ssum
        return p1,p2,p3

    def cal_tran_num(self,p1,p2,p3):
        for _ in range(self.tran_num):
            random_number = random.random()
            if random_number < p1:
                return 0
            elif random_number < p1 + p2:
                return 1
            else:
                return 2

    def get_tran_inid(self,pop2,pop1):
        p1,p2,p3=self.cal_pop_div(pop1)
        tran_inid={}

        population_dict = {tuple(individual.chromosome): individual.fitness for individual in pop2}
        while len(tran_inid)<self.tran_num:
            n=self.cal_tran_num(p1,p2,p3)
            sorted_dict= dict(sorted(population_dict.items(), key=lambda item: item[1][n]))
            first_key = next(iter(sorted_dict))
            first_value = sorted_dict[first_key]
            tran_inid[first_key] = first_value
            del population_dict[first_key]
        path_list=list(tran_inid.keys())
        var_list= [list(item) for item in path_list]
        # print(path_list)
        return var_list

    def convert_all_dv(self,tran_dv,task2,task1):
        # tran_dv = self.get_tran_inid(pop2, pop1)
        conver_dv = []
        for dv in tran_dv:
            cdv = self.convert_dv(task2, dv, task1)
            if cdv is not None:
                conver_dv.append(cdv)
        return(conver_dv)

    def tran_pop(self,conver_dv,pop1,task1):
        ag=self.calnsga(task1)
        for i in range(len(conver_dv)):
            individual = Individual()
            individual.chromosome = conver_dv[i]
            individual.fitness=ag.evaluate_individual(individual)
            pop1.append(individual)
        return pop1

    def tran_task(self,task2,task1):
        # print(f"转换任务{task2}的种群到{task1}")
        pop1=self.pop_list[task1-1]
        pop2=self.pop_list[task2-1]
        tran_dv=self.get_tran_inid(pop2,pop1)
        conver_dv=self.convert_all_dv(tran_dv,task2,task1)
        pop1=self.tran_pop(conver_dv,pop1,task1)
        self.pop_list[task1-1]=pop1
        return(pop1)

    def tran(self):
        for task in self.Task:
            ran = random.choice([item for item in self.Task if item != task])
            self.tran_task(ran, task)

    def calnsga(self,task,seed=None):
        return_data=self.Return_data[task-1]
        des_var=self.Des_var[task-1]
        paths=self.Paths[task-1]
        if seed is None:
            dv = des_var.init(paths, self.each_path_size, self.init_node,seed)
        else:
            dv = des_var.init(paths, self.each_path_size, self.init_node,self.seed)
        eva_fun=evaFun(des_var=des_var,node_data=return_data,var=dv,init_strength=self.init_strength,ce_ratio=self.ce_ratio,risk_ratio=self.risk_ratio)
        if self.is_max_min:
            min_max=[eva_fun.cal_spi_max_min(),eva_fun.cal_cost_max_min(),eva_fun.cal_time_max_min()]
        else:
            min_max=[(0,1),(0,1),(0,1)]
        eva_fun2=evaFun2(des_var=des_var,node_data=return_data,init_strength=self.init_strength,ce_ratio=self.ce_ratio,risk_ratio=self.risk_ratio,min_max=min_max)
        ag=nsgaii(num_generations=self.learn_iter, dv=dv, tournament_size=20,eva_fun=eva_fun2, crossover_rate=1, mutation_rate=0.2,pop_size=self.pop_size)
        return(ag)

    def get_evafun(self,task,seed=None):
        return_data=self.Return_data[task-1]
        des_var=self.Des_var[task-1]
        paths=self.Paths[task-1]
        if seed is None:
            dv = des_var.init(paths, self.each_path_size, self.init_node,seed)
        else:
            dv = des_var.init(paths, self.each_path_size, self.init_node,self.seed)
        eva_fun=evaFun(des_var=des_var,node_data=return_data,var=dv,init_strength=self.init_strength,ce_ratio=self.ce_ratio,risk_ratio=self.risk_ratio)
        if self.is_max_min:
            min_max=[eva_fun.cal_spi_max_min(),eva_fun.cal_cost_max_min(),eva_fun.cal_time_max_min()]
        else:
            min_max=[(0,1),(0,1),(0,1)]
        eva_fun2=evaFun2(des_var=des_var,node_data=return_data,init_strength=self.init_strength,ce_ratio=self.ce_ratio,risk_ratio=self.risk_ratio,min_max=min_max)
        return(eva_fun2)

    def m_calnsga(self):
        AG=[]
        for task in self.Task:
            ag = self.calnsga(task)
            AG.append(ag)
        return(AG)

    def load_pf(self,task):
        c = self.conn3.cursor()
        c.execute(
            'SELECT pf FROM calc_results WHERE task=? AND initnode=?',
            (task,self.init_node))
        result = c.fetchone()
        if result:
            # print(f"{task}-{self.init_node}-pf已打开")
            return json.loads(result[0])
        else:
            pf=self.cal_pf(task,self.init_node)
            pf_json=json.dumps(pf)
            c.execute('INSERT INTO calc_results (task, initnode, pf) VALUES (?, ?, ?)',
                (task, self.init_node, pf_json))
            print(f"{task}-{self.init_node}-pf已保存")
            self.conn3.commit()
        return(pf)

    def save_pf(self,task,pop):
        pf=self.cal_pf(task,self.init_node,pop)
        c = self.conn3.cursor()
        pf_json = json.dumps(pf)
        c.execute('INSERT OR REPLACE INTO calc_results (task, initnode, pf) VALUES (?, ?, ?)',
            (task,self.init_node,pf_json))
        print(f"{task}-{self.init_node}-pf已更新")
        self.conn3.commit()
        return(pf)

    def cal_pf(self,task,init_node,pop=None):
        ag1 = self.calnsga(task, seed=1)
        if pop is None:
            ag2 = self.calnsga(task,seed=2)
            ag3 = self.calnsga(task,seed=3)
            pop1 = ag1.run()
            pop2 = ag2.run()
            pop3 = ag3.run()
            pop=pop1+pop2+pop3

        pf =ag1.non_dominated_sort(pop)
        # print(pf[0])
        A = []
        for ini in pf[0]:
            fit=[i * 0.95 for i in ini.fitness]
            A.append(fit)
        # A = np.array(A)
        return(A)


    def cal_HV(self,pop,ref_point = np.array([1.0, 1.0, 1.0])):
        A = []
        for ini in pop:
            A.append(ini.fitness)
        A = np.array(A)
        ind = HV(ref_point=ref_point)
        return (ind(A))

    def cal_IGD(self,pop,task):
        pf=self.load_pf(task)
        A = []
        for ini in pop:
            if ini.rank==0:
                A.append(ini.fitness)
        # print(A)
        # print(type(pf[0]))

        A = np.array(A)

        pf=np.array(pf)
        ind = IGD(pf)
        return(ind(A))

    def cal_PD(self,pop):
        A = []
        for ini in pop:
            if ini.rank == 0:
                A.append(ini.fitness)
        A = np.array(A)
        n = len(pop)
        C = np.eye(n, dtype=bool)

        D = cdist(A, A, metric='minkowski', p=0.1)

        np.fill_diagonal(D, np.inf)
        score = 0

        for k in range(n - 1):
            while True:
                d, J = np.min(D, axis=1), np.argmin(D, axis=1)
                i = np.argmax(d)
                if D[J[i], i] != -np.inf:
                    D[J[i], i] = np.inf
                if D[i, J[i]] != -np.inf:
                    D[i, J[i]] = np.inf
                # print(C[i, :])
                P = C[i, :]
                while not P[J[i]]:
                    newP = np.any(C[P, :], axis=0)
                    if np.array_equal(P, newP):
                        break
                    else:
                        P = newP

                if not P[J[i]]:
                    break

            C[i, J[i]] = True
            C[J[i], i] = True
            D[i, :] = -np.inf
            score += d[i]

            return score

    def run(self,cal_HV=False,cal_PD=False,cal_IGD=False):
        AG=self.m_calnsga()
        for i in range(len(self.Task)):
            self.pop_list[i]=AG[i].run(self.pop_list[i])


            # print(len(self.pop_list[i]))
        HV_list = []
        IGD_list = []
        PD_list = []
        if cal_HV and cal_PD and cal_IGD:
            for i in range(len(self.Task)):
                HV_list.append(self.cal_HV(self.pop_list[i]))
                PD_list.append(self.cal_PD(self.pop_list[i]))
                IGD_list.append(self.cal_IGD(self.pop_list[i],self.Task[i]))
            # print(HV_list,IGD_list)

        if cal_HV and cal_PD:
            for i in range(len(self.Task)):
                HV_list.append(self.cal_HV(self.pop_list[i]))
                PD_list.append(self.cal_PD(self.pop_list[i]))
            # print(HV_list,IGD_list)

        elif cal_HV:
            for i in range(len(self.Task)):
                HV_list.append(self.cal_HV(self.pop_list[i]))
            # print(HV_list)

        elif cal_PD:
            for i in range(len(self.Task)):
                PD_list.append(self.cal_PD(self.pop_list[i]))
            # print(PD_list)
        return self.pop_list,HV_list,PD_list,IGD_list




