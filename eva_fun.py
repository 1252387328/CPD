import numpy as np
class evaFun:

    def __init__(self,des_var, node_data, var,init_strength,ce_ratio,risk_ratio):
        self.node_data = node_data
        self.paths=[]
        self.paths_strategy=[]
        for i in var:
            self.paths.append(i[::2])
            self.paths_strategy.append(i[1::2])
        self.unique_paths = []
        for item in self.paths:
            if item not in self.unique_paths:
                self.unique_paths.append(item)
        
        self.ce_ratio = ce_ratio
        self.risk_ratio=risk_ratio
        self.paths_strength=des_var.calculate_all_path_strength(self.paths,init_strength)
        self.unique_paths_strength = des_var.calculate_all_path_strength(self.unique_paths, init_strength)
        # imax=Interval.imax
        # imin=Interval.imin

    def cal_spi_max_min(self):
        allspimax=[]
        allspimin=[]
        for i in range(len(self.unique_paths)):
            spimax=0
            spimin=0
            for j in range(len(self.unique_paths[i])):
                spi1=max(0,self.node_data.perf(self.unique_paths[i][j],0)*self.unique_paths_strength[i][j],\
                -min(self.node_data.perf(self.unique_paths[i][j],k) for k in range(1,5))+self.node_data.perf(self.unique_paths[i][j],0))
                spi2 = min(0, self.node_data.perf(self.unique_paths[i][j], 0) * self.unique_paths_strength[i][j], \
                -max(self.node_data.perf(self.unique_paths[i][j],k) for k in range(1,5))+self.node_data.perf(self.unique_paths[i][j],0))
                spimax=spimax+spi1
                spimin=spimin+spi2
            allspimax.append(spimax)
            allspimin.append(spimin)
            spi_max=max(allspimax)
            spi_min=min(allspimin)
            # if isinstance(spi_max,Interval):
            #     spi_max=spi_max.b
            # if isinstance(spi_min,Interval):
            #     spi_min=spi_min.a
        return(spi_min,spi_max)

    def cal_time_max_min(self):
        alltimemax = []
        alltimemin = []
        for i in range(len(self.unique_paths)):
            timemax = 0
            timemin = 0
            for j in range(len(self.unique_paths[i])):

                t1=max(0,self.node_data.rdev_time(self.unique_paths[i][j]) * self.unique_paths_strength[i][j],\
                       min(self.node_data.time(self.unique_paths[i][j], k) for k in range(1,5))-self.node_data.time(self.unique_paths[i][j], 0))

                t2=min(0,self.node_data.rdev_time(self.unique_paths[i][j]) * self.unique_paths_strength[i][j],\
                       max(self.node_data.time(self.unique_paths[i][j], k) for k in range(1,5))-self.node_data.time(self.unique_paths[i][j], 0))
                timemax=timemax+t1
                timemin=timemin+t2
            alltimemax.append(timemax)
            alltimemin.append(timemin)
        time_max=max(alltimemax)
        time_min=min(alltimemin)
        # if isinstance(time_max, Interval):
        #     time_max=time_max.a
        # if isinstance(time_min, Interval):
        #     time_min=time_min.b
        return(time_min,time_max)

    def cal_cost_max_min(self):
        allcostmax = []
        allcostmin = []
        for i in range(len(self.unique_paths)):
            costmax = 0
            costmin = 0
            for j in range(len(self.unique_paths[i])):
                c1=max(0,self.node_data.rdev_cost(self.unique_paths[i][j]) * self.unique_paths_strength[i][j],\
                max([self.node_data.cost(self.unique_paths[i][j],k) for k in range(1,5)])-self.node_data.cost(
                        self.unique_paths[i][j], 0))

                ce1=max(0,max([self.node_data.ce(self.unique_paths[i][j], k) for k in range(1,5)]))
                c2 = min(0, self.node_data.rdev_cost(self.unique_paths[i][j]) * self.unique_paths_strength[i][j], \
                         min([self.node_data.cost(self.unique_paths[i][j], k) for k in range(1, 5)]) - self.node_data.cost(
                             self.unique_paths[i][j], 0))

                ce2 = min(0, min([self.node_data.ce(self.unique_paths[i][j], k) for k in range(1, 5)]))
                costmax=costmax+c1+self.ce_ratio*ce1
                costmin=costmin+c2+self.ce_ratio*ce2

            allcostmax.append(costmax)
            allcostmin.append(costmin)
        cost_max=max(allcostmax)
        cost_min=min(allcostmin)
        # if isinstance(cost_max, Interval):
        #     cost_max=cost_max.a
        # if isinstance(cost_min, Interval):
        #     cost_min=cost_min.b
        return(cost_min,cost_max)


    def cal_spi(self):  # 服务性能影响度

        allspi = []
        for i in range(len(self.paths)):
            sum_spi = 0
            for j in range(len(self.paths[i])):
                # print(i,j)
                spi=self.node_data.perf(self.paths[i][j],0)*self.paths_strength[i][j] if self.paths_strategy[i][j]==0 else \
                -self.node_data.perf(self.paths[i][j],self.paths_strategy[i][j])+self.node_data.perf(self.paths[i][j],0) if self.paths_strategy[i][j]>0 else 0

                sum_spi=sum_spi+spi

            allspi.append(sum_spi)
        return allspi

    def cal_cost(self):  # 成本
        # self.var = self.int_var(self.var)
        cost = []

        for i in range(len(self.paths)):
            sum_c = 0
            for j in range(len(self.paths[i])):
                c=self.node_data.rdev_cost(self.paths[i][j])*self.paths_strength[i][j] if self.paths_strategy[i][j] ==-1 else \
                    self.node_data.cost(self.paths[i][j],self.paths_strategy[i][j])-self.node_data.cost(self.paths[i][j],0) if self.paths_strategy[i][j]>0 else 0

                ce=self.node_data.ce(self.paths[i][j],self.paths_strategy[i][j])-self.node_data.ce(self.paths[i][j],0) if self.paths_strategy[i][j]>0 else 0

                # risk=self.node_data.risk(self.paths[i][j],self.paths_strategy[i][j])-self.node_data.risk(self.paths[i][j],0) if self.paths_strategy[i][j]>0 else 0

                # sum_c = sum_c + c + self.ce_ratio * ce + self.risk_ratio*risk
                sum_c = sum_c + c + self.ce_ratio * ce

            cost.append(sum_c)
        return cost

    def cal_time(self):  # 工期
        # self.var = self.int_var(self.var)
        time = []
        for i in range(len(self.paths)):
            sum_t=0
            for j in range(len(self.paths[i])):
                t=self.node_data.rdev_time(self.paths[i][j])*self.paths_strength[i][j] if self.paths_strategy[i][j] ==-1 else \
                    self.node_data.time(self.paths[i][j],self.paths_strategy[i][j])-self.node_data.time(self.paths[i][j],0) if self.paths_strategy[i][j]>0 else 0
                sum_t=sum_t+t
            time.append(sum_t)
        return time

class evaFun2:

    def __init__(self, des_var, node_data, init_strength, ce_ratio, risk_ratio,min_max):
        self.node_data = node_data
        self.des_var=des_var
        self.ce_ratio = ce_ratio
        self.risk_ratio = risk_ratio
        self.init_strength=init_strength
        self.spimin = min_max[0][0]
        self.spimax = min_max[0][1]
        self.costmin = min_max[1][0]
        self.costmax = min_max[1][1]
        self.timemin = min_max[2][0]
        self.timemax = min_max[2][1]


    def cal_path(self,var):
        paths = var[::2]
        paths_strategy = var[1::2]
        paths_strength = self.des_var.calculate_path_strength(paths, self.init_strength)
        return(paths,paths_strategy,paths_strength)

    def cal_spi(self,var):  # 服务性能影响度
        paths, paths_strategy, paths_strength=self.cal_path(var)
        sum_spi = 0
        for j in range(len(paths)):
            spi = self.node_data.perf(paths[j], 0) * paths_strength[j] if \
            paths_strategy[j] == 0 else \
                max(0,-self.node_data.perf(paths[j], paths_strategy[j]) + self.node_data.perf(
                    paths[j], 0)) if paths_strategy[j] > 0 else 0

            sum_spi = sum_spi + spi
        return (sum_spi-self.spimin)/(self.spimax-self.spimin)

    def cal_cost(self,var):  # 成本
        # self.var = self.int_var(self.var)
        paths, paths_strategy, paths_strength = self.cal_path(var)
        sum_c = 0
        for j in range(len(paths)):
            c = self.node_data.rdev_cost(paths[j]) * paths_strength[j] if \
            paths_strategy[j] == -1 else \
                max(0,self.node_data.cost(paths[j], paths_strategy[j]) - self.node_data.cost(
                    paths[j], 0)) if paths_strategy[j] > 0 else 0

            ce = max(self.node_data.ce(paths[j], paths_strategy[j]) - self.node_data.ce(
                paths[j], 0),0) if paths_strategy[j] > 0 else 0

            # risk=self.node_data.risk(self.paths[i][j],self.paths_strategy[i][j])-self.node_data.risk(self.paths[i][j],0) if self.paths_strategy[i][j]>0 else 0

            # sum_c = sum_c + c + self.ce_ratio * ce + self.risk_ratio*risk
            sum_c = sum_c + c + self.ce_ratio * ce
        return (sum_c-self.costmin)/(self.costmax-self.costmin)

    def cal_time(self,var):  # 工期
        # self.var = self.int_var(self.var)
        paths, paths_strategy, paths_strength = self.cal_path(var)
        sum_t = 0
        for j in range(len(paths)):
            t = self.node_data.rdev_time(paths[j]) * paths_strength[j] if \
            paths_strategy[j] == -1 else \
                max(0,self.node_data.time(paths[j], paths_strategy[j]) - self.node_data.time(
                   paths[j], 0)) if paths_strategy[j] > 0 else 0
            sum_t = sum_t + t
        return (sum_t-self.timemin)/(self.timemax-self.timemin)


    # def discrete_var(self,v):
    #     if v > 4.5:
    #         v = 4
    #     if v < -1.5:
    #         v = -2
    #     X = [ -1, 0, 1, 2, 3, 4]
    #
    #     D = [1 / abs(v - x + 1e-6) for x in X]
    #     D1 = [d / sum(D) for d in D]
    #     # print(np.random.choice(X, 1, p=D1)[0])
    #     return int(np.random.choice(X, 1, p=D1)[0])

    # def int_var(self,var):
    #     for i in range(len(var)):
    #         for j in range(len(var[0])):
    #             if type(var[i][j]) != int:
    #                 var[i][j]=self.discrete_var(var[i][j])
    #     return var
