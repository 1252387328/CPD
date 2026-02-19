import xlrd
import networkx as nx
import numpy as np


class readData:
    # (零件编号，备选零件编号，0为原零件)
    # perf  零件性能
    # cost   零件价格
    # time   零件工期
    # ce   零件碳排放
    # (零件编号)
    # em_cost   # 紧急采购零件价格
    # em_time   # 紧急采购零件工期
    # redev_cost   # 重新设计零件价格
    # redev_time   # 重新设计零件工期
    def __init__(self, dir):
        self.readbook = xlrd.open_workbook(dir)

    def dsm(self):  # 读取dsm矩阵
        sheet = self.readbook.sheet_by_index(0)
        nrowsmax = sheet.nrows  # 最大行数
        G = nx.DiGraph()
        nodemax=int(max(max(sheet.col_values(0)[1:]),max(sheet.col_values(1)[1:])))
        for i in range(1,nodemax+1):
            G.add_node(i)
            G.nodes[i]['strength'] = 1.0
        DSM = []
        for i in range(nodemax):
            row = []
            for j in range(nodemax):
                row.append(0)
            DSM.append(row)
        for i in range(1, nrowsmax):
            lng = sheet.cell(i, 2).value#权重
            # lngmin=sheet.cell(i, 3).value
            # lngmax=sheet.cell(i, 4).value
            # winterval=Interval(lngmin,lngmax)
            if isinstance(lng, str) == 0 and sheet.cell(i, 0).value != sheet.cell(i, 1).value:
                G.add_weighted_edges_from([(int(sheet.cell(i, 0).value), int(sheet.cell(i, 1).value), lng)])
                DSM[int(sheet.cell(i, 0).value)-1][int(sheet.cell(i, 1).value)-1]=lng
        return G,DSM

    def spi_ratio(self):  # 读取服务性能影响系数
        sheet = self.readbook.sheet_by_index(1)
        nrowsmax = sheet.nrows

        spinum = 0  # 服务性能数目
        spiname = []  # 服务性能名称

        for i in range(1, nrowsmax):
            name = sheet.cell(i, 0).value
            if (name != ''):
                spiname.append(name)
                spinum = spinum + 1
        SPI = np.empty((spinum, 1), dtype=object)
        for i in range(spinum):
            SPI[i][0] = [[], [], []]
        j = 0
        for i in range(1, nrowsmax):
            node = sheet.cell(i, 1).value
            weight = sheet.cell(i, 2).value
            name = sheet.cell(i, 0).value
            if (name != ''):
                SPI[j][0][0].append(name)
            if (isinstance(node, str) == 1):
                j = j + 1
            if (isinstance(node, str) == 0):
                SPI[j][0][1].append(node)
                SPI[j][0][2].append(weight)
        return SPI

    def node(self):  # 读取零件信息
        sheet = self.readbook.sheet_by_index(2)
        nrowsmax = sheet.nrows  # 最大行数
        node = []
        for i in range(1, nrowsmax):
            node.append([sheet.cell(i, 2).value, sheet.cell(i, 3).value, sheet.cell(i, 4).value, sheet.cell(i, 5).value,
                         sheet.cell(i, 7).value, sheet.cell(i, 8).value, sheet.cell(i, 9).value,
                         sheet.cell(i, 10).value])
        return (node)

    def back_node_number(self,i):
        sheet = self.readbook.sheet_by_index(2)
        return int(sheet.cell(i, 6).value)


    def backnode(self):  # 读取备选零件信息
        sheet = self.readbook.sheet_by_index(2)
        nrowsmax = sheet.nrows  # 最大行数
        back_node = []
        for i in range(1, nrowsmax):

            node = [[] for j in range(self.back_node_number(i))]

            for j in range(self.back_node_number(i)):
                node[j].append(
                    [sheet.cell(i, 5 * j + 12).value, sheet.cell(i, 5 * j + 13).value, sheet.cell(i, 5 * j + 14).value,
                     sheet.cell(i, 5 * j + 15).value])

            back_node.append(node)

        return back_node

class returnData:
    def __init__(self, node_data ,nodedata ,backnodedata):
        self.nodedata = nodedata
        self.backnodedata=backnodedata
        self.node_data =node_data


    def perf(self, node_number, back_number):
        return self.nodedata[node_number-1][0] if back_number == 0 else self.backnodedata[node_number-1][back_number-1][0][0]

    def cost(self,node_number, back_number):
        return self.nodedata[node_number-1][1] if back_number == 0 else self.backnodedata[node_number-1][back_number - 1][0][1]

    def time(self,node_number, back_number):
        return self.nodedata[node_number-1][2] if back_number == 0 else self.backnodedata[node_number-1][back_number - 1][0][2]

    def ce(self,node_number, back_number):
        return self.nodedata[node_number-1][3] if back_number == 0 else self.backnodedata[node_number-1][back_number - 1][0][3]

    def em_cost(self,node_number):
        return self.nodedata[node_number-1][4]

    def em_time(self,node_number):
        return self.nodedata[node_number - 1][5]

    def rdev_cost(self,node_number):
        return self.nodedata[node_number - 1][6]

    def rdev_time(self,node_number):
        return self.nodedata[node_number - 1][7]

    def spi_ratio(self):
        return self.node_data.spi_ratio()

    def len(self):
        return len(self.nodedata)

    def get_G(self):
        G,DSM=self.node_data.dsm()
        return G

    def get_DSM(self):
        G,DSM=self.node_data.dsm()
        return DSM

    def get_node_neighbors(self,node_number):
        return list(self.get_G().neighbors(node_number))

    def get_weight(self,node1,node2):
        return self.get_G()[node1][node2]['weight']

    def back_node_num(self,node):
        return self.node_data.back_node_number(node)




