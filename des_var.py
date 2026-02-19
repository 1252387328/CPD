import networkx as nx
import random


class desVar:
    def __init__(self, node_data, min_strength):

        self.node_data = node_data

        self.min_strength=min_strength

        self.graph, self.dsm = self.node_data.dsm()

    def calculate_strength(self, path_list, start_strength):
        current_strength = start_strength
        for i in range(len(path_list) - 1):
            node1 = path_list[i]
            node2 = path_list[i + 1]
            # edge_weight = G[node1][node2]['weight'] #精确权重
            edge_weight = self.dsm[node1 - 1][node2 - 1]
            current_strength = current_strength * edge_weight
        return current_strength

    def calculate_path_strength(self, path_list, start_strength):
        current_strength = start_strength
        path_strength=[]
        path_strength.append(current_strength)
        for i in range(len(path_list) - 1):
            node1 = path_list[i]
            node2 = path_list[i + 1]
            # edge_weight = G[node1][node2]['weight'] #精确权重
            edge_weight = self.dsm[node1 - 1][node2 - 1]
            current_strength = current_strength * edge_weight
            path_strength.append(current_strength)
        return path_strength

    def calculate_all_path_strength(self, paths, start_strength):
        strength=[]
        for path in paths:
            strength.append(self.calculate_path_strength(path, start_strength))
        return strength

    def find_paths(self,path_number,init_node,init_strength):
        graph, dsm = self.node_data.dsm()
        paths = []
        queue = [(init_node, [init_node])]
        while queue:
            (node, path) = queue.pop(0)
            strength = graph.nodes[node]['strength']
            if strength < self.min_strength:
                continue
            for neighbor in graph.neighbors(node):
                neighbor_strength = strength * graph[node][neighbor]['weight']
                if neighbor_strength >= self.min_strength and neighbor not in path:
                    new_path = path + [neighbor]
                    if self.calculate_strength(new_path, init_strength) < self.min_strength:
                        paths.append(new_path)
                        # print(new_path, self.calculate_strength(new_path, self.init_strength))
                    queue.append((neighbor, new_path))
            if (len(paths) >= path_number):
                break

        paths=paths[:path_number]
        return paths

    def init(self, paths, each_path_size, init_node, seed=1234):
        var = []
        if seed is not None:
            random.seed(seed)
        for j in range(len(paths)):
            for n in range(each_path_size):
                var_tmp = []
                for i in range(len(paths[j])):
                    var_tmp.append(paths[j][i])
                    var_tmp.append(random.randint(1, self.node_data.back_node_number(paths[j][i])) if paths[j][i]==init_node else random.randint(-1, self.node_data.back_node_number(paths[j][i])))
                var.append(var_tmp)
        return var

