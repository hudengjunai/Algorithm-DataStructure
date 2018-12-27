from enum import Enum,unique
import numpy as np
from collections import deque

@unique
class Color(Enum):
    white = 0
    gray = 1
    black = 2

class Node(object):
    """node in graph store node id,node parent 'pi' node color for detect search node distance for a node to given source"""
    def __init__(self,idx):
        self._id = idx
        self.pi = None
        self.color =None
        self.depth = None

class Graph(object):
    "graph is a logical of G<V,E> graph = vertex + edge"
    def __init__(self,node_nums):
        self.nodes_dict =dict([(idx,Node(idx)) for idx in range(node_nums)])
        self.adj_matrix = np.zeros(shape=(node_nums,node_nums),dtype=np.int32)

    def add_edge(self,*args):
        """"""
        for pair in args:
            i,j = pair
            self.adj_matrix[i,j]=1
            self.adj_matrix[j,i]=1

    def add_direct_edge(self,*args):
        for pair in args:
            i,j = pair
            self.adj_matrix[i,j]=1

    def broad_first_search(self):
        for idx,node in self.nodes_dict.items():
            node.pi = None
            node.color = Color.white
            node.depth =0
        que = deque()
        que.append(self.nodes_dict[0])
        self.nodes_dict[0].color = Color.gray
        self.nodes_dict[0].depth = 0
        while que:
            node = que.popleft()
            n_id = node._id
            for idx,link in enumerate(self.adj_matrix[n_id]):
                if link>0 and idx != n_id:
                    if self.nodes_dict[idx].color == Color.white:
                        self.nodes_dict[idx].color = Color.gray
                        self.nodes_dict[idx].pi = n_id
                        self.nodes_dict[idx].depth = node.depth+1
                        que.append(self.nodes_dict[idx])

            node.color = Color.black
            print("tag black",node._id)
            print("queue",[node._id for node in que])

    def deep_first_search(self):
        self.time = 0
        for idx,node in self.nodes_dict.items():
            node.pi = None
            node.color = Color.white
            node.depth =0
        def Visit(node):
            node.color = Color.gray
            print("detect node {0},time {1}".format(node._id,self.time))
            node.d_time = self.time
            self.time +=1
            for idx,link in enumerate(self.adj_matrix[node._id]):
                if link>0 and self.nodes_dict[idx].color == Color.white:
                    self.nodes_dict[idx].pi = node._id

                    Visit(self.nodes_dict[idx])
            node.color = Color.black
            node.f_time = self.time
            self.time +=1
            print("node:{0},color:{1},time:{2},{3},".format(node._id,node.color,node.d_time,node.f_time))

        for idx, node in self.nodes_dict.items():
            if node.color == Color.white:
                print("root visit",node._id)
                Visit(node)

    def top_sort(self):
        self.deep_first_search()
        node_list = [v for k,v in self.nodes_dict.items()]
        node_list = sorted(node_list,key=lambda x:x.f_time)
        print([node._id for node in node_list])

    def __str__(self):
        return "the matrix of adjcent is \n %s"%(str(self.adj_matrix))

class GraphWeight(object):
    def __init__(self,nodes):
        self.node_list =[Node(idx) for idx in range(nodes)]
        self.adj_list = [[] for i in range(nodes)]

    def add_node_weight(self,*args):
        for w_link in args:
            i,j,w = w_link
            self.adj_list[i].append((j,w))
            self.adj_list[j].append((i,w))

    def build_weight_list(self):
        self.weight_link = []
        for entry_id,link_list in enumerate(self.adj_list):
            for linked_id,wt in link_list:
                if linked_id>entry_id:
                    self.weight_link.append((entry_id,linked_id,wt))
    def kruskal_msp(self):
        self.build_weight_list()
        for node in self.node_list:
            node.pi = node._id

        def root_node(node):
            p = node
            while p.pi !=p._id:
                p = self.node_list[p.pi]
            return p
        def make_union(u,v):
            #set u.root to v root
            root_node(u).pi = root_node(v).pi

        sorted_weight_link = sorted(self.weight_link,key=lambda x:x[2])
        A = []
        msp_w = 0
        for wt_link in sorted_weight_link:
            i,j,w = wt_link
            u,v = self.node_list[i],self.node_list[j]
            if root_node(u) != root_node(v):
                make_union(u,v)
                A.append((i,j))
                msp_w += w
        return A,msp_w

    def prim_msp(self,r=0):
        #default generate from node 0

        print("--------------------prim alg")
        for v in self.node_list:
            v.l_w = np.inf
            v.pi = None
        self.node_list[0].l_w=0
        self.node_list[0].pi = None
        Q = self.node_list[:] #G-V
        A = []
        cost = 0
        while Q:
            Q = sorted(Q,key=lambda x:x.l_w)
            Q_ids = [item._id for item in Q]
            u = Q.pop(0) # the first item
            A.append((u.pi,u._id))

            cost += u.l_w
            for w_linked in self.adj_list[u._id]:
                j,w = w_linked

                if self.node_list[j].l_w>w and j in Q_ids:
                    self.node_list[j].pi = u._id
                    self.node_list[j].l_w = w
        return A,cost





class Graph_Direct_Weight(object):
    def __init__(self,nodes):
        self.node_list = [Node(i) for i in range(nodes)]
        self.adj_list = [[] for i in range(nodes)]

    def add_dw_edge(self,*args):
        for wlink in args:
            s,t,w = wlink
            self.adj_list[s].append((t,w))

    def bellman_ford(self,s):
        """ s is the source id"""
        for node in self.node_list:
            node.dis = np.inf
            node.pi = None

        self.node_list[s].dis = 0

        edges = [(i,j,w) for i,node_link in enumerate(self.adj_list) for j,w in node_link ]
        def relax(u,v,w):
            if self.node_list[v].dis>self.node_list[u].dis+w:
                self.node_list[v].dis = self.node_list[u].dis +w
                self.node_list[v].pi = u

        for _ in self.node_list:
            for wedge in edges:
                i,j,w = wedge
                relax(i,j,w)
        for e in edges:
            u,v,w = e
            if self.node_list[v].dis >self.node_list[u].dis +w:
                return False
        return True

    def path(self,v):
        " to find the v distance and path"
        #print path from target to source
        path = deque()
        node= self.node_list[v]
        path.append(node._id)
        while not node.pi is None:
            node=self.node_list[node.pi]
            path.appendleft(node._id)
        return path

    def simple_dag_shortestpath(self,s):
        # $24.2 page 364
        def relax(u,v,w):
            if self.node_list[v].dis>self.node_list[u].dis+w:
                self.node_list[v].dis = self.node_list[u].dis +w
                self.node_list[v].pi = u

        for node in self.node_list:
            node.pi = None
            node.dis = np.inf
        self.node_list[s].dis = 0
        for node in self.node_list:
            for edge in self.adj_list[node._id]:
                t,w = edge
                relax(node._id,t,w)
        dist = [node.dis for node in self.node_list]
        return dist
    def dijkstra(self,s):
        # $24,3 Dijkstra alg
        for node in self.node_list:
            node.pi = None
            node.dis = np.inf
        self.node_list[s].dis =0

        Q = self.node_list[:]
        S = []
        while Q:
            Q = sorted(Q,key=lambda x:x.dis)
            u = Q.pop(0)
            S.append(u)
            i = u._id
            for j,w in self.adj_list[i]:
                if self.node_list[j].dis > self.node_list[i].dis +w:
                    self.node_list[j].dis = self.node_list[i].dis+w
                    self.node_list[j].pi = self.node_list[i]._id
        dist = [node.dis for node in self.node_list]
        return dist

class Graph_Mat(object):
    def __init__(self,nodes):
        self.node_list = [Node(i) for i in range(nodes)]
        self.adj_wmat = np.ones(shape=(nodes,nodes),dtype=np.int32)*np.inf
        for i in range(nodes):
            self.adj_wmat[i,i]=0

    def add_wd_edge(self,*args):
        for pair in args:
            i,j,w = pair
            self.adj_wmat[i,j]=w

    def slow_all_pair_shortest_path(self):
        # use dynamci plan to calculate the full path matrix
        L = self.adj_wmat[:]
        n = self.adj_wmat.shape[0]
        def extend_shortest_path(L,w):
            n = L.shape[0]
            C = np.ones_like(L)*np.inf
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        C[i,j]= min(C[i,j],L[i,k]+w[k,j])
            return C
        print("before cal distance ,the matrix is \n ",self.adj_wmat)
        for i in range(2,n):
            L = extend_shortest_path(L,self.adj_wmat)
        return L
    def fast_all_pair_shortest_path(self):
        pass


class Graph_MaxFlow(object):
    def __init__(self,nodes):
        self.node_list = [Node(i) for i in range(nodes)]
        self.cap_matrix = np.zeros(shape=(nodes,nodes),dtype=np.int32)
        self.flow =  np.zeros_like(self.cap_matrix)

    def add_cap_edge(self,*args):
        for pair in args:
            i,j,cap = pair
            self.cap_matrix[i,j]= cap

    def bfs_path(self,s,t):
        "find the path and the f"
        for node in self.node_list:
            node.color = Color.white
            node.pi = None
            node.depth = 0
        q = []
        q.append(self.node_list[s])
        self.node_list[s].color = Color.gray

        while q:
            node = q.pop(0)
            i = node._id
            for j,w in enumerate(self.cap_matrix[node._id]):
                if self.cap_matrix[i,j]>0 and self.node_list[j].color == Color.white:
                    self.node_list[j].color = Color.gray
                    self.node_list[j].pi = i
                    self.node_list[j].depth = node.depth +1
                    q.append(self.node_list[j])
            node.color = Color.black

        #search finished
        path = deque()
        node = self.node_list[t]
        c = np.inf
        path.append(node._id)
        while not node.pi is None:
            npi = self.node_list[node.pi]
            path.appendleft(npi._id)
            c = min(c,self.cap_matrix[npi._id,node._id])
            node =npi

        if path[0] != s:
            return None,0
        else:
            return path,c

    def maxflow_ford_fulkerson(self,i,j):
        # $ maxflow-mincut 26.5 example
        path,c = self.bfs_path(0,3)
        while path:
            for i in range((len(path)-1)):
                u,v = path[i],path[i+1]
                self.flow[u,v] += c
                self.flow[v,u] -= c
                self.cap_matrix[u,v] -=c
                self.cap_matrix[v,u] +=c
            path, c = self.bfs_path(0, 3)

        maxflow = sum(self.flow[0])
        flow_matrix = self.flow
        return maxflow,flow_matrix


if __name__=='__main__':

    # test broad first search
    graph = Graph(8)
    graph.add_edge((0,1),(0,4),(1,5),(2,6),(3,7),(2,3),(5,6),(6,7),(2,5),(3,6))
    print(graph)
    graph.broad_first_search()

    # test deep first search
    graph = Graph(6)
    graph.add_direct_edge((0,1),(0,3),(3,1),(4,3),(1,4),(2,4),(2,5),(5,5))
    graph.deep_first_search()

    # test toplogy sort
    graph.top_sort()

    # test kruskal_minmum_span tree
    graph = GraphWeight(9)
    graph.add_node_weight((0,1,4),(1,2,8),(2,3,7),(3,4,9),(4,5,10),(5,6,2),(6,7,1),(7,8,7),(7,0,8))
    graph.add_node_weight((1,7,11),(2,8,2),(2,5,4),(3,5,14),(6,8,6))
    A,w = graph.kruskal_msp()
    print(A,w)

    # test prim minum_span tree
    graph2 = graph
    A,cost = graph2.prim_msp()
    print("output of prim algrithm")
    print(A,cost)

    # test bellman ford single source min distance

    print("test the path search in weighted graph")
    graph = Graph_Direct_Weight(5)
    graph.add_dw_edge((0,1,6),(0,4,7),(1,2,5),(1,3,-4),(1,4,8),(2,1,-2),(3,2,7),(3,0,2),(4,2,-3),(4,3,9))
    graph.bellman_ford(0)
    print("cal dis")
    path_id = graph.path(3)
    print("get path")
    print(path_id)

    # test directed no cycle single source shortest path
    print("test the simple single source no cycle shortest path")
    graph = Graph_Direct_Weight(6)
    graph.add_dw_edge((0,1,5),(0,2,3),(1,2,2),(1,3,6),(2,4,4),(2,5,2),(3,4,-1),(3,5,1),(4,5,-2))
    dist = graph.simple_dag_shortestpath(1)
    print(dist)

    # test dijkstra alg
    print("test the dijkstra alg")
    graph = Graph_Direct_Weight(5)
    graph.add_dw_edge((0,1,10),(0,4,5),(1,2,1),(1,4,2),(2,3,4),(3,0,7),(3,2,6),(4,1,3),(4,3,9),(4,3,2))
    dist = graph.dijkstra(0)
    print(dist)

    # test the full path shortest search
    print("test the full path shortest search")
    graph = Graph_Mat(5)
    graph.add_wd_edge((0,1,3),(0,2,8),(0,4,-4),(1,3,1),(1,4,7),(2,1,4),(3,2,-5),(3,0,2),(4,3,6))
    L = graph.slow_all_pair_shortest_path()
    print(L)

    # test the maxflow alg
    graph = Graph_MaxFlow(6)
    graph.add_cap_edge((0,1,16),(0,5,13),(1,2,12),(1,5,10),(2,5,9),(2,3,20),(4,2,7),(4,3,4),(5,4,14),(5,1,4))
    maxflow,flow_matrix = graph.maxflow_ford_fulkerson(0,3)
    print("max flow \n",maxflow,"\n",flow_matrix)


