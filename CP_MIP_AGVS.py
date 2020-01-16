from collections import defaultdict
import pandas as pd
import numpy as np
import math
from docplex.cp.model import *
from sys import stdout
import matplotlib.pyplot as plt

'''
#First Layout
AGV = [1, 2, 3]
Start = [1, 9, 5]
requests = [(1, 4), (2, 5), (3, 6)]
Task_duration = [1, 1, 1, 1, 1, 1]
Precedence = [(1, 4), (2, 5), (3, 6), (6, 1)]
before = [1, 2, 3, 6]  # Already contains the P1->D1 conditions
after = [4, 5, 6, 1]  # Already contains the P1->D1 conditions
real_tasks = [1, 2, 3, 4, 5, 6]
W_p = [1, 2, 3]
W_d = [4, 5, 6]
Nodes_Tasks = [4, 7, 2, 6, 3, 9]
# NODES & DISTANCES
Nodes_id = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Nodes = [(0, 2), (1, 2), (2, 2), (0, 1), (1, 1), (2, 1), (0, 0), (1, 0), (2, 0)]
x_nodes = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_nodes = [2, 2, 2, 1, 1, 1, 0, 0, 0]
from_vector = [1, 1, 4, 4, 2, 2, 6, 6, 8, 8, 9, 9]
to_vector = [2, 4, 7, 5, 5, 3, 5, 3, 7, 5, 8, 6]
dist_vector = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Capacity Parameters
Capacity = [1, 2, 2]
Flow = [+1, +1, +1, -1, -1, -1]  # 1:16 each task has it. is like duration so probably will be [r1 -1]
'''


#Second Layout
####VARY AGVS
#AGV = [1, 2, 3, 4, 5, 6]
#Start = [1, 6, 10, 17, 24, 28]
#Capacity = [1, 2, 1, 2, 1, 2]

#AGV = [1, 2, 3, 4, 5]
#Start = [1, 6, 10, 17, 24]
#Capacity = [1, 2, 1, 2, 1]

#AGV = [1, 2, 3, 4]
#Start = [1, 6, 10, 17]
#Capacity = [1, 2, 1, 2]

#AGV = [1, 2, 3]
#Start = [1, 6, 10]
#Capacity = [1, 2, 1]

#VARY CAPACITIES

'''
requests = [(1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16)]
Task_duration = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Precedence = [(9, 5), (10, 6), (11, 7), (12, 8), (5, 10), (6, 11), (7, 12), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16)]
before = [9, 10, 11, 12, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8]  # Already contains the P1->D1 conditions
after = [5, 6, 7, 8, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16]  # Already contains the P1->D1 conditions
real_tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
W_p = [1, 2, 3, 4, 5, 6, 7, 8]
W_d = [9, 10, 11, 12, 13, 14, 15, 16]
Nodes_Tasks = [2, 10, 14, 18, 20, 3, 8, 27, 30, 12, 9, 2, 1, 21, 30, 7]
Nodes_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
Flow = [+1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1]  # 1:16 each task has it. is like duration so probably will be [r1 -1]
'''




########################################
requests = [(1, 13), (2, 14), (3, 15), (4, 16), (5, 17), (6, 18), (7, 19), (8, 20), (9, 21), (10, 22), (11, 23), (12, 24)]
Task_duration = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Precedence = [(1, 13), (2, 14), (3, 15), (4, 16), (5, 17), (6, 18), (7, 19), (8, 20), (9, 21), (10, 22), (11, 23), (12, 24), (24, 1), (22, 3), (20, 5), (18, 7)]
before = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 22, 20, 18]
after = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1, 3, 5, 7]
real_tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
W_p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
W_d = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
Nodes_Tasks = [3, 1, 10, 15, 20, 25, 13, 17, 30, 34, 40, 9, 2, 8, 11, 26, 31, 33, 34, 41, 4, 7, 24, 11]
Flow = [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

Nodes_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
Nodes = [(6, 44), (14, 36), (14, 44), (22, 36), (22, 44), (30, 36), (30, 44), (38, 36), (38, 44), (44, 36), (44, 44), (50, 36), (50, 44), (56, 44), (56, 36),(60, 30), (64, 28), (60, 24), (64, 20), (60, 16), (64, 12), (56, 8), (58, 4), (56, 4), (48, 8), (40, 8), (48, 4), (36, 8), (40, 4), (28, 8), (36, 4), (20, 8), (28, 4), (12, 8), (20, 4), (6, 16), (12, 4), (6, 24), (4, 4), (6, 30), (2, 18), (2, 30)]
from_vector = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,    9, 9,  10, 11, 12, 13, 13, 13,14, 14, 15, 15, 16, 17, 18, 18, 18, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25, 25, 26, 26, 26, 27, 28, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 32, 33, 33, 34, 34, 35, 35, 36, 37, 38, 39, 39, 36, 41, 40, 41, 42]
to_vector = [3, 2, 5, 4, 5, 4, 7, 6, 7, 6, 8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 15, 14, 15, 12,15, 17, 16, 17, 18, 19, 19, 20, 21, 21, 22, 23, 23, 22, 25, 24, 24, 25, 27, 26, 29, 28, 29, 31, 29, 31, 30, 33, 31, 28, 32, 35, 30, 33, 34, 35, 37, 32, 35, 36, 39, 34, 37, 38, 39, 40, 41, 36, 41, 38, 2, 42, 1]
x_nodes = [6, 14, 14, 22, 22, 30, 30, 38, 38, 44, 44, 50, 50, 56, 56, 60, 64, 60, 64, 60, 64, 56, 58, 56, 48, 40, 48, 36, 40, 28, 36, 20, 28, 12, 20, 6, 12, 6, 4, 6, 2, 2]
y_nodes = [44, 36, 44, 36, 44, 36, 44, 36, 44, 36, 44, 36, 44, 44, 36, 30, 28, 24, 20, 16, 12, 8, 4, 4, 8, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 16, 4, 24, 4, 30, 18, 30]
dist_vector = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


AGV = [1, 2, 3, 4, 5, 6]
Capacity = [1, 1, 1, 1, 1, 1]
Start = [1, 8, 16, 24, 32, 42]


########################################

'''
#Third Layout
AGV = [1, 2, 3, 4, 5, 6]
Start = [1, 6, 12, 18, 24, 30]
Capacity = [1, 2, 1, 2, 1, 2]

requests = [(1, 11), (2, 12), (3, 13), (4, 14), (5, 15), (6, 16), (7, 17), (8, 18), (9, 19), (10, 20)]
Task_duration = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Precedence = [(19, 7), (16, 2), (19,8), (3, 15), (1, 11), (2, 12), (3, 13), (4, 14), (5, 15), (6, 16), (7, 17), (8, 18), (9, 19), (10, 20)]
before = [19, 16, 19, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
after = [7, 2, 8, 15, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
real_tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
W_p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
W_d = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Nodes_Tasks = [1, 10, 23, 23, 7, 22, 6, 9, 5, 4, 16, 26, 12, 2, 25, 22, 12, 6, 18, 23]
Nodes_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
Nodes = [(6, 44), (14, 36), (14, 44), (22, 36), (22, 44), (30, 36), (30, 44), (38, 44), (38, 36), (44, 30), (48, 28), (44, 24), (48, 20), (44, 16), (48, 12), (36, 8), (40, 4), (36, 4), (28, 8), (20, 8), (28, 4), (12, 8), (20, 4), (6, 16), (12, 4), (6, 24), (4, 4), (6, 30), (2, 18), (2, 30)]
x_nodes = [6, 14, 14, 22, 22, 30, 30, 38, 38, 44, 48, 44, 48, 44, 48, 36, 40, 36, 28, 20, 28, 12, 20, 6, 12, 6, 4, 6, 2, 2]
y_nodes = [44,36,44,36,44,36,44,36,44,30,28,24,20,16,12,8,4,4,8,8,4,8,4,16,4,24,4,30,18,30]
dist_vector = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
from_vector = [1, 1, 2, 2, 3, 3, 5, 4, 7, 7, 6, 9, 8, 8, 10, 11, 12, 12, 13, 15, 14, 14, 16, 17, 18, 18, 19, 19, 21, 20, 20, 23, 23, 22, 25, 24, 27, 27, 29, 30, 26, 26, 28, 28]
to_vector = [3, 2, 5, 4, 5, 4, 7, 6, 8, 9, 8, 11, 10, 11, 12, 13, 13, 14, 15, 17, 17, 16, 19, 18, 19, 21, 20, 23, 23, 25, 22, 25, 22, 24, 27, 26, 29, 26, 30, 1, 28, 1, 3, 2]
Flow = [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
'''

'''
AGV = [1, 2, 3]
Start = [1, 9, 5]
Capacity = [1, 1, 1]
requests = [(1, 11), (2, 12), (3, 13), (4, 14), (5, 15), (6, 16), (7, 17), (8, 18), (9, 19), (10, 20)]
Task_duration = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Precedence = [(14, 9), (2, 13), (8, 11), (13, 6), (19, 7), (16, 2), (19, 8), (3, 15)]
before = [16, 19, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Already contains the P1->D1 conditions
after = [2, 8, 15, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # Already contains the P1->D1 conditions
real_tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Flow = [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
W_p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
W_d = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Nodes_Tasks = [1, 10, 23, 24, 7, 22, 6, 9, 5, 4, 16, 26, 12, 2, 25, 22, 12, 6, 18, 30]

# NODES & DISTANCES
x_nodes = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
y_nodes = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
dist_vector = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

Nodes_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
Nodes = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)]
from_vector = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 25, 26, 27, 28, 29]
to_vector = [2, 7, 3, 8, 4, 9, 5, 10, 6, 11, 12, 8, 13, 9, 14, 10, 15, 11, 16, 12, 17, 18, 14, 19, 20, 15, 16, 21, 17, 22, 18, 23, 24, 20, 25, 21, 26, 22, 27, 23, 28, 24, 29, 30, 26, 27, 28, 29, 30]
'''



dummy = AGV.copy()
zeros = AGV.copy()
for a in range(len(AGV)):
    dummy[a] = "%s%s" % ("d", a)
    zeros[a] = 0
requests_and_start = dummy + requests
requests_and_finish = requests + dummy
total_requests = dummy + requests + dummy
tasks = dummy + real_tasks + dummy
Duration = zeros + Task_duration + zeros

I = len(requests_and_start)
O = len(requests_and_finish)
A = len(AGV)
R = len(requests)
T = len(tasks)
R_T = len(real_tasks)
T_R = len(total_requests)

time_horizon = list(range(1, 300))
M = len(time_horizon)
D = len(dummy)
P = len(before)


def distance(p1, p2):
    distance = math.sqrt(((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2))
    return(distance)


# print(dist_vector)

new_edges_df = pd.DataFrame(
    {'X': from_vector,
     'Y': to_vector,
     'Z': dist_vector
     })

list_edges = [tuple(r) for r in new_edges_df.values]

#####


class Graph():
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight


graph = Graph()

edges = list_edges

for edge in edges:
    graph.add_edge(*edge)


def dijsktra(graph, initial, end):
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path


def SP(source, dest):
    A = dijsktra(graph, source, dest)
    dfObj = pd.DataFrame(edges)
    edges_0 = list(dfObj[0])
    edges_1 = list(dfObj[1])
    edges_2 = list(dfObj[2])

    appenddf1 = pd.DataFrame(
        {'X': edges_1,
         'Y': edges_0,
         'Z': edges_2
         })
    appenddf2 = pd.DataFrame(
        {'X': edges_0,
         'Y': edges_1,
         'Z': edges_2
         })

    dfObj = appenddf2.append(pd.DataFrame(data=appenddf1), ignore_index=True)
    dfObj = pd.DataFrame(dfObj)

    sum = 0
    for i in range(len(A) - 1):
        sum = sum + int(dfObj.loc[(dfObj['X'] == A[i]) & (dfObj['Y'] == A[i + 1])]['Z'])
    return(sum)


#==============CONSTRAINT PROGRAMMING MODEL==============#

mdl = CpoModel()
# Variables
# Alloc
Alloc = [mdl.integer_var(min=1, max=len(AGV), name="Alloc" + str(r)) for r in range(T_R)]
# Succ[u]
Succ = [mdl.integer_var(min=len(AGV), max=(len(real_tasks) + len(AGV) + len(AGV) - 1), name="Succ" + str(r)) for r in range(D + R_T)]
# StartTime
Starttime = [mdl.integer_var(min=0, max=M, name="Starttime" + str(j)) for j in range(T)]

#===================================
Alloc_T = [mdl.integer_var(min=1, max=len(AGV), name="Alloc_T" + str(r)) for r in range(T)]
CFlow = [mdl.integer_var(min=0, max=28, name="Alloc_T" + str(r)) for r in range(D + R_T)]


# Constraints
# 1 Considering 20 requests: 6 dummy start, 8 real requests, 6 dummy end
for k in range(A):
    mdl.add(Alloc[k] == k + 1)
    mdl.add(Alloc[R + D + k] == k + 1)
    mdl.add(Alloc_T[k] == k + 1)
    mdl.add(Alloc_T[R_T + D + k] == k + 1)

# 2
for r in range(D):
    mdl.add(sum([(Succ[r] == s + D) for s in range(R_T + D)]) == 1)

# 3
for o in range(R_T + D):
    mdl.add(Alloc_T[o] == element(Alloc_T, Succ[o]))  # Check, maybe is o + D

# Vinculate Alloc_T and Alloc
for o in range(R):
    mdl.add(Alloc[o + D] == Alloc_T[o + D])
    mdl.add(Alloc[o + D] == Alloc_T[o + R + D])

# 4
for d in range(D):
    mdl.add(Starttime[d] == 0)

# 5
for k in range(A):
    for d in range(D):
        for r in range(R_T):
            mdl.add(if_then(logical_and(Alloc[d] == k + 1, Succ[d] == r + D), Starttime[r + D] >= SP(Start[k], Nodes_Tasks[r])))

# 6
mdl.add(mdl.all_diff(Succ))


# 7
for r in range(R):
    mdl.add(Starttime[W_p[r] + D - 1] + 1 + SP(Nodes_Tasks[W_p[r] - 1], Nodes_Tasks[W_d[r] - 1]) <= Starttime[W_d[r] + D - 1])

# 8
for r1 in range(R_T):
    for r2 in range(R_T):
        mdl.add(if_then(Succ[r1 + D] == r2 + D, Starttime[r1 + D] + 1 + SP(Nodes_Tasks[r1], Nodes_Tasks[r2]) <= Starttime[r2 + D]))

# 9
for u in range(P):
    mdl.add(Starttime[before[u] + D - 1] + Duration[before[u] + D - 1] <= Starttime[after[u] + D - 1])


# 10
for i in range(R_T):
    for j in range(R_T):
        if(i != j):
            if (Nodes_Tasks[i] == Nodes_Tasks[j]):
                mdl.add(logical_or(Starttime[i + D] >= Starttime[j + D] + 1, Starttime[i + D] + 1 <= Starttime[j + D]))


# Capacity Constraints
for r1 in range(R_T):
    for r2 in range(R_T):
        for k in range(A):
            mdl.add(if_then(logical_and(Succ[r1 + D] == r2 + D, Alloc_T[r1 + D] == k + 1), Flow[r1] + Flow[r2] <= Capacity[k]))  # This may be redundant

# 11
for d in range(D):
    mdl.add(CFlow[d] == 0)


# 12
for r1 in range(R_T):
    for r2 in range(R_T):
        mdl.add(if_then(Succ[r1 + D] == r2 + D, CFlow[r2 + D] == CFlow[r1 + D] + Flow[r2]))

# 13
for r1 in range(R_T):
    for k in range(A):
        mdl.add(if_then(Alloc_T[r1 + D] == k + 1, CFlow[r1 + D] <= Capacity[k]))  # This may be redundant


##### LOGIC CUT

Prev_Alloc_T = [1, 2, 3, 4, 5, 6, 5, 6, 6, 2, 1, 4, 6, 3, 4, 5, 1, 2, 5, 6, 6, 2, 1, 4, 6, 3, 4, 5, 1, 2, 1, 2, 3, 4, 5, 6]
Prev_Starttime = [0, 0, 0, 0, 0, 0, 11, 1, 9, 7, 15, 1, 14, 2, 8, 1, 2, 2, 14, 6, 12, 14, 20, 5, 24, 13, 16, 8, 12, 4, 0, 0, 0, 0, 0, 0]

for r in range(T_R):
    mdl.add(Alloc_T[r] == Prev_Alloc_T[r])

#mdl.add(sum([((Starttime[W_d[r] + D - 1] - Starttime[W_p[r] + D - 1]) > (Prev_Starttime[W_d[r] + D - 1] - Prev_Starttime[W_p[r] + D - 1])) for r in range(R)]) == 8)

for r in range(R):
    mdl.add(Starttime[W_d[r] + D - 1] - Starttime[W_p[r] + D - 1] > 1+ Prev_Starttime[W_d[r] + D - 1] - Prev_Starttime[W_p[r] + D - 1])


#####





mdl.add(mdl.minimize(sum(Starttime[j + D] for j in range(R_T))))




print("\nSolving model....")
msol = mdl.solve(TimeLimit=360, url="https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/", key="api_18e77e8b-09aa-4e07-8a4a-57b0aae5fdae")

# https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/
# api_eb240cba-695f-4d53-a009-d9a7a2f313a0


# Solution
Starttime_V = list(range(0, len(Starttime)))
for i in range(len(Starttime)):
    Starttime_V[i] = msol.get_value(Starttime[i])

Alloc_V = list(range(0, len(Alloc)))
for i in range(len(Alloc)):
    Alloc_V[i] = msol.get_value(Alloc[i])

Succ_V = list(range(0, len(Succ)))
for i in range(len(Succ)):
    Succ_V[i] = msol.get_value(Succ[i])

Alloc_T_V = list(range(0, len(Alloc_T)))
for i in range(len(Alloc_T)):
    Alloc_T_V[i] = msol.get_value(Alloc_T[i])

CFlow_V = list(range(0, len(CFlow)))
for i in range(len(CFlow)):
    CFlow_V[i] = msol.get_value(CFlow[i])

# Print Solution:

print("Alloc: ", list(Alloc_V))
print("Succ: ", list(Succ_V))
print("Starttime: ", list(Starttime_V))
print("Alloc_T: ", list(Alloc_T_V))
print("CFlow: ", list(CFlow_V))


listone = AGV.copy()
listtwo = AGV.copy()
listtree = Alloc_T_V.copy()

for a in listone:
    listone[a - 1] = "%s%s" % ("DS", a)
    listtwo[a - 1] = "%s%s" % ("DE", a)
for b in list(range(len(listtree))):
    listtree[b] = "%s%s" % ("AGV", listtree[b])

tasks_col = listone + real_tasks + listtwo
agv_col = listtree
succ_col = Succ_V
succ_col[:] = [x - 1 for x in Succ_V]
succ_col = succ_col + listtwo
time_col = Starttime_V[:len(Starttime_V) - D] + listtwo

solution_df = pd.DataFrame(
    {'Tasks': tasks_col,
     'AGVs': agv_col,
     'Succ': succ_col,
     'Time': time_col
     })

solution_df['index'] = solution_df.index

print(solution_df)


# MIP

from collections import defaultdict
import pandas as pd
import numpy as np
import math
from docplex.cp.model import *
from sys import stdout
import matplotlib.pyplot as plt
from gurobipy import *
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import os
import rlcompleter
import sys
import random
import time
from os.path import join, getsize
from gurobipy import *
from pyreadline import Readline
from gurobipy import *


m = Model("repp")


# DECISION VARIABLES FROM CP

Alloc = Alloc_V
Succ = Succ_V
Starttime = Starttime_V
Alloc_T = Alloc_T_V
CFlow = CFlow_V

#####


def distance(p1, p2):
    distance = math.sqrt(((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2))
    return(distance)


new_edges_df = pd.DataFrame(
    {'X': from_vector,
     'Y': to_vector,
     'Z': dist_vector
     })

list_edges = [tuple(r) for r in new_edges_df.values]

dfObj = pd.DataFrame(list_edges)
edges_0 = list(dfObj[0])
edges_1 = list(dfObj[1])
edges_2 = list(dfObj[2])

appenddf1 = pd.DataFrame(
    {'X': edges_1,
     'Y': edges_0,
     'Z': edges_2
     })
appenddf2 = pd.DataFrame(
    {'X': edges_0,
     'Y': edges_1,
     'Z': edges_2
     })

dfObj = appenddf2.append(pd.DataFrame(data=appenddf1), ignore_index=True)
dfObj = pd.DataFrame(dfObj)

edges_a = Nodes_id
edges_b = Nodes_id
edges_c = [0] * (len(Nodes_id))

appenddf3 = pd.DataFrame(
    {'X': edges_a,
     'Y': edges_b,
     'Z': edges_c
     })

dfObj_wait = dfObj.append(pd.DataFrame(data=appenddf3), ignore_index=True)
dfObj_wait = pd.DataFrame(dfObj_wait)
dfObj['index_col'] = dfObj.index
dfObj_wait['index_col'] = dfObj_wait.index

wait_df = dfObj_wait[(len(appenddf1) * 2):]

#==============PREPROCESSING===============#

setOfAGVs = AGV
setOfPeriods = list(range(0, M))
setOfPeriods2 = list(range(1, M))
setOfArcs = list(dfObj['index_col'])
ArcsPlus = list(dfObj_wait['index_col'])
setOfNodes = Nodes_id


def Opp(a):
    # print(list(dfObj.loc[dfObj['index_col'] == a]['X']))
    dfObj2 = dfObj_wait[dfObj_wait['Y'].isin(list(dfObj_wait.loc[dfObj_wait['index_col'] == a]['X']))]
    b = list(dfObj2[dfObj2['X'].isin(list(dfObj_wait.loc[dfObj_wait['index_col'] == a]['Y']))]['index_col'])
    c = b[0]
    return(c)


def Wait(a):
    b = list(wait_df.loc[wait_df['X'] == a]['index_col'])
    c = b[0]
    return(c)

#============DECISION VARIABLES============#


# print(setOfPeriods)
# print(setOfAGVs)
# print(ArcsPlus)
# print(setOfArcs)
# print(dfObj)
# print(dfObj_wait)

# x[t, k, a]
dfObj = dfObj[0:len(setOfArcs)]

x = {}
for t in setOfPeriods:
    for k in setOfAGVs:
        for a in ArcsPlus:
            x[t, k, a] = m.addVar(obj=0.0, vtype='B', name='x[%s,%s,%s]' % (t, k, a))

z = {}
z = m.addVar(obj=1.0, vtype='C', name="z")

m.update()

#==============CONSTRAINTS==============#


# 1
for k in setOfAGVs:
    m.addConstr(quicksum(x[0, k, a] for a in list(dfObj.loc[dfObj['X'] == Start[k - 1]]['index_col'])), '=', 1)  # Thats correct

# 2
for k in setOfAGVs:
    for t in setOfPeriods:
        m.addConstr(quicksum(x[t, k, a] for a in ArcsPlus), '=', 1)  # No two vehicles can go on the same arc at one same time

# 3
for i in setOfNodes:
    for t in list(range(1, M)):  # ojo es M-1 it needs to be 0 because you need to link the first one
        for k in setOfAGVs:
            m.addConstr(quicksum(x[t, k, a] for a in list(dfObj_wait.loc[dfObj_wait['X'] == i]['index_col'])) - quicksum(x[t - 1, k, a] for a in list(dfObj_wait.loc[dfObj_wait['Y'] == i]['index_col'])), '=', 0)

# 4
for a in setOfArcs:
    for t in setOfPeriods:  # Need to add ir y venir the waiting nodes
        m.addConstr(quicksum(x[t, k, a] for k in setOfAGVs) + quicksum(x[t, k, Opp(a)] for k in setOfAGVs), '<=', 1)

# 5
for r in list(range(len(real_tasks))):
    m.addConstr(x[Starttime[r + D], Alloc_T[r + D], Wait(Nodes_Tasks[r])], '=', 1)


# 6 NO LONGER NECESSARY
# for r in list(range(len(requests))):
#    m.addConstr(x[Starttime[r + D + R], Alloc_T[r + D], Wait(W_d[r])], '=', 1)

# 7
for i in setOfNodes:
    for t in list(range(1, M)):
        m.addConstr(quicksum(x[t, k, a] for k in setOfAGVs for a in list(dfObj_wait.loc[dfObj_wait['X'] == i]['index_col'])), '<=', 1 + quicksum(1.0 for r in list(range(len(real_tasks))) if (t == Starttime[r + D] - 1) & (i == Nodes_Tasks[r])) * quicksum(1.0 for r in list(range(len(real_tasks))) if (t == Starttime[r + D] - 1) & (i == Nodes_Tasks[r])))


# 8 OBJECTIVE FUNCTION
m.addConstr(z, '=', quicksum(x[t, k, a] for k in setOfAGVs for t in setOfPeriods for a in ArcsPlus))


#m.params.TimeLimit = 100.0
m.update()
m.setObjective(z, GRB.MINIMIZE)
m.optimize()


# Print Solution
Var_Name = list(range(0, m.numVars))
Var_Value = list(range(0, m.numVars))
counter = 0

for v in m.getVars():
    Var_Name[counter] = v.varName
    Var_Value[counter] = v.x
    counter = counter + 1

result_df = pd.DataFrame(
    {'Var_Name': Var_Name,
     'Var_Value': Var_Value
     })

result_df = result_df.loc[result_df['Var_Value'] == 1]
print(result_df)

result_df_2 = result_df["Var_Name"].str.split("[", n=1, expand=True)
result_df_3 = result_df_2[1].str.split(",", n=1, expand=True)
result_df_4 = result_df_3[1].str.split(",", n=1, expand=True)
result_df_5 = result_df_4[1].str.split("]", n=1, expand=True)

col_1 = list(result_df['Var_Value'])
col_2 = list(result_df_2[0])
col_3 = list(result_df_3[0])
col_4 = list(result_df_4[0])
col_5 = list(result_df_5[0])

final_solution_df = pd.DataFrame(
    {'Variable': col_2,
     'time_period': col_3,
     'AGV': col_4,
     'index_col': col_5
     })
nodes_df_from = pd.DataFrame(
    {'from': Nodes_id,
     'x_from': x_nodes,
     'y_from': y_nodes
     })
nodes_df_to = pd.DataFrame(
    {'to': Nodes_id,
     'x_to': x_nodes,
     'y_to': y_nodes
     })

dfObj_wait["index_col"] = dfObj_wait["index_col"].astype('str')
final_solution_df = final_solution_df[final_solution_df['Variable'] == "x"]
results = final_solution_df.merge(dfObj_wait, on='index_col')
results.drop(columns=['Z'])
results.rename(columns={'X': 'from'}, inplace=True)
results.rename(columns={'Y': 'to'}, inplace=True)
results.rename(columns={'Y': 'dist'}, inplace=True)
results = results.merge(nodes_df_from, on='from')
results = results.merge(nodes_df_to, on='to')
results["time_period"] = pd.to_numeric(results["time_period"])
results["AGV"] = pd.to_numeric(results["AGV"])
results = results.sort_values(by=['time_period'])

results = results.iloc[:, 0:6]

print(results[results['AGV'] == 1])
print(results[results['AGV'] == 2])
print(results[results['AGV'] == 3])
print(results[results['AGV'] == 4])
print(results[results['AGV'] == 5])
print(results[results['AGV'] == 6])