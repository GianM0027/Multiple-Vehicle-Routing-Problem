# - - - - - - - - - - - - - - - - - - - - - IMPORTS - - - - - - - - - - - - - - - - - - - - - #
from z3 import *
import numpy as np
from math import log2
from itertools import combinations
import networkx as nx
import time
import json


def exactly_one(variables):
    # At least one of the variables must be true
    at_least_one = Or(variables)

    # At most one of the variables must be true
    at_most_one = And(
        [Implies(variables[i], And([Not(variables[j]) for j in range(len(variables)) if j != i])) for i in
         range(len(variables))])

    return And(at_least_one, at_most_one)


def inputFile(num):
    # Instantiate variables from file
    if num < 10:
        instances_path = "instances/inst0" + str(num) + ".dat"  # inserire nome del file
    else:
        instances_path = "instances/inst" + str(num) + ".dat"  # inserire nome del file

    data_file = open(instances_path)
    lines = []
    for line in data_file:
        lines.append(line)
    data_file.close()
    n_couriers = int(lines[0].rstrip('\n'))
    n_items = int(lines[1].rstrip('\n'))
    max_load = list(map(int, lines[2].rstrip('\n').split()))
    size_item = list(map(int, lines[3].rstrip('\n').split()))
    for i in range(4, len(lines)):
        lines[i] = lines[i].rstrip('\n').split()

    for i in range(4, len(lines)):
        lines[i] = [lines[i][-1]] + lines[i]
        del lines[i][-1]

    dist = np.array([[lines[j][i] for i in range(len(lines[j]))] for j in range(4, len(lines))])

    last_row = dist[-1]
    dist = np.insert(dist, 0, last_row, axis=0)
    dist = np.delete(dist, -1, 0)

    dist = dist.astype(int)

    return n_couriers, n_items, max_load, [0] + size_item, dist


def find_routes(routes, current_node, remaining_edges, current_route):
    if current_node == 0 and len(current_route) > 1:
        routes.append(list(current_route))
    else:
        for i in range(len(remaining_edges)):
            if remaining_edges[i][0] == current_node:
                next_node = remaining_edges[i][1]
                current_route.append(remaining_edges[i])
                find_routes(routes, next_node, remaining_edges[:i] + remaining_edges[i + 1:], current_route)
                current_route.pop()

    solution_route = []
    for i in range(len(routes)):
        temp_route = []
        for s in routes[i]:
            temp_route.append(s[0])
        temp_route = temp_route[1:]
        solution_route.append(temp_route)

    return solution_route


def createGraph(all_distances):
    all_dist_size = all_distances.shape[0]
    size_item = all_distances.shape[0] - 1
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(all_dist_size))

    # Add double connections between nodes
    for i in range(all_dist_size):
        for j in range(i + 1, size_item + 1):  # size item + 1 because we enclude also the depot in the graph
            G.add_edge(i, j)
            G.add_edge(j, i)

    # Assign edge lengths
    lengths = {(i, j): all_distances[i][j] for i, j in G.edges()}
    nx.set_edge_attributes(G, lengths, 'length')

    return G


def print_loads(print_routes, max_l, s_item):
    print("\n- - Print Loads - -")
    print("Size Items: ", s_item)
    for r in print_routes:
        print(f"{print_routes.index(r)} - Max Load: {max_l[print_routes.index(r)]}")
        load = 0
        print(r)
        for i in range(1, len(r)):
            load += s_item[r[i]]
        print(f"Total Load: {load}\n")


def main(instance_num=1, remaining_time=300, upper_bound=None):
    n_couriers, n_items, max_load, size_item, all_distances = inputFile(instance_num)
    s = Solver()
    s.set("timeout", (int(remaining_time) * 1000))

    # Defining a graph which contain all the possible paths
    G = createGraph(all_distances)

    x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in
         range(n_items + 1)]  # x[i][j][k] == True : route (i->j) is used by courier k | set of Archs

    u = [Int(f"u_{j}") for j in G.nodes]

    objective = Int('objective')

    # - - - - - - - - CONSTRAINTS - - - - - - - - #

    # No routes from any node to itself
    for k in range(n_couriers):
        s.add([Not(x[i][i][k]) for i in range(n_items + 1)])

    for i in range(1, n_items + 1):  # start from 1 to exclude the depot
        s.add(PbEq([(x[i][j][k], 1) for j in range(n_items + 1) for k in range(n_couriers)],
                   1))  # each node is left exactly once by each courier

    for j in range(1, n_items + 1):  # start from 1 to exclude the depot
        s.add(PbEq([(x[i][j][k], 1) for i in range(n_items + 1) for k in range(n_couriers)],
                   1))  # each node is entered exactly once by each courier

    # - - - - - - - - PARTE FATTA IN CHIAMATA - - - - - - - - #

    # Every item must be delivered
    # (each 3-dimensional column must contain only 1 true value, depot not included in this constraint)
    for j in G.nodes:
        if j != 0:  # no depot
            s.add(exactly_one([x[i][j][k] for k in range(n_couriers) for i in G.nodes if i != j]))

    # Every node should be entered and left once and by the same vehicle
    # (number of times a vehicle enters a node is equal to the number of times it leaves that node)
    for i in range(1, n_items + 1):  # start from 1 to exclude the depot
        s.add(PbEq([(x[i][j][k], 1) for j in range(n_items + 1) for k in range(n_couriers)],
                   1))  # each node is left exactly once by each courier

    for j in range(1, n_items + 1):  # start from 1 to exclude the depot
        s.add(PbEq([(x[i][j][k], 1) for i in range(n_items + 1) for k in range(n_couriers)],
                   1))  # each node is entered exactly once by each courier

    # each courier leaves and enters exactly once in the depot
    # (the number of predecessors and successors of the depot must be exactly one for each courier)
    for k in range(n_couriers):
        s.add(Sum([x[i][0][k] for i in range(n_items + 1) if i != 0]) == 1)
        s.add(Sum([x[0][j][k] for j in range(n_items + 1) if j != 0]) == 1)

    # For each vehicle, the total load over its route must be smaller than its max load size
    for k in range(n_couriers):
        #s.add(PbLe([(v[i][k], size_item[i+1]) for i in range(n_items)], max_load[k]))
        s.add(PbLe([(x[i+1][j+1][k], size_item[i]) for i in range(n_items) for j in range(n_items)],
                   max_load[k]))

    # - - - - - - - - - - - - - - - - - NO SUBTOURS PROBLEM - - - - - - - - - - - - - - - - - - - - - - #

    #  Miller-Tucker-Zemlin formulation
    for k in range(n_couriers):
        for i in range(n_items):
            for j in range(n_items):
                s.add(u[i] + If(x[i][j][k], 1, 0) <= u[j] + n_items * (1 - If(x[i][j][k], 1, 0)))
                s.add(u[i] > 0)

    # - - - - - - - MTZ BUSACCHI - - - - - - - #

    """s.add(u[0] == 1)

    # all the other points must be visited after the depot
    for i in G.nodes:
        if i != 0:  # excluding the depot
            s.add(u[i] >= 2)

    # MTZ approach core
    for z in range(n_couriers):
        for i, j in G.edges:
            if i != 0 and j != 0 and i != j:  # excluding the depot
                s.add(x[i][j][z] * u[j] >= x[i][j][z] * (u[i] + 1))"""

    # - - - - - - - - - - - - - - - - - SOLVING - - - - - - - - - - - - - - - - - - - - - - #

    total_distance = Sum(
        [If(x[i][j][k], int(all_distances[i][j]), 0) for k in range(n_couriers) for i in range(n_items + 1) for j in
         range(n_items + 1)])

    min_distance = Sum(
        [If(x[i][j][0], int(all_distances[i][j]), 0) for i in range(n_items + 1) for j in range(n_items + 1)])

    max_distance = Sum(
        [If(x[i][j][0], int(all_distances[i][j]), 0) for i in range(n_items + 1) for j in range(n_items + 1)])

    for k in range(n_couriers):
        temp = Sum(
            [If(x[i][j][k], int(all_distances[i][j]), 0) for i in range(n_items + 1) for j in range(n_items + 1)])
        min_distance = If(temp < min_distance, temp, min_distance)
        max_distance = If(temp > max_distance, temp, max_distance)

    if upper_bound is None:
        s.add(objective == Sum(total_distance, (max_distance - min_distance)))
    else:
        s.add(objective == Sum(total_distance, (max_distance - min_distance)))
        s.add(upper_bound > objective)

    if s.check() == sat:
        model = s.model()

        edges_list = [(i, j) for z in range(n_couriers) for i, j in G.edges if model.evaluate(x[i][j][z])]

        routes = find_routes([], 0, edges_list, [])

        print("\nEdge List: ", edges_list)
        print("Routes: ", routes)
        print("- - - - - - - - - - - - - - - -")
        print("Upper bound: ", upper_bound)
        print("Objective: ", model.evaluate(objective))
        print("Min Distance: ", model.evaluate(min_distance))
        print("Max Distance: ", model.evaluate(max_distance))
        print("Total Distance: ", model.evaluate(total_distance))

        print_loads(routes, max_load, size_item)

        new_objective = model.evaluate(objective)

        return new_objective
    else:
        print("\nMERDA")
        return 0


inst = 1
temp = main(inst, 300)

for _ in range(10):
    temp = main(inst, 300, temp)