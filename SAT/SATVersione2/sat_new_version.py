# - - - - - - - - - - - - - - - - - - - - - IMPORTS - - - - - - - - - - - - - - - - - - - - - #
from z3 import *
import numpy as np
from math import log2
from itertools import combinations
import networkx as nx
import time
import json


def toBinary(num, length=None):
    num_bin = bin(num).split("b")[-1]
    if length:
        return "0" * (length - len(num_bin)) + num_bin
    return num_bin


def at_least_one_bw(bool_vars):
    return at_least_one_np(bool_vars)


def at_most_one_bw(bool_vars, name):
    constraints = []
    n = len(bool_vars)
    m = math.ceil(math.log2(n))
    r = [Bool(f"r_{name}_{i}") for i in range(m)]
    binaries = [toBinary(i, m) for i in range(n)]
    for i in range(n):
        for j in range(m):
            phi = Not(r[j])
            if binaries[i][j] == "1":
                phi = r[j]
            constraints.append(Or(Not(bool_vars[i]), phi))
    return And(constraints)


def exactly_one_bw(bool_vars, name):
    return And(at_least_one_bw(bool_vars), at_most_one_bw(bool_vars, name))


def at_least_one_seq(bool_vars):
    return at_least_one_np(bool_vars)


def at_most_one_seq(bool_vars, name):
    constraints = []
    n = len(bool_vars)
    s = [Bool(f"s_{name}_{i}") for i in range(n - 1)]
    constraints.append(Or(Not(bool_vars[0]), s[0]))
    constraints.append(Or(Not(bool_vars[n - 1]), Not(s[n - 2])))
    for i in range(1, n - 1):
        constraints.append(Or(Not(bool_vars[i]), s[i]))
        constraints.append(Or(Not(bool_vars[i]), Not(s[i - 1])))
        constraints.append(Or(Not(s[i - 1]), s[i]))
    return And(constraints)


def exactly_one_seq(bool_vars, name):
    return And(at_least_one_seq(bool_vars), at_most_one_seq(bool_vars, name))


def at_least_one_np(bool_vars):
    return Or(bool_vars)


def at_most_one_np(bool_vars, name=""):
    return And([Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)])


def exactly_one_np(bool_vars, name=""):
    return And(at_least_one_np(bool_vars), at_most_one_np(bool_vars, name))


def at_least_one_he(bool_vars):
    return at_least_one_np(bool_vars)


def at_most_one_he(bool_vars, name):
    if len(bool_vars) <= 4:
        return And(at_most_one_np(bool_vars))
    y = Bool(f"y_{name}")
    return And(And(at_most_one_np(bool_vars[:3] + [y])), And(at_most_one_he(bool_vars[3:] + [Not(y)], name + "_")))


def exactly_one_he(bool_vars, name):
    return And(at_most_one_he(bool_vars, name), at_least_one_he(bool_vars))


def at_least_k_seq(bool_vars, k, name):
    return at_most_k_seq([Not(var) for var in bool_vars], len(bool_vars) - k, name)


def at_most_k_seq(bool_vars, k, name):
    constraints = []
    n = len(bool_vars)
    s = [[Bool(f"s_{name}_{i}_{j}") for j in range(k)] for i in range(n - 1)]
    constraints.append(Or(Not(bool_vars[0]), s[0][0]))
    constraints += [Not(s[0][j]) for j in range(1, k)]
    for i in range(1, n - 1):
        constraints.append(Or(Not(bool_vars[i]), s[i][0]))
        constraints.append(Or(Not(s[i - 1][0]), s[i][0]))
        constraints.append(Or(Not(bool_vars[i]), Not(s[i - 1][k - 1])))
        for j in range(1, k):
            constraints.append(Or(Not(bool_vars[i]), Not(s[i - 1][j - 1]), s[i][j]))
            constraints.append(Or(Not(s[i - 1][j]), s[i][j]))
    constraints.append(Or(Not(bool_vars[n - 1]), Not(s[n - 2][k - 1])))
    return And(constraints)


def exactly_k_seq(bool_vars, k, name):
    return And(at_most_k_seq(bool_vars, k, name), at_least_k_seq(bool_vars, k, name))


at_most_one = at_most_one_seq
at_least_one = at_least_one_seq
exactly_one = exactly_one_seq


def no_subtour(a, b):
    return And(b[0] == Not(a[0]), b[1] == Or(And(a[1], Not(a[0])), And(Not(a[1]), a[0])),
               b[2] == Or(And(a[2], Not(And(a[1], a[0]))), And(Not(a[2]), And(a[1], a[0]))))


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


# - - - - - - - - - - - - - - - - - - - - - MAIN - - - - - - - - - - - - - - - - - - - - - #

def main(instance_num=1, remaining_time=300, upper_bound=None):
    n_couriers, n_items, max_load, size_item, all_distances = inputFile(instance_num)

    s = Solver()

    s.set("timeout", (int(remaining_time) * 1000))

    x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in
         range(n_items + 1)]  # x[i][j][k] == True : route (i->j) is used by courier k | set of Archs

    v = [[Bool(f"v_{i}_{k}") for k in range(n_couriers)] for i in range(n_items)]  # vehicle k is assigned to node i

    len_bin = int(log2(n_items)) + 1
    u = [[[Bool(f"u_{i}_{k}_{b}") for b in range(len_bin)] for k in range(n_couriers)] for i in
         range(n_items)]  # encoding in binary of order index of node i

    # - - - - - - - - CONSTRAINTS - - - - - - - - #

    for k in range(n_couriers):  # No routes from any node to itself
        s.add([Not(x[i][i][k]) for i in range(n_items + 1)])

    for i in range(1, n_items + 1):  # For each node there is exactly one arc entering and leaving from it
        s.add(exactly_one([x[i][j][k] for j in range(n_items + 1) for k in range(n_couriers)], f"arc_leave{i}"))

    for j in range(1, n_items + 1):  # For each node there is exactly one arc entering and leaving from it
        s.add(exactly_one([x[i][j][k] for i in range(n_items + 1) for k in range(n_couriers)], f"arc_enter{j}"))

    for k in range(n_couriers):  # Each courier ends at the depot
        s.add(exactly_one([x[j][0][k] for j in range(1, n_items + 1)], f"courier_ends_{k}"))

    for k in range(n_couriers):  # Each courier depart from the depot
        s.add(exactly_one([x[0][j][k] for j in range(1, n_items + 1)], f"courier_starts_{k}"))

    # For each vehicle, the total load over its route must be smaller than its max load size
    for k in range(n_couriers):
        s.add(PbLe([(v[i][k], size_item[i]) for i in range(n_items)], max_load[k]))

    for i in range(n_items):  # Each item is carried by exactly one courier
        s.add(exactly_one([v[i][k] for k in range(n_couriers)], f"item_carried_{i}"))

    for k in range(n_couriers):  # If courier k goes to location (i, j), then courier k must carry item i, j
        for i in range(1, n_items + 1):
            for j in range(1, n_items + 1):
                s.add([Implies(x[i][j][k], And(v[i - 1][k], v[j - 1][k]))])

    for k in range(n_couriers):  # Each courier carries at least one item
        s.add(at_least_one_seq([v[i][k] for i in range(n_items)]))

    # - - - - - - - - - - - - - - - - - NO SUBTOURS PROBLEM - - - - - - - - - - - - - - - - - - - - - - #

    # The order of visiting locations must be consistent with the binary representations
    for k in range(n_couriers):  # The order of visiting locations must be consistent with the binary representations
        s.add([Implies(x[i][j][k], no_subtour(u[i - 1][k], u[j - 1][k]))
               for i in range(n_items + 1) for j in range(n_items + 1)
               if len(u[i - 1][k]) >= 3 and len(u[j - 1][k]) >= 3 if i != j])
    """
    for k in range(n_couriers):
        for i in range(1, n_items + 1):
            for j in range(1, n_items + 1):
                s.add(Implies(x[i][j][k], And(Or([x[j][f][k] for f in range(n_items + 1)]),
                                              Or([x[m][i][k] for m in range(n_items + 1)]))))
    """

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
        objective = Int('objective')
        s.add(objective >= Sum(total_distance, (max_distance - min_distance)))
    else:
        objective = Int('objective')
        s.add(objective >= Sum(total_distance, (max_distance - min_distance)))
        s.add(upper_bound > objective)

    if s.check() == sat:
        model = s.model()

        edges_list = [(i, j) for z in range(n_couriers) for i in range(n_items + 1) for j in range(n_items + 1) if
                      model.evaluate(x[i][j][z])]

        routes = find_routes([], 0, edges_list, [])

        print("\nEdge List: ", edges_list)
        print("Routes: ", routes)
        print("- - - - - - - - - - - - - - - -")
        print("Upper bound: ", upper_bound)
        print("Objective: ", model.evaluate(objective))
        print("Min Distance: ", model.evaluate(min_distance))
        print("Max Distance: ", model.evaluate(max_distance))
        print("Total Distance: ", model.evaluate(total_distance))

        new_objective = model.evaluate(objective)

        return new_objective
    else:
        print("\nMERDA")
        return 0


inst = 1
solution = None
while type(solution) != int:
    solution = main(inst, 300, solution)




