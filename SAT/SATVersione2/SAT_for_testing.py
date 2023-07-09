# - - - - - - - - - - - - - - - - - - - - - IMPORTS - - - - - - - - - - - - - - - - - - - - - #
from z3 import *
import numpy as np
from math import log2
from itertools import combinations
import networkx as nx
import time
import json

# - - - - - - - - - - - - - - - - - - - - - CONFIGURATIONS - - - - - - - - - - - - - - - - - - - - - #
DEFAULT_MODEL = "defaultModel"
DEFAULT_IMPLIED_CONS = "impliedConsDefaultModel"
DEFAULT_SYMM_BREAK_CONS = "symmBreakDefaultModel"
DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS = "impliedAndSymmBreakDefaultModel"

configurations = [DEFAULT_MODEL, DEFAULT_IMPLIED_CONS, DEFAULT_SYMM_BREAK_CONS, DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS]


# - - - - - - - - - - - - - - - - - - - - - FUNCTIONS - - - - - - - - - - - - - - - - - - - - - #

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

    return routes


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

    return n_couriers, n_items, max_load, size_item, dist


def no_subtour(a, b):
    return And(b[0] == Not(a[0]), b[1] == Or(And(a[1], Not(a[0])), And(Not(a[1]), a[0])),
               b[2] == Or(And(a[2], Not(And(a[1], a[0]))), And(Not(a[2]), And(a[1], a[0]))))


at_most_one = at_most_one_seq
at_least_one = at_least_one_seq
exactly_one = exactly_one_seq


# - - - - - - - - - - - - - - - - - - - - - MODEL - - - - - - - - - - - - - - - - - - - - - #



def model(instance_num, configuration, remaining_time, solver_flag):
    obj = 0
    routes = []

    n_couriers, n_items, max_load, size_item, all_distances = inputFile(instance_num)

    if not solver_flag:
        s = Solver()
    else:
        s = Optimize()

    print(f"[{instance_num} - {configuration}] Remaining time: {remaining_time}")
    s.set("timeout", (int(remaining_time) * 1000))

    x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in
         range(n_items + 1)]  # x[i][j][k] == True : route (i->j) is used by courier k | set of Archs

    v = [[Bool(f"v_{i}_{k}") for k in range(n_couriers)] for i in range(n_items)]  # vehicle k is assigned to node i

    len_bin = int(log2(n_items)) + 1
    u = [[[Bool(f"u_{i}_{k}_{b}") for b in range(len_bin)] for k in range(n_couriers)] for i in
         range(n_items)]  # encoding in binary of order index of node i

    # - - - - - - - - - - - - - - - -  CONSTRAINTS - - - - - - - - - - - - -  - - - #

    # No routes from any node to itself
    for k in range(n_couriers):
        s.add([Not(x[i][i][k]) for i in range(n_items + 1)])

    # - - - - - - - - - - - - - - - - -
    # Each node (i, j) is visited only once
    # for each node there is exactly one arc entering and leaving from it
    """for i in range(1, n_items + 1):  # start from 1 to exclude the depot
        s.add(PbEq([(x[i][j][k], 1) for j in range(n_items + 1) for k in range(n_couriers)],
                   1))  # each node is left exactly once by each courier

    for j in range(1, n_items + 1):  # start from 1 to exclude the depot
        s.add(PbEq([(x[i][j][k], 1) for i in range(n_items + 1) for k in range(n_couriers)],
                   1))  # each node is entered exactly once by each courier"""

    # For each node there is exactly one arc entering and leaving from it
    for i in range(1, n_items + 1):
        s.add(exactly_one([x[i][j][k] for j in range(n_items + 1) for k in range(n_couriers)], f"arc_leave{i}"))

    # For each node there is exactly one arc entering and leaving from it
    for j in range(1, n_items + 1):
        s.add(exactly_one([x[i][j][k] for i in range(n_items + 1) for k in range(n_couriers)], f"arc_enter{j}"))

    # - - - - - - - - - - - - - - - - -

    # Each courier ends at the depot
    for k in range(n_couriers):
        #s.add(PbEq([(x[j][0][k], 1) for j in range(1, n_items + 1)], 1))
        s.add(exactly_one([x[j][0][k] for j in range(1, n_items + 1)], f"courier_ends_{k}"))

    # Each courier depart from the depot
    for k in range(n_couriers):
        #s.add(PbEq([(x[0][j][k], 1) for j in range(1, n_items + 1)], 1))
        s.add(exactly_one([x[0][j][k] for j in range(1, n_items + 1)], f"courier_starts_{k}"))

    # For each vehicle, the total load over its route must be smaller than its max load size
    for k in range(n_couriers):
        s.add(PbLe([(v[i][k], size_item[i]) for i in range(n_items)], max_load[k]))

    # Each item is carried by exactly one courier
    for i in range(n_items):
        #s.add(PbEq([(v[i][k], 1) for k in range(n_couriers)], 1))
        s.add(exactly_one([v[i][k] for k in range(n_couriers)], f"item_carried_{i}"))

    # If courier k goes to location (i, j), then courier k must carry item i, j
    for k in range(n_couriers):
        for i in range(1, n_items + 1):
            for j in range(1, n_items + 1):
                s.add([Implies(x[i][j][k], And(v[i - 1][k], v[j - 1][k]))])

    # Each courier carries at least one item
    for k in range(n_couriers):
        s.add(at_least_one_np([v[i][k] for i in range(n_items)]))

    # If (i, j) == True than --> for all the other k (i, j) != True
    if configuration == DEFAULT_IMPLIED_CONS or configuration == DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS:
        for i in range(n_items + 1):
            for j in range(n_items + 1):
                for k in range(n_couriers):
                    other_couriers = [k_prime for k_prime in range(n_couriers) if k_prime != k]
                    s.add(Implies(x[i][j][k], And([Not(x[i][j][k_prime]) for k_prime in other_couriers])))

        # For every courier, each row contains only one True
        for i in range(n_items + 1):
            for k in range(n_couriers):
                for j in range(n_items + 1):
                    other_destinations = [j_prime for j_prime in range(n_items + 1) if j_prime != j]
                    s.add(Implies(x[i][j][k], And([Not(x[i][j_prime][k]) for j_prime in other_destinations])))

    # - - - - - - - - - - - - - - - - - SYMMETRY BREAKING - - - - - - - - - - - - - - - - - - - - - - #

    if configuration == DEFAULT_SYMM_BREAK_CONS or configuration == DEFAULT_IMPLIED_AND_SYMM_BREAK_CONS:
        for k1 in range(n_couriers):
            for k2 in range(n_couriers):
                if k1 != k2:
                    load_k1 = Sum([If(v[i][k1], size_item[i], 0) for i in range(n_items)])
                    load_k2 = Sum([If(v[i][k2], size_item[i], 0) for i in range(n_items)])
                    s.add(Implies(max_load[k1] < max_load[k2], load_k1 <= load_k2))

    # - - - - - - - - - - - - - - - - - NO SUBTOURS PROBLEM - - - - - - - - - - - - - - - - - - - - - - #

    # The order of visiting locations must be consistent with the binary representations
    for k in range(n_couriers):
        s.add([Implies(x[i][j][k], no_subtour(u[i - 1][k], u[j - 1][k]))
               for i in range(n_items + 1) for j in range(n_items + 1)
               if len(u[i - 1][k]) >= 3 and len(u[j - 1][k]) >= 3 if i != j])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    start_time = time.time()

    total_distance = Sum(
        [If(x[i][j][k], int(all_distances[i][j]), 0) for k in range(n_couriers) for i in range(n_items + 1) for j in
         range(n_items + 1)])

    min_distance = Sum(
        [If(x[i][j][0], int(all_distances[i][j]), 0) for i in range(n_items + 1) for j in range(n_items + 1)])
    max_distance = Sum(
        [If(x[i][j][0], int(all_distances[i][j]), 0) for i in range(n_items + 1) for j in range(n_items + 1)])

    objective = None
    if solver_flag:
        for k in range(n_couriers):
            temp = Sum(
                [If(x[i][j][k], int(all_distances[i][j]), 0) for i in range(n_items + 1) for j in range(n_items + 1)])
            min_distance = If(temp < min_distance, temp, min_distance)
            max_distance = If(temp > max_distance, temp, max_distance)

        objective = s.minimize(total_distance + (max_distance - min_distance))

    if s.check() == sat:
        elapsed_time = time.time() - start_time
        model = s.model()

        # - - - - - - - - - - - - - - #

        min_dist = float('inf')
        max_dist = float('-inf')

        for k in range(n_couriers):
            temp = sum(int(all_distances[i][j]) for i in range(n_items + 1) for j in range(n_items + 1) if
                       model.evaluate(x[i][j][k]))

            if temp < min_dist:
                min_dist = temp

            if temp > max_dist:
                max_dist = temp

        # - - - - - - - - - - - - - - #

        edges_list = [(i, j) for z in range(n_couriers) for i in range(n_items + 1) for j in range(n_items + 1) if
                      model.evaluate(x[i][j][z])]

        routes = find_routes([], 0, edges_list, [])

        if solver_flag:
            obj = int(str(s.lower(objective)))
        else:
            obj = int(str(model.evaluate(total_distance))) + (int(str(model.evaluate(max_distance))) - int(str(model.evaluate(min_distance))))

        return obj, elapsed_time, routes

    else:
        obj = -1
        return obj, time.time() - start_time, []


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def find_best(instance, config):
    print("Stated  to find a solution")
    best_total_distance, elapsed_time, best_solution = model(instance, config, 300, False)
    remaining_time = 300 - elapsed_time
    status = False
    final_time = elapsed_time
    if remaining_time < 0:
        return 300, status, 0, []

    if remaining_time > 0:
        temp_total_distance, elapsed_time, temp_solution = model(instance, config, remaining_time, True)
        print("temp_total_distance", temp_total_distance)
        print("best_total_distance", best_total_distance)
        remaining_time = remaining_time - elapsed_time
        if temp_total_distance != -1 and remaining_time > 0:
            status = True
            if temp_total_distance <= best_total_distance:
                best_total_distance = temp_total_distance
                best_solution = temp_solution
                final_time = 300 - remaining_time

    if not best_solution:
        final_time = 300
        best_total_distance = 0

    solution_route = []
    for i in range(len(best_solution)):
        temp_route = []
        for s in best_solution[i]:
            temp_route.append(s[0])
        temp_route = temp_route[1:]
        solution_route.append(temp_route)

    return final_time, status, best_total_distance, solution_route


def main():
    # number of instances over which iterate
    n_istances = 21
    test_instances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 19]
    single_test = [1]

    for instance in single_test:
        inst = {}
        count = 1
        for configuration in configurations:
            print(
                f"\n\n\n###################    Instance {instance}/{n_istances}, Configuration {count} out of {len(configurations)} -> {configuration}    ####################")
            run_time, status, obj, solution = find_best(instance, configuration)

            # JSON
            config = {}
            config["time"] = int(run_time)
            config["optimal"] = status
            config["obj"] = obj
            config["solution"] = solution

            inst[configuration] = config
            count += 1

        if not os.path.exists("res_testFinale/"):
            os.makedirs("res_testFinale/")
        with open(f"res_testFinale/{instance}.JSON", "w") as file:
            file.write(json.dumps(inst, indent=3))


main()
