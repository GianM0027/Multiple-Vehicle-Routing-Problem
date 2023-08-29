# - - - - - - - - - - - - - - - - - - - - - IMPORTS - - - - - - - - - - - - - - - - - - - - - #
from matplotlib import cm, pyplot as plt
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

configurations = [DEFAULT_MODEL]


# - - - - - - - - - - - - - - - - - - - - - FUNCTIONS - - - - - - - - - - - - - - - - - - - - - #
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


def print_graph(G, n_couriers, tour_edges, x, model):
    # Calculate the node colors
    colormap = cm._colormaps.get_cmap("Set3")
    node_colors = {}
    for z in range(n_couriers):
        for i, j in G.edges:
            if model.evaluate(x[i][j][z]):
                node_colors[i] = colormap(z)
                node_colors[j] = colormap(z)
    node_colors[0] = 'pink'
    # Convert to list to maintain order for nx.draw
    color_list = [node_colors[node] for node in G.nodes]

    nx.draw(G.edge_subgraph(tour_edges), with_labels=True, node_color=color_list)
    plt.show()


def print_loads(model, print_routes, max_l, loads, s_item):
    print("\n- - Print Loads - -")
    print("Size Items: ", s_item)
    for k in range(len(max_l)):
        print(f"{k} - Max Load: {max_l[k]}")
        print(print_routes[k])
        print(f"Total Load: {model.evaluate(loads[k])}\n")


# - - - - - - - - - - - - - - - - - - - - - MAIN - - - - - - - - - - - - - - - - - - - - - #

def find_model(instance_num, remaining_time=300, upper_bound=None):
    n_couriers, n_items, max_load, size_item, all_distances = inputFile(instance_num)
    s = Solver()
    s.set("timeout", (int(remaining_time) * 1000))

    # Defining a graph which contain all the possible paths
    G = createGraph(all_distances)

    # decision variables
    x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in
         range(n_items + 1)]  # x[i][j][k] == True : route (i->j) is used by courier k | set of Archs

    courier_loads = [Int(f"courier_loads_{k}") for k in range(n_couriers)]

    u = [Int(f"u_{j}") for j in G.nodes]

    objective = Int('objective')

    # - - - - - - - - - - - - - - - - CONSTRAINTS - - - - - - - - - - - - - - - - #

    # No routes from any node to itself
    for k in range(n_couriers):
        s.add([Not(x[i][i][k]) for i in range(n_items + 1)])

    # Every item must be delivered
    # (each 3-dimensional column must contain only 1 true value, depot not included in this constraint)
    for j in G.nodes:
        if j != 0:  # no depot
            s.add(exactly_one([x[i][j][k] for k in range(n_couriers) for i in G.nodes if i != j]))

    # Every node should be entered and left once and by the same vehicle
    # (number of times a vehicle enters a node is equal to the number of times it leaves that node)
    for k in range(n_couriers):
        for i in G.nodes:
            s1 = Sum([x[i][j][k] for j in G.nodes if i != j])
            s2 = Sum([x[j][i][k] for j in G.nodes if i != j])
            s.add(s1 == s2)

    # each courier leaves and enters exactly once in the depot
    # (the number of predecessors and successors of the depot must be exactly one for each courier)
    for k in range(n_couriers):
        s.add(Sum([x[i][0][k] for i in G.nodes if i != 0]) == 1)
        s.add(Sum([x[0][j][k] for j in G.nodes if j != 0]) == 1)

    # For each vehicle, the total load over its route must be smaller than its max load size
    for k in range(n_couriers):
        s.add(courier_loads[k] == Sum([If(x[i][j][k], size_item[i], 0) for i, j in G.edges]))
        s.add(courier_loads[k] > 0)
        s.add(courier_loads[k] <= max_load[k])

    # - - - - - - - - - - - - - - - - NO SUBTOUR PROBLEM - - - - - - - - - - - - - - - - #

    s.add(u[0] == 1)

    # all the other points must be visited after the depot
    for i in G.nodes:
        if i != 0:  # excluding the depot
            s.add(u[i] >= 2)

    # MTZ approach core
    for z in range(n_couriers):
        for i, j in G.edges:
            if i != 0 and j != 0 and i != j:  # excluding the depot
                s.add(x[i][j][z] * u[j] >= x[i][j][z] * (u[i] + 1))

    # - - - - - - - - - - - - - - - - SOLVING - - - - - - - - - - - - - - - - #

    total_distance = Sum(
        [If(x[i][j][k], int(all_distances[i][j]), 0) for k in range(n_couriers) for i, j in G.edges])

    min_distance = Sum(
        [If(x[i][j][0], int(all_distances[i][j]), 0) for i, j in G.edges])

    max_distance = Sum(
        [If(x[i][j][0], int(all_distances[i][j]), 0) for i, j in G.edges])

    for k in range(n_couriers):
        temp = Sum(
            [If(x[i][j][k], int(all_distances[i][j]), 0) for i, j in G.edges])
        min_distance = If(temp < min_distance, temp, min_distance)
        max_distance = If(temp > max_distance, temp, max_distance)

    """  
        # OBJECTIVE 1
        if upper_bound is None:
            s.add(objective == Sum(total_distance, (max_distance - min_distance)))
        else:
            s.add(objective == Sum(total_distance, (max_distance - min_distance)))
            s.add(upper_bound > objective)
        """

    # OBJECTIVE 2
    if upper_bound is None:
        s.add(objective == max_distance)
    else:
        s.add(objective == max_distance)
        s.add(upper_bound > objective)

    start_time = time.time()
    if s.check() == sat:
        elapsed_time = time.time() - start_time
        model = s.model()

        tot_item = []
        for z in range(n_couriers):
            tour_edges = [(i, j) for i, j in G.edges if model.evaluate(x[i][j][z])]
            items = []
            current = 0
            while len(tour_edges) > 0:
                for i, j in tour_edges:
                    if i == current:
                        items.append(j)
                        current = j
                        tour_edges.remove((i, j))
            tot_item.append([i for i in items if i != 0])
        print("Solution: ", tot_item)
        print("- - - - - - - - - - - - - - - -")
        print("Upper bound: ", upper_bound)
        print("Objective: ", model.evaluate(objective))
        print("Min Distance: ", model.evaluate(min_distance))
        print("Max Distance: ", model.evaluate(max_distance))
        print("Total Distance: ", model.evaluate(total_distance))

        new_objective = model.evaluate(objective)

        return elapsed_time, new_objective, tot_item
    else:
        print("minimal solution found")
        elapsed_time = time.time() - start_time
        return elapsed_time, -1, []


def find_best(instance, config):

    print("Stated  to find a solution, configuration: ", config)
    run_time, temp_obj, temp_solution = find_model(instance, 300, None)
    remaining_time = 300 - run_time
    print("remaining_time: ", remaining_time)
    best_obj, best_solution = temp_obj, temp_solution

    while remaining_time > 0:
        run_time, temp_obj, temp_solution = find_model(instance, remaining_time, temp_obj)
        remaining_time = remaining_time - run_time
        if temp_obj == -1:
            return remaining_time, True, str(best_obj), best_solution
        else:
            best_obj, best_solution = temp_obj, temp_solution

    print("time limit exceeded")
    return remaining_time, False, str(best_obj), best_solution


def main():
    # number of instances over which iterate
    n_istances = 21
    test_instances = [1, 2, 3, 4, 5]

    for instance in range(len(test_instances)):
        inst = {}
        count = 1
        for configuration in configurations:
            print(
                f"\n\n\n###################    Instance {instance + 1}/{n_istances}, Configuration {count} out of {len(configurations)} -> {configuration}    ####################")
            runTime, status, obj, solution = find_best(instance + 1, configuration)

            # JSON
            config = {}
            config["time"] = runTime
            config["optimal"] = status
            config["obj"] = obj
            config["solution"] = solution


            inst[configuration] = config
            count += 1

        if not os.path.exists("res/"):
            os.makedirs("res/")
        with open(f"res/{instance+1}.JSON", "w") as file:
            file.write(json.dumps(inst, indent=3))


main()