# - - - - - - - - - - - - - - - - - - - - - IMPORTS - - - - - - - - - - - - - - - - - - - - - #
from math import log2

from matplotlib import cm, pyplot as plt
from z3 import *
import numpy as np
import networkx as nx


def exactly_one(variables):
    # At least one of the variables must be true
    at_least_one = Or(variables)

    # At most one of the variables must be true
    at_most_one = And(
        [Implies(variables[i], And([Not(variables[j]) for j in range(len(variables)) if j != i])) for i in
         range(len(variables))])

    return And(at_least_one, at_most_one)

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

# returns True if two lists of booleans are exactly equal
def equal_counts(a, b):
    if len(a) != len(b):
        return False
    return And([ai == bi for ai, bi in zip(a, b)])

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

def print_loads(model, print_routes, max_l, loads, s_item):
    print("\n- - Print Loads - -")
    print("Size Items: ", s_item)
    for k in range(len(max_l)):
        print(f"{k} - Max Load: {max_l[k]}")
        print(print_routes[k])
        print(f"Total Load: {model.evaluate(loads[k])}\n")

# Sum between a list of decision variables x and a list of booleans y
# x and y must be the binary representation of a integer number as a python list
# (it returns the result as a list or a string according to the value of "type"
def sum_binary(x, y, type = list):
    length = max(len(x), len(y))

    s = Solver()

    # both addends x and y must have same lenght

    a = [Bool(f"a_{i}") for i in range(length)]
    b = [Bool(f"b_{i}") for i in range(length)]


    for i in range(len(x)):
        if type(x[i]) != BoolRef:
            s.add(If(bool(x[i]), a[i], Not(a[i])))
        else:
            s.add(Implies(x[i], a[i]))
            s.add(Implies(Not(x[i]), Not(a[i])))

    for i in range(len(y)):
        if type(y[i]) != BoolRef:
            s.add(If(bool(y[i]), b[i], Not(b[i])))
        else:
            s.add(If(y[i], b[i], Not(b[i])))


    solution = [Bool(f"sol_{i}") for i in range(length+1)]
    carry = [Bool(f"carry_{i}") for i in range(length+1)]

    # sum constraints
    s.add(Not(carry[length]))

    for i in range(-1, -length-1, -1):
        # 0 + 1 + 1 = 0 with carry = 1  ///  0 + 1 + 0 = 1 with carry = 0
        s.add(Implies(Xor(a[i], b[i]), If(carry[i], Not(solution[i]), solution[i])))
        s.add(Implies(Xor(a[i], b[i]), If(carry[i], carry[i - 1], Not(carry[i - 1]))))

        # 1 + 1 + 1 = 1 with carry = 1  ///  0 + 1 + 1 = 0 with carry = 1
        s.add(Implies(And(a[i], b[i]), If(carry[i], solution[i], Not(solution[i]))))
        s.add(Implies(And(a[i], b[i]), carry[i - 1]))

        # 0 + 0 + 0 = 0 with carry 0   ///  0 + 0 + 1 = 1 with carry 0
        s.add(Implies(And(Not(a[i]), Not(b[i])), If(carry[i], solution[i], Not(solution[i]))))
        s.add(Implies(And(Not(a[i]), Not(b[i])), Not(carry[i - 1])))

    s.add(Implies(carry[0], solution[0]))

    if s.check() == sat:
        model = s.model()
        solution = [1 if model.evaluate(solution[i]) == True else 0 for i in range(length+1)]
        while solution and solution[0] == 0:
            solution.pop(0)
        if type == list:
            return "".join([str(item) for item in solution])
        else:
            return solution
    else:
        return False

def main(instance_num=1, remaining_time=300, upper_bound=None):
    n_couriers, n_items, max_load, size_item, all_distances = inputFile(instance_num)
    s = Solver()
    s.set("timeout", (int(remaining_time) * 1000))

    # Defining a graph which contain all the possible paths
    G = createGraph(all_distances)

    # decision variables
    x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in G.nodes] for i in G.nodes]  # x[i][j][k] == True : route (i->j) is used by courier k | set of Archs

    max_load_range = int(log2(max(max_load[:])))+1
    courier_loads = [[Bool(f"courier_loads_{k}_{n}") for k in range(n_couriers)] for n in range(max_load_range)]
    objective = Int('objective')



    # - - - - - - - - CONSTRAINTS - - - - - - - - #

    # No routes from any node to itself
    for k in range(n_couriers):
        s.add([Not(x[i][i][k]) for i in G.nodes])

    # Every item must be delivered (and only once)
    # (each 3-dimensional column and row must contain only 1 true value, depot not included in this constraint)
    for j in G.nodes:
        if j != 0:  # no depot
            s.add(exactly_one([x[i][j][k] for k in range(n_couriers) for i in G.nodes if i != j]))

    # Every node should be entered and left once and by the same vehicle
    # the fact that a vehicle enters a node implies that the node is also left from the same vehicle
    for k in range(n_couriers):
        for i in G.nodes:
            a = Sum([x[i][j][k] for j in G.nodes])
            b = Sum([x[j][i][k] for j in G.nodes])
            s.add(a == b)

    # each courier leaves and enters exactly once in the depot
    # (the number of predecessors and successors of the depot must be exactly one for each courier)
    for k in range(n_couriers):
        s.add(exactly_one([x[i][0][k] for i in G.nodes if i != 0]))
        s.add(exactly_one([x[0][j][k] for j in G.nodes if j != 0]))



    for k in range(n_couriers):
        #s.add(PbLe([(v[i][k], size_item[i+1]) for i in range(n_items)], max_load[k]))
        # s.add(courier_loads[k] == Sum([If(x[i][j][k], size_item[i],0) for i, j in G.edges]))
        # s.add(courier_loads[k] <= max_load[k])
        print(Sum([If(x[i][j][k], size_item[i],0) for i, j in G.edges]))
        current_load = format(Sum([If(x[i][j][k], size_item[i],0) for i, j in G.edges]),'b')
        max_load_bool = format(max_load[k],'b')
        for n in range(max_load_range):
            if n>len(current_load): s.add(courier_loads[k][n] == 0)
            else: s.add(courier_loads[k][n] == current_load[n])
        s.add(Or(courier_loads[k][:]))
        s.add(courier_loads[k][:] <= max_load_bool)



    # - - - - - - - - - - - - - - - - - NO SUBTOURS PROBLEM - - - - - - - - - - - - - - - - - - - - - - #
    # a represents the number of the package
    # b will contain a number of 1s proportional to the delivery order (the smaller the number of 1s, the earlier the delivery)
    u = [[Bool(f"u_{a}_{b}") for b in G.nodes] for a in G.nodes]

    # point zero is the first to be visited (depot)
    s.add(exactly_one(u[0][:]))

    # all the other points must be visited after the depot
    for i in G.nodes:
        if i != 0:  # excluding the depot
            s.add(at_least_k_seq(u[i][:], 2, "at_least_2"))

    # MTZ approach core
    for z in range(n_couriers):
        for i, j in G.edges:
            if i != 0 and j != 0 and i != j:  # excluding the depot
                s.add(Implies(x[i][j][z], Sum(u[i][:]) < Sum(u[j][:])))


    # - - - - - - - - - - - - - - - - - SOLVING - - - - - - - - - - - - - - - - - - - - - - #

    total_distance = Sum(
        [If(x[i][j][k], int(all_distances[i][j]), 0) for k in range(n_couriers) for i,j in G.edges])

    min_distance = Sum(
        [If(x[i][j][0], int(all_distances[i][j]), 0) for i,j in G.edges])

    max_distance = Sum(
        [If(x[i][j][0], int(all_distances[i][j]), 0) for i,j in G.edges])

    for k in range(n_couriers):
        temp = Sum(
            [If(x[i][j][k], int(all_distances[i][j]), 0) for i,j in G.edges])
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
    lower_bound = max(all_distances[0,:]) + max(all_distances[:,0])
    s.add(objective > lower_bound)

    if upper_bound is None:
        s.add(objective == max_distance)
    else:
        s.add(objective == max_distance)
        s.add(upper_bound > objective)



    if s.check() == sat:
        model = s.model()

        edges_list = []

        for z in range(n_couriers):
            tour_edges = [(i, j) for i, j in G.edges if model.evaluate(x[i][j][z])]
            edges_list += tour_edges
            print(f"Courier {z} tour (by Gian): ", tour_edges)
        print("- - - - - - - - - - - - - - - -")
        print("Upper bound: ", upper_bound)
        print("Objective: ", model.evaluate(objective))
        print("Min Distance: ", model.evaluate(min_distance))
        print("Max Distance: ", model.evaluate(max_distance))
        print("Total Distance: ", model.evaluate(total_distance))

        new_objective = model.evaluate(objective)

        routes = find_routes([], 0, edges_list, [])
        print("ROUTEEEES -> ", routes)

        return new_objective
    else:
        print("\nMERDA")
        return 0

inst = 1
temp = main(inst, 300)

for _ in range(10):
    temp = main(inst, 300, temp)
    if temp == 0:
        break