import json
import sys

import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import networkx as nx
import matplotlib.cm as cm
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


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

    # print general information about the problem instance
    print("Number of items: ", n_items)
    print("Number of couriers: ", n_couriers)
    print("")

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


def main(num):
    n_couriers, n_items, max_load, size_item, all_distances = inputFile(num)

    # model
    model = gp.Model()
    model.setParam('TimeLimit', 300)

    # Defining a graph which contain all the possible paths
    G = createGraph(all_distances)

    # decision variables
    x = [model.addVars(G.edges, vtype=gp.GRB.BINARY) for _ in range(n_couriers)]
    ordering = [model.addVars(G.nodes, vtype=GRB.INTEGER, ub=n_items) for _ in
                range(n_couriers)]  # ordering[z,i] ha valore p se i è la p-esima meta del corriere z

    # objective function (minimize total distance travelled + difference between min and max path normalized)
    maxTravelled = model.addVar(vtype=GRB.INTEGER, name="maxTravelled")
    minTravelled = model.addVar(vtype=GRB.INTEGER, name="minTravelled")
    for z in range(n_couriers):
        courierTravelled = quicksum(all_distances[i, j] * x[z][i, j] for i, j in G.edges)
        model.addConstr(courierTravelled <= maxTravelled)
        model.addConstr(courierTravelled >= minTravelled)
    sumOfAllPaths = gp.LinExpr(
        quicksum(all_distances[i, j] * x[z][i, j] for z in range(n_couriers) for i, j in G.edges))

    model.setObjective(sumOfAllPaths, GRB.MINIMIZE)
    model.setObjectiveN((maxTravelled - minTravelled), GRB.MINIMIZE)

    # CONSTRAINTS

    # Every item must be delivered
    # (each 3-dimensional raw must contain only 1 true value, depot not included in this constraint)
    for j in G.nodes:
        if j != 0:  # no depot
            model.addConstr(quicksum(x[z][i, j] for z in range(n_couriers) for i in G.nodes if i != j) == 1)

    # Every node should be entered and left once and by the same vehicle
    # (number of times a vehicle enters a node is equal to the number of times it leaves that node)
    for z in range(n_couriers):
        for i in G.nodes:
            model.addConstr(quicksum(x[z][i, j] - x[z][j, i] for j in G.nodes if i != j) == 0)

    # each courier leaves and enters exactly once in the depot
    # (the number of predecessors and successors of the depot must be exactly one for each courier)
    for z in range(n_couriers):
        model.addConstr(quicksum(x[z][i, 0] for i in G.nodes if i != 0) == 1)
        model.addConstr(quicksum(x[z][0, j] for j in G.nodes if j != 0) == 1)

    # each courier does not exceed its max_load
    # sum of size_items must be minor than max_load for each courier
    for z in range(n_couriers):
        model.addConstr(quicksum(size_item[j] * x[z][i, j] for i, j in G.edges) <= max_load[z])

    # subtour elimination

    # item delivered by each courier
    # items_delivered = [sum(x[z][i, j].x for i, j in G.edges) for z in range(n_couriers)]

    # the depot is always the first point visited
    for z in range(n_couriers):
        model.addConstr(ordering[z][0] == 0)

    # all the other points must be visited after the depot
    for z in range(n_couriers):
        for i in G.nodes:
            if i != 0:  # excluding the depot
                model.addConstr(ordering[z][i] >= 1)

    # delivery ordering
    for z in range(n_couriers):
        for i, j in G.edges:
            if i != 0 and j != 0:  # excluding the depot and self loops
                #if j is delivered after i then priority of j >= than priority of i
                model.addConstr(x[z][i, j]*ordering[z][j] >= x[z][i, j]*(ordering[z][i]+1))

    # start solving process
    # model.setParam("MIPFocus", 0)
    # model.setParam("ImproveStartGap", 0.1)
    #model.tune()
    model.optimize()

    # print information about solving process (not verbose)
    print("\n\n\n#####################    SOLVER   ######################")
    print("Number of items: ", n_items)
    print("Number of couriers: ", n_couriers)
    print("Time taken: ", model.Runtime)
    print("Objective: ", model.ObjVal)
    print("Min path: ", minTravelled.x)
    print("Max path: ", maxTravelled.x)
    if (model.status == GRB.OPTIMAL):
        print("Optimal solution found")
    else:
        print("Optimal solution not found")

    tot_item = []
    for z in range(n_couriers):
        item = []
        for i, j in G.edges:
            if x[z][i, j].x >= 1:
                if i not in item:
                    item.append(i)
                if j not in item:
                    item.append(j)
        tot_item.append([i for i in item if i != 0])
    print(tot_item)

    """ JSON
    output = {}
    output["time"] = model.Runtime
    output["optimal"] = model.status == GRB.OPTIMAL
    output["obj"] = model.ObjVal
    output["solution"] = tot_item
    print(json.dumps(output))"""

    """
    #print information about solving process (verbose)
    print("\n\n\n###############################################################################")

    # print general information about each courier
    for z in range(n_couriers):
        print(f"\nCourier {z}: ")
        print("Max load: ", max_load[z])
        print("Final load: ", quicksum(size_item[j] * x[z][i, j].x for i,j in G.edges))
        print("Total distance: ", quicksum(all_distances[i,j] * x[z][i, j].x for i,j in G.edges))

    # print ordering variables
    z = 0
    for dictionary in ordering:
        print(f"\nCourier: {z}")
        for key in dictionary.keys():
            print(f"{key}:", dictionary[key].X)
        print("")
        z+=1

    # print general information about the problem instance
    print("Number of items: ", n_items)
    print("Number of couriers: ", n_couriers)
    print("Size_items: ", size_item)
    print("all_distances:\n", all_distances, "\n")
    """
    # print plots
    tour_edges = [edge for edge in G.edges for z in range(n_couriers) if x[z][edge].x >= 1]

    # Calculate the node colors
    colormap = cm._colormaps.get_cmap("Set3")
    node_colors = {}
    for z in range(n_couriers):
        for i, j in G.edges:
            if x[z][i, j].x >= 1:
                node_colors[i] = colormap(z)
                node_colors[j] = colormap(z)
    node_colors[0] = 'pink'
    # Convert to list to maintain order for nx.draw
    color_list = [node_colors[node] for node in G.nodes]

    nx.draw(G.edge_subgraph(tour_edges), with_labels=True, node_color=color_list)
    plt.show()


    print("############################################################################### \n")


# passare come parametro solo numero dell'istanza (senza lo 0)
main(3)
