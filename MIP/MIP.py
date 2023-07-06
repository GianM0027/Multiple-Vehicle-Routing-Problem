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
    u = model.addVars(G.nodes, vtype=GRB.INTEGER, ub=n_items)

    # objective function (minimize total distance travelled + difference between min and max path normalized)

    maxTravelled = model.addVar(vtype=GRB.INTEGER, name="maxTravelled")
    minTravelled = model.addVar(vtype=GRB.INTEGER, name="minTravelled")
    for z in range(n_couriers):
        courierTravelled = quicksum(all_distances[i, j] * x[z][i, j] for i, j in G.edges)
        model.addConstr(courierTravelled <= maxTravelled)
        model.addConstr(courierTravelled >= minTravelled)


    #sumOfAllPaths = gp.LinExpr(quicksum(all_distances[i, j] * x[z][i, j] for z in range(n_couriers) for i, j in G.edges))
    #model.setObjective(sumOfAllPaths+(maxTravelled-minTravelled), GRB.MINIMIZE)
    model.setObjective(maxTravelled, GRB.MINIMIZE)

    ################################## CONSTRAINTS ####################################

    """
    # implied constraints
    # (each row for each courier must contain at most 1 true value, depot not included in this constraint)
    for z in range(n_couriers):
        for i in G.nodes:
            model.addConstr(quicksum(x[z][i, j] for j in G.nodes if i != j) <= 1)

    # same values of (i,j) cannot be true in different z (two couriers cannot travel the same sub-path)
    for i, j in G.edges:
        model.addConstr(quicksum(x[z][i, j] for z in range(n_couriers)) <= 1)
    
    # symmetry breaking constraint
    #couriers with lower max_load must bring less weight
    for z1 in range(n_couriers):
        for z2 in range(n_couriers):
            if max_load[z1] > max_load[z2]:
                model.addConstr(quicksum(size_item[j] * x[z1][i, j] for i, j in G.edges) >= quicksum(size_item[j] * x[z2][i, j] for i, j in G.edges))
    """

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


    #sub-tour elimination constraint
    # the depot is always the first point visited (??????????)
    model.addConstr(u[0] == 1)

    # all the other points must be visited after the depot (??????????)
    for i in G.nodes:
        if i != 0:  # excluding the depot
            model.addConstr(u[i] >= 2)

    #(??????????????????????????????????????????????????????????????)
    for z in range(n_couriers):
        for i, j in G.edges:
            if i != 0 and j != 0 and i != j:  # excluding the depot
                model.addConstr(x[z][i, j] * u[j] >= x[z][i, j] * (u[i] + 1))

    """ There are no explicit sub-tour elimination constraints. Instead, the constraints described above,
       along with the objective function (to minimize the total tour length), implicitly ensure that the resulting
       solution is a single tour without any sub-tours."""

    # start solving process
    # model.setParam("ImproveStartGap", 0.1)
    #model.tune()
    #model.setParam("MIPFocus", 1) #1 -> find feasible solutions quickly \ 2 -> focus on optimality \ focus on objective bound
    #gp.setParam("PoolSearchMode", 2)  # To find better intermediary solution
    #model.setParam('Presolve', 2)
    model.optimize()

    # print information about solving process (not verbose)
    print("\n\n\n#####################    SOLVER   ######################")
    print("Number of items: ", n_items)
    print("Number of couriers: ", n_couriers)
    print("Time taken: ", model.Runtime)
    print("Objective: ", model.ObjVal)
    #print("Min path: ", minTravelled.x)
    print("Max path: ", maxTravelled.x)
    if (model.status == GRB.OPTIMAL):
        print("Optimal solution found")
    else:
        print("Optimal solution not found")

    tot_item = []
    for z in range(n_couriers):
        tour_edges = [(i,j) for i,j in G.edges if x[z][i, j].x >= 1]
        items = []
        current = 0
        while len(tour_edges) > 0:
            for i, j in tour_edges:
                if i == current:
                    items.append(j)
                    current = j
                    tour_edges.remove((i,j))
        tot_item.append([i for i in items if i != 0])
    print("Solution: ",tot_item)

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
    print("u variable: ")
    for key in u.keys():
        print(f"{key}: ", u[key].X)
        
    # print general information about the problem instance
    print("Number of items: ", n_items)
    print("Number of couriers: ", n_couriers)
    print("Size_items: ", size_item)
    print("all_distances:\n", all_distances, "\n")
    """
    # print plots
    tour_edges = [edge for edge in G.edges for z in range(n_couriers) if x[z][edge].x >= 1]

    for z in range(n_couriers):
        print(f"courier {z}: ", [edge for edge in G.edges if x[z][edge].x >= 1], end=" ")
        print(" -> ", [all_distances[edge] for edge in G.edges if x[z][edge].x >= 1], end=" ")
        print(" -> ", quicksum(all_distances[i, j] * x[z][i, j].x for i, j in G.edges))
        print(max([sum(all_distances[i, j] * x[z][i, j].x for i, j in G.edges) for z in range(n_couriers)]))

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
main(12)
