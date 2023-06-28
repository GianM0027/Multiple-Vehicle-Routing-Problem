import sys

import matplotlib.colors
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
        instances_path = "instances/inst0"+str(num)+".dat"  # inserire nome del file
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

    return n_couriers, n_items, max_load, [0]+size_item, dist



def createGraph(all_distances):
    all_dist_size = all_distances.shape[0]
    size_item = all_distances.shape[0]-1
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(all_dist_size))

    # Add double connections between nodes
    for i in range(all_dist_size):
        for j in range(i + 1, size_item + 1): #size item + 1 because we enclude also the depot in the graph
            G.add_edge(i, j)
            G.add_edge(j, i)

    # Assign edge lengths
    lengths = {(i, j): all_distances[i][j] for i, j in G.edges()}
    nx.set_edge_attributes(G, lengths, 'length')

    return G


def main(num):
    n_couriers, n_items, max_load, size_item, all_distances = inputFile(num)

    #model
    model = gp.Model()

    # Defining a graph which contain all the possible paths
    G = createGraph(all_distances)



    #decision variables
    x = [model.addVars(G.edges, vtype=gp.GRB.BINARY) for _ in range(n_couriers)]
    ordering = [model.addVars(G.nodes, vtype=GRB.INTEGER) for _ in range(n_couriers)] #ordering[z,i] ha valore p se i è la p-esima meta del corriere z



    #objective function (minimize total distance travelled + difference between min and max path normalized)
    maxTravelled = model.addVar(vtype=GRB.CONTINUOUS, name="maxTravelled")
    minTravelled = model.addVar(vtype=GRB.CONTINUOUS, name="minTravelled")
    for z in range(n_couriers):
        courierTravelled = quicksum(all_distances[i, j] * x[z][i, j] for i, j in G.edges)
        model.addConstr(courierTravelled <= maxTravelled, f"maxTravelledConstr_{z}")
        model.addConstr(courierTravelled >= minTravelled, f"minTravelledConstr_{z}")
    sumOfAllPaths = gp.LinExpr(quicksum(all_distances[i, j] * x[z][i, j] for z in range(n_couriers) for i, j in G.edges))
    model.setObjective(sumOfAllPaths+n_items*(maxTravelled - minTravelled), GRB.MINIMIZE)



    #CONSTRAINTS

    # Every item must be delivered
    # (each 3-dimensional raw, must contain only 1 true value, depot not included in this constraint)
    for j in G.nodes:
        if j != 0: #no depot
            model.addConstr(quicksum(x[z][i, j] for z in range(n_couriers) for i in G.nodes if i != j) == 1)

    # Every node should be entered and left once and by the same vehicle
    # (number of times a vehicle enters a node is equal to the number of times it leaves that node)
    for z in range(n_couriers):
        for i in G.nodes:
            model.addConstr(quicksum(x[z][i, j]-x[z][j,i] for j in G.nodes if i != j) == 0)

    # each courier leaves and enters exactly once in the depot
    # (the number of predecessors and successors of the depot must be exactly one for each courier)
    for z in range(n_couriers):
        model.addConstr(quicksum(x[z][i, 0] for i in G.predecessors(0)) == 1)
        model.addConstr(quicksum(x[z][0, j] for j in G.successors(0)) == 1)

    # each courier does not exceed its max_load
    # sum of size_items must be minor than max_load for each courier
    for z in range(n_couriers):
        model.addConstr(quicksum(size_item[j] * x[z][i, j] for i,j in G.edges) <= max_load[z])


    # subtour elimination (Miller-Tucker-Zemlin formulation)

    # item delivered by each courier
    #items_delivered = [sum(x[z][i, j].x for i, j in G.edges) for z in range(n_couriers)]

    # the depot is always the first point visited
    for z in range(n_couriers):
        model.addConstr(ordering[z][0] == 0)

    # all the other points must be visited after the depot
    for z in range(n_couriers):
        for i in G.nodes:
            if i != 0:  # excluding the depot
                model.addConstr(ordering[z][i] >= 1)

    # MTZ for delivery ordering
    for z in range(n_couriers):
        for i,j in G.edges:
            if i != j and (i != 0 and j != 0):  # excluding the depot and self loops
                model.addConstr(ordering[z][i] - ordering[z][j] + 1 <= (1 - x[z][i, j]) * quicksum(x[z][k, l] for k,l in G.edges))



    # start solving process
    model.optimize()


    #print information about solving process
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
    print("all_distances:\n", all_distances, "\n")
    print("Size_items: ", size_item)

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


    print(minTravelled.x)
    print(maxTravelled.x)

    nx.draw(G.edge_subgraph(tour_edges), with_labels=True, node_color=color_list)
    plt.show()


    print("############################################################################### \n")




#passare come parametro solo numero dell'istanza (senza lo 0)
main(5)


"""
Euristiche per speed up:


Warm start: If you have a feasible solution already, you can use that as a starting point for the solver. This can often help to significantly speed up the solution process, especially if your initial solution is good. This is not applicable if you don't have an initial feasible solution.

Adjust solver parameters: There are various parameters in Gurobi that can influence the solving process. Some of them that could be potentially interesting for your problem are:

MIPFocus: This parameter lets you shift the focus of the solver towards finding feasible solutions quickly (MIPFocus=1), proving optimality (MIPFocus=2), or improving the best bound (MIPFocus=3). You can experiment with these settings and see what works best for your problem.
Heuristics: This parameter controls the amount of time spent in MIP heuristics. You could increase this parameter to find better feasible solutions early.
Cuts: The aggressiveness of cut generation can be controlled via the Cuts parameter. Cuts can help to improve the LP relaxation bound, but generating and adding them into the model takes time.
Preprocessing: Gurobi's presolver can often simplify the problem before it is solved, by removing redundant constraints and variables, tightening bounds, etc. This is generally helpful, but in some cases it might be beneficial to turn off some or all of the presolve.

Solution Pool: Gurobi's solution pool feature allows you to gather more than one solution during the MIP solve process. You can control how many solutions you want to collect and also the quality of these solutions. This could be useful if you are more interested in finding good solutions quickly rather than proving optimality.

Using a heuristic solution: You could consider developing a heuristic to find a quick, possibly suboptimal solution, and then feed that solution to the MIP solver as a starting solution. The heuristic could be based on domain-specific knowledge, or a simplification of the problem. For example, you might solve a relaxed version of the problem (ignoring some constraints) as a heuristic.
"""