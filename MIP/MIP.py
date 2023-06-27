import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import networkx as nx

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



    #objective function (minimize total distance travelled)
    model.setObjective(quicksum(all_distances[i, j] * x[z][i, j] for z in range(n_couriers) for i,j in G.edges),GRB.MINIMIZE)



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


    # subtour elimination (Explicit Dantzig-Fulkerson-Johnson formulation)

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
    """
    for z in range(n_couriers):
        tour_edges = [edge for edge in G.edges if x[z][edge].x >= 1]
        nx.draw(G.edge_subgraph(tour_edges), with_labels=True)
        plt.figure(z)
    plt.show()
    """

    print("############################################################################### \n")




#passare come parametro solo numero dell'istanza (senza lo 0)
main(12)
