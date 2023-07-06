import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, cm
from z3 import *



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

    return n_couriers, n_items, max_load, size_item, dist


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

    # Defining a graph which contain all the possible paths
    G = createGraph(all_distances)

    # model
    s = Optimize()

    # decision variables
    x = [[[Bool(f"x_{i}_{j}_{z}") for z in range(n_couriers)] for j in G.nodes] for i in G.nodes]
    u = [Int(f"u_{i}") for i in G.nodes]

    # objective function (minimize total distance travelled + difference between min and max path normalized)
    maxTravelled = Int("maxTravelled")
    minTravelled = Int("minTravelled")
    for z in range(n_couriers):
        courierTravelled = Sum([If(x[i][j][z], int(all_distances[i, j]), 0) for i, j in G.edges])
        s.add(courierTravelled <= maxTravelled)
        s.add(courierTravelled >= minTravelled)

    sumOfAllPaths = Sum([If(x[i][j][k], int(all_distances[i][j]), 0) for k in range(n_couriers) for i,j in G.edges])
    s.minimize(sumOfAllPaths + (maxTravelled - minTravelled))


    # Constraints
    # Each courier ends at the depot
    for k in range(n_couriers):
        s.add(PbEq([(x[j][0][k], 1) for j in range(1, n_items + 1)], 1))

    # Each courier depart from the depot
    for k in range(n_couriers):
        s.add(PbEq([(x[0][j][k], 1) for j in range(n_items + 1)], 1))

    # No routes from any node to itself
    for k in range(n_couriers):
        s.add([Not(x[i][i][k]) for i in range(n_items + 1)])

    # Each node (i, j) is visited only once
    # for each node there is exactly one arc entering and leaving from it
    for i in range(1, n_items + 1):  # start from 1 to exclude the depot
        s.add(PbEq([(x[i][j][k], 1) for j in range(n_items + 1) for k in range(n_couriers)],
                   1))  # each node is left exactly once by each courier

    for j in range(1, n_items + 1):  # start from 1 to exclude the depot
        s.add(PbEq([(x[i][j][k], 1) for i in range(n_items + 1) for k in range(n_couriers)],
                   1))  # each node is entered exactly once by each courier


    # start solving process
    if s.check() == sat:
        model = s.model()
        #print the graph
        tour_edges = [(i,j) for i,j in G.edges for z in range(n_couriers) if model.evaluate(x[i][j][z])]

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
        color_list = [node_colors[i] for i in G.nodes]

        nx.draw(G.edge_subgraph(tour_edges), with_labels=True, node_color=color_list)
        plt.show()
    else:
        print("\n\n ######################   UNSAT   #####################")


main(1)