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

    return n_couriers, n_items, max_load, size_item, dist


def create_weight_matrix(all_distances, size_item):
    all_dist = range(all_distances.shape[0])
    weight_matrix = np.array(all_distances)
    for i in all_dist:
        for j in all_dist:
            if i != 0 and j != 0:
                if i > j:
                    weight_matrix[i, j] = size_item[i-1]
                if j < i:
                    weight_matrix[i, j] = size_item[j-1]
    weight_matrix[0, 1:] = size_item[:]
    weight_matrix[1:, 0] = size_item[:]
    return weight_matrix



def graph(all_distances):
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
    all_dist_size = all_distances.shape[0]
    weights_matrix = create_weight_matrix(all_distances, size_item)

    #model
    model = gp.Model()

    # Defining a graph which contain all the possible paths
    G = graph(all_distances)



    #decision variables
    x = [model.addVars(G.edges, vtype=gp.GRB.BINARY) for _ in range(n_couriers)]
    ordering = [model.addVars(G.nodes, vtype=GRB.INTEGER) for _ in range(n_couriers)] #ordering[z,i] ha valore p se i è la p-esima meta del corriere z



    #objective function (minimize total distance travelled)
    model.setObjective(quicksum(all_distances[i, j] * x[z][i, j] for z in range(n_couriers) for i,j in G.edges),GRB.MINIMIZE)



    #CONSTRAINTS

    # tutti i pacchi vanno consegnati ma diversi corrieri passano per diversi punti
    # (ogni riga 3dimensionale e corrispettiva colonna 3dimensionale, depot escluso, devono contenere UNA E UNA SOLA presa in carico)
    for j in G.nodes:
        if j != 0:
            model.addConstr(quicksum(x[z][i, j]+x[z][j, i] for z in range(n_couriers) for i in G.nodes if i != j ) == 1)

    #ogni corriere entra ed esce una volta dal depot (parte e torna allo starting point)
    for z in range(n_couriers):
        model.addConstr(quicksum(x[z][i, 0] for i in G.predecessors(0)) == 1)
        model.addConstr(quicksum(x[z][0, j] for j in G.successors(0)) == 1)

    # per ogni corriere non si supera il max_load

    #eliminate sub-tour
    #model.addConstrs(ordering[z][0] == 1 for z in range(n_couriers))  # first city in each tour is depot


    """
    # constraint per il numero di edges che possono entrare e uscire da ogni nodo
    for z in range(n_couriers):
        model.addConstrs(
            gp.quicksum(x[z][i, j] for i in G.predecessors(j)) == 1 for j in G.nodes)  # Enter each city exactly once
        model.addConstrs(
            gp.quicksum(x[z][i, j] for j in G.successors(i)) == 1 for i in G.nodes)  # Leave each city exactly  once
    """

    #eliminate subtour
    #model.addConstrs(u[z][0] == 1 for z in range(n_couriers))  # first city in each tour is depot
    #model.addConstrs(u[z][i] >= 2 for z in range(n_couriers) for i in G.nodes)  # other cities > 1

    #for z in range(n_couriers):
        #model.addConstrs(quicksum(u[z][i] for i in G.nodes) == (quicksum(x[z][i,j] for i,j in G.edges)*(quicksum(x[z][i,j] for i,j in G.edges)+1))/2)
        #model.addConstrs(u[z][i] - u[z][j] + (n - 1) * x[z][i, j] + (n - 3) * x[z][j, i] <= n - 2 for i, j in G.edges if j != 0)



    """
    #constraints

    for i in range(all_dist_size):
        # tutti i pacchi vanno consegnati (ogni colonna 3dimensionale deve contenere almeno una presa in carico)
        model.addConstr((delivery_order[:, :, i].sum() >= 1))
        for j in range(all_dist_size):
            if i != j:
                # un solo corriere passa per una determinata posizione [i,j] (un pacco viene consegnato da un solo corriere)
                model.addConstr((delivery_order[:, i, j].sum() <= 1))
            else:
                # la diagonale deve essere sempre composta da zeri
                model.addConstr((delivery_order[:, i, j].sum() == 0))

    # per ogni corriere non si supera il max_load
    for z in range(n_couriers):
        model.addConstr(sum(weights_matrix[i,j]*delivery_order[z, i, j] for i in range(all_dist_size) for j in range(all_dist_size)) <= max_load[z])


    # Evitare subtour: non passi dalla posizione [1,2] alla posizione [8,10], le consegne sono valide solo se [i,j] -> [j,k]
    
    
    
    """

    model.optimize()


    #print information about solving process
    print("\n\n\n###############################################################################")
    print("Number of items: ", n_items)
    print("Number of couriers: ", n_couriers)
    print("all_distances:\n", all_distances, "\n")
    print("Size_items: ", size_item)
    print("weight_matrix:\n", weights_matrix)


    for z in range(n_couriers):
        print(f"\nCourier {z}: ")
        current_load = quicksum(weights_matrix[i,j]*x[z][i, j] for i,j in G.edges)
        print("Max load: ", max_load[z])
        print("Final load: ", current_load.getValue())
        total_dist = quicksum(all_distances[i, j] * x[z][i, j].x for z in range(n_couriers) for i,j in G.edges)
        print("Total distance: ", total_dist)

    z = 0
    for dictionary in ordering:
        print(f"\nCourier: {z}")
        for key in dictionary.keys():
            print(f"{key}:", dictionary[key].X)
        print("")
        z+=1

    for z in range(n_couriers):
        tour_edges = [edge for edge in G.edges if x[z][edge].x >= 1]
        nx.draw(G.edge_subgraph(tour_edges), with_labels=True)
        plt.figure(z)
    plt.show()

    print("############################################################################### \n")




#passare come parametro solo numero dell'istanza (senza lo 0)
main(1)
