import sys

import gurobipy
import numpy as np
from gurobipy import GRB, Model, quicksum
np.set_printoptions(threshold=sys.maxsize)


def input(num):
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


def main(num):
    n_couriers, n_items, max_load, size_item, dist = input(num)

    first_delivery = 0
    last_delivery = n_items+1
    x = np.zeros(dist.shape)

    model = gurobipy.Model()

    #decision variables position [i] in couriers and ordering refers to i-th item
    delivery_order = model.addMVar(shape=(n_items+2,n_couriers), lb=0, ub=n_items, vtype=GRB.INTEGER)
    #couriers = model.addMVar(n_items, ub=n_couriers, vtype=GRB.INTEGER)
    #ordering = model.addMVar(n_items, ub=n_items, vtype=GRB.INTEGER)
    a_max_dist = model.addMVar(shape=(1,n_couriers))

    """
    IDEA attuale (vedi disegno su quadernino):
    Matrice di boolean per ogni corriere, la shape sarà la stessa di all_distances, infatti la casella [i,j] 
    di questa matrice corrisponderà all'item nella casella [i,j] di all_distances.
    Sostanzialmente è un grattacielo di matrici, constraint: 
    - in cui ogni colonna 3 dimensionale dovrà contenere al massimo un 1 (un solo corriere passa per una determinata destinazione)
    - dagli indici [i,j] si recupera il peso dei due item interessati dallo spostamento e si calcola da lì il max_load massimo da non superare
    - tutti i pacchi vanno consegnati ????
    - partenza e arrivo a zero
    - ALTRE CONSTRAINT???
    
    Introduci qualcosa per il calcolo dinamico del peso trasportato dai corrieri
    """

    #constraints









    #objective


    # We can specify the solver to use as a parameter of solve
    model.optimize()


    print("\n\n\n###############################################################################")
    #print("Couriers: ",couriers.X)
    #print("Ordering: ",ordering.X)
    print("Delivery order: \n", delivery_order.X)
    print("############################################################################### \n")


#passare come parametro solo numero dell'istanza (senza lo 0)
main(2)

#istanza 2 -> 6 corrieri, 9 items