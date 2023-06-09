import sys
import numpy as np
import gurobipy as gp
from gurobipy import *
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

    model = gp.Model()

    #decision variables position [i] in couriers and ordering refers to i-th item
    couriers = model.addMVar(n_items, ub=n_couriers, vtype=GRB.INTEGER)
    ordering = model.addMVar(n_items, ub=n_items, vtype=GRB.INTEGER)
    a_max_dist = model.addMVar(shape=(1,n_couriers))

    # RIMODELLA IL PROBLEMA IN MODO CHE PER OGNI CORRIERE K ESISTA UNA MATRICE UGUALE A DIST, MA AL POSTO DELLE DISTANZE CI SONO
    # ZERO E UNO A SECONDA CHE QUEL PACCO SIA STATO PRESO IN CARICO DAL CORRIERE K. CAPISCI SE FATTIBILE
    # https://www.youtube.com/watch?v=zWPNHCWEOTE





    #constraints






    #objective


    # We can specify the solver to use as a parameter of solve
    model.optimize()


    print("\n\n\n###############################################################################")
    print("Couriers: ",couriers.X)
    print("Ordering: ",ordering.X)
    print("############################################################################### \n")


#passare come parametro solo numero dell'istanza (senza lo 0)
main(2)

#istanza 2 -> 6 corrieri, 9 items