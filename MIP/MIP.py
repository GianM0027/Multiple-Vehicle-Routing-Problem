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
    STEPS = range(n_items + 2)
    COURIERS = range(n_couriers)

    first_delivery = 0
    last_delivery = n_items+1

    model = gp.Model()

    #decision variables
    delivery_order = model.addMVar(shape=(n_items + 2,n_couriers), ub=n_items, vtype=GRB.INTEGER)
    a_max_dist = model.addMVar(shape=(1,n_couriers))


    # Constraint: All couriers start and end at the origin
    for i in COURIERS:
        model.addConstr(delivery_order[first_delivery][i] == 0)
        model.addConstr(delivery_order[last_delivery][i] == 0)

    # Constraint: Each item is delivered (all values different but zero)

    # Constraint: Each item is delivered only once
    model.addConstr(sum(delivery_order[i,j] for i in STEPS for j in COURIERS) == (n_items*(n_items+1)/2))

    # Constraint: Couriers do not exceed their maximum load

    # Constraint: Avoid reloads within couriers

    #objective



    # We can specify the solver to use as a parameter of solve
    model.optimize()


    print("\n\n\n###############################################################################")
    print(delivery_order.X)
    print("############################################################################### \n")


#passare come parametro solo numero dell'istanza (senza lo 0)
main(2)

#istanza 2 -> 6 corrieri, 9 items