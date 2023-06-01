import sys
import numpy as np
from pulp import *
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
    STEPS = range(0, n_items + 2)
    COURIERS = range(0, n_couriers)

    prob = LpProblem("MIP_solver", LpMinimize)

    delivery_order = LpVariable.dicts("delivery_order", (COURIERS, STEPS), 0, n_items, LpInteger) #dizionario con corrieri come keys e possibili items come values
    a_max_dist = LpVariable.dicts("a_max_dist", COURIERS, cat='Integer')

    print(delivery_order.keys())

    # Constraint: All couriers start and end at the origin
    for key in delivery_order:
        prob += delivery_order[key][0] == 0
        prob += delivery_order[key][n_items+1] == 0

    # Constraint: Each item is delivered only once
    #?????????

    """"
    # Constraint: Couriers do not exceed their maximum load
    for c in COURIERS:
        load_sum = 0
        for i in STEPS:
            delivery_value = value(delivery_order[i][c])
            if delivery_value != 0 and delivery_value != None:
                load_sum += size_item[delivery_value]  # Access the size of the item
        prob += load_sum <= max_load[c]

    # Constraint: Avoid reloads within couriers
    for c in COURIERS:
        for i in STEPS_NO_FIRST_NO_LAST:
            prob += delivery_order[i][c] != 0 or lpSum(delivery_order[s][c] for s in range(i, n_items + 3)) == 0

    #objective
    #prob += lpSum(a_max_dist[c] for c in COURIERS)
    """

    # We can specify the solver to use as a parameter of solve
    prob.solve()




#passare come parametro solo numero dell'istanza (senza lo 0)
main(2)

#istanza 2 -> 6 corrieri, 9 items