import sys
import gurobipy
import numpy as np
from gurobipy import GRB, Model, quicksum
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


def printPathAndDistance(delivery_order, all_dist):
    loop = range(delivery_order.shape[0])
    dist = 0
    last_point = 0
    print("(Start,0) -> ", end="")
    for i in loop:
        for j in loop:
            if delivery_order[i,j] == 1:
                print(f"({i},{j}) -> ", end="")
                last_point = j
                dist += all_dist[i,j]
    print(f"({last_point}, End)")
    print("Distance: ", dist)

def main(num):
    n_couriers, n_items, max_load, size_item, all_distances = inputFile(num)
    all_dist_size = all_distances.shape[0]
    weights_matrix = create_weight_matrix(all_distances, size_item)

    model = gurobipy.Model()

    delivery_order = model.addMVar(shape=(n_couriers, all_dist_size, all_dist_size), vtype=GRB.BINARY)
    ordering = model.addMVar(shape=(n_couriers,n_items), ub=n_items, vtype=GRB.INTEGER) #uFrom[k,i] ha valore p se i è la p-esima meta del corriere k

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


    #ogni corriere parte e torna all'origine
    for z in range(n_couriers):
        model.addConstr(delivery_order[z, :, 0].sum()== 1)
        model.addConstr(delivery_order[z, 0, :].sum() == 1)


    # per ogni corriere non si supera il max_load
    for z in range(n_couriers):
        model.addConstr(sum(weights_matrix[i,j]*delivery_order[z, i, j] for i in range(all_dist_size) for j in range(all_dist_size)) <= max_load[z])


    # Evitare subtour: non passi dalla posizione [1,2] alla posizione [8,10], le consegne sono valide solo se [i,j] -> [j,k]
    for k in range(n_couriers):
        model.addConstr(ordering[k,0] == 1)

    for k in range(n_couriers):
        for i in range(1,n_items):
            model.addConstr(ordering[k,i] >= 2)

    for k in range(n_couriers):
        for i in range(1,n_items):
            for j in range(1, n_items):
                model.addConstr(ordering[k,j] + (n_items-2) <= ordering[k,i] + (n_items-1)*delivery_order[k,i,j])


    #objective
    #model.setObjective(sum(all_distances[i,j]*delivery_order[z, i, j] for z in range(n_couriers) for i in range(all_dist_size) for j in range(all_dist_size)), GRB.MINIMIZE)

    # We can specify the solver to use as a parameter of solve
    model.optimize()



    print("\n\n\n###############################################################################")
    print("Number of items: ", n_items)
    print("Number of couriers: ", n_couriers)
    print("all_distances:\n", all_distances, "\n")
    print("Size_items: ", size_item)
    print("weight_matrix:\n",weights_matrix)
    for z in range(n_couriers):
        print("\nDelivery order ", z, "->\n", delivery_order[z,:,:].X)
        current_load = sum(weights_matrix[i,j]*delivery_order[z, i, j] for i in range(all_dist_size) for j in range(all_dist_size))
        print("Max load: ", max_load[z])
        print("Final load: ", int(current_load.getValue()))
        printPathAndDistance(delivery_order[z,:,:].X, all_distances)
    print("\n",ordering.X)
    print("############################################################################### \n")


#passare come parametro solo numero dell'istanza (senza lo 0)
main(1)
