import sys
import numpy as np
from pulp import *
np.set_printoptions(threshold=sys.maxsize)



def writeInputFile(num):
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

    dist_str = str(dist)
    dist_str = dist_str[0] + dist_str[1:-1].replace("[", "|").replace("]", "|") + dist_str[-1]
    dist_str = dist_str.replace("\n", "")
    dist_str = dist_str[0] + dist_str[1:-1].replace("  ", " ") + dist_str[-1]  # eliminazione doppi spazi
    dist_str = dist_str[0] + dist_str[1:-1].replace("   ", " ") + dist_str[-1]  # eliminazione tripli spazi
    dist_str = dist_str[0] + dist_str[1:-1].replace("| ", "|") + dist_str[-1]
    dist_str = dist_str[0] + dist_str[1:-1].replace("| |", "|") + dist_str[-1]
    dist_str = dist_str[0] + dist_str[1:-1].replace("||", "|") + dist_str[-1]
    dist_str = dist_str[0] + dist_str[1:-1].replace(" ", ", ") + dist_str[-1]
    dist_str = dist_str[0] + dist_str[1:-1].replace(", ,", ", ") + dist_str[-1]

    print("n_couriers = " + str(n_couriers) + ";")
    print("n_items = " + str(n_items) + ";")
    print("max_load = " + str(max_load) + ";")
    print("size_item = " + str(size_item) + ";")
    print("all_distances = " + str(dist_str) + ";")

    with open("input.dzn", "w") as output_file:
        output_file.write("n_couriers = " + str(n_couriers) + ";\n")
        output_file.write("n_items = " + str(n_items) + ";\n")
        output_file.write("max_load = " + str(max_load) + ";\n")
        output_file.write("size_item = " + str(size_item) + ";\n")
        output_file.write("all_distances = " + str(dist_str) + ";")

def main(num):
    writeInputFile(num) #funzione che crea il file da dare in input al modello

    prob = LpProblem("Brewery Problem", LpMaximize)

    A = LpVariable("Ale", 0, None, LpInteger)
    B = LpVariable("Beer", 0, None, LpInteger)

    prob += 13 * A + 23 * B, "Profit"
    prob += 5 * A + 15 * B <= 480, "Corn"
    prob += 4 * A + 4 * B <= 160, "Hop"
    prob += 35 * A + 20 * B <= 1190, "Malt"

    # We can specify the solver to use as a parameter of solve
    prob.solve()



#passare come parametro solo numero dell'istanza (senza lo 0)
main(2)
