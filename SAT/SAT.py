from z3 import *
import numpy as np
from math import log2

numero = 1

# This function imposes the constraint that 'b' is equal to 'a' + 1 in binary notation.
def binary_increment(a, b):
    constraints = []
    carry = {}
    num_digits = len(a)
    constraints.append(b[0] == Not(a[0]))
    constraints.append(b[1] == Or(And(a[1],Not(a[0])), And(Not(a[1]),a[0])))
    carry[1] = a[0]

    for i in range(2,num_digits):
        carry[i] = And(a[i-1],carry[i-1])
        constraints.append(b[i] == Or(And(a[i],Not(carry[i])),And(Not(a[i]),carry[i])))

    return And(constraints)

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


n_couriers, n_items, max_load, size_item, all_distances = inputFile(numero)

#s = Solver()
s = Optimize()

x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in
     range(n_items + 1)]  # x[k][i][j] == True : route (i->j) is used by courier k | set of Archs
v = [[Bool(f"v_{i}_{k}") for k in range(n_couriers)] for i in range(n_items)]  # vehicle k is assigned to node i

len_bin = int(log2(n_items)) + 1
u = [[[Bool(f"u_{i}_{b}_{k}") for k in range(n_couriers)] for b in range(len_bin)] for i in range(n_items)]  # encoding in binary of order index of node i

# - - - - - - - - - - - - - - - -  CONSTRAINTS - - - - - - - - - - - - -  - - - #

# No routes from any node to itself
for k in range(n_couriers):
    s.add([x[i][i][k] == False for i in range(n_items + 1)])

# Each node (i, j) is visited only once
# for each node there is exactly one arc entering and leaving from it

for i in range(1, n_items + 1):  # start from 1 to exclude the depot
    s.add(PbEq([(x[i][j][k], 1) for j in range(n_items + 1) for k in range(n_couriers)],
               1))  # each node is left exactly once by each courier

for j in range(1, n_items + 1):  # start from 1 to exclude the depot
    s.add(PbEq([(x[i][j][k], 1) for i in range(n_items + 1) for k in range(n_couriers)],
               1))  # each node is entered exactly once by each courier

# Each courier ends at the depot
for k in range(n_couriers):
    s.add(PbEq([(x[j][0][k], 1) for j in range(1, n_items + 1)], 1))

# Each courier depart from the depot
for k in range(n_couriers):
    s.add(PbEq([(x[0][j][k], 1) for j in range(n_items + 1)], 1))

# For each vehicle, the total load over its route must be smaller than its max load size
for k in range(n_couriers):
    s.add(PbLe([(v[i][k], size_item[i]) for i in range(n_items)], max_load[k]))

# Each item is carried by exactly one courier
for i in range(n_items):
    s.add(PbEq([(v[i][k], 1) for k in range(n_couriers)], 1))

# If courier k goes from location (i, j) to location (j, f), then courier k must also carry item f
for k in range(n_couriers):
    for i in range(n_items):
        for j in range(n_items):
            s.add([Implies(And(x[i][j][k], x[j][f][k]), v[f][k]) for f in range(n_items)])


# - - - - - - - - - - - - - - - - - NO SUBTOURS PROBLEM - - - - - - - - - - - - - - - - - - - - - - #

"""# The order of visiting locations must be consistent with the binary representations
for k in range(n_couriers):
    for j in range(n_items):
        for i in range(n_items):
            if i!=j:
                #print(f"u[i] -> {u[i]}")
                #print(f"u[i][k]) -> {u[i][k]}")
                s.add(Implies(x[i][j][k], binary_increment(u[i][k], u[j][k])))"""

# If courier k goes from location (i, j), then the next location must be (j, f)
# and if courier k goes to location (j, f), then the previous location must be (i, j)
for k in range(n_couriers):
    for i in range(n_items):
        for j in range(n_items):
            for f in range(n_items):
                s.add(Implies(And(x[i][j][k], x[j][f][k]), And(Or([x[i][j][k] for i in range(n_items)]), Or([x[j][f][k] for f in range(n_items)]))))


total_distance = Sum(
    [If(x[i][j][k], int(all_distances[i][j]), 0) for k in range(n_couriers) for i in range(n_items + 1) for j in
     range(n_items + 1)])
s.minimize(total_distance)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

if s.check() == sat:
    model = s.model()
    r = [[[model.evaluate(x[i][j][k]) for k in range(n_couriers)] for j in range(n_items + 1)] for i in
         range(n_items + 1)]
    ut = [[model.evaluate(v[i][k]) for k in range(n_couriers)] for i in range(n_items)]
    print("R: " + str(r) + "\n")
    print("UT: " + str(ut) + "\n")

    for k in range(n_couriers):
        route_string = "Route courier " + str(k) + " = "
        for i in range(n_items + 1):
            for j in range(n_items + 1):
                if model.evaluate(x[i][j][k]):
                    temp = str(x[i][j][k]).split('_')
                    route_string += "(" + str(temp[1]) + "-" + str(temp[2]) + ") "
        print(route_string + "\n")

    for k in range(n_couriers):
        actual_load = 0
        for i in range(n_items):
            if model.evaluate(v[i][k]):
                temp = str(v[i][k]).split('_')
                print(f"Item: {int(temp[1]) + 1} <- [{temp[2]}]")
                actual_load += size_item[i]
        print("Courier Max Load = " + str(max_load[k]))
        print("Courier Load = " + str(actual_load))
        print("\n")

        # total distance traveled minimized
    print(model.evaluate(total_distance))
    print("\n")


else:
    print("UNSATISFIABLE")
