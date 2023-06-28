from z3 import *
import numpy as np
from math import log2

numero = 1

def bin_plus_1(a,b): #constraints to impose that b == a+1 in binary notation
    constraints = []
    c = {}
    num_digits = len(a)
    constraints.append(b[0] == Not(a[0]))
    constraints.append(b[1] == Or(And(a[1],Not(a[0])), And(Not(a[1]),a[0])))
    c[1] = a[0]
    for i in range(2,num_digits):
        c[i] = And(a[i-1],c[i-1])
        constraints.append(b[i] == Or(And(a[i],Not(c[i])),And(Not(a[i]),c[i])))
    return And(constraints)

def inputFile(num):
    # Instantiate variables from file
    if num < 10:
        instances_path = "/Users/maurodore/GitHubRepos/CDMO-project/SAT/instances/inst0" + str(
            num) + ".dat"  # inserire nome del file
    else:
        instances_path = "/Users/maurodore/GitHubRepos/CDMO-project/SAT/instances/inst" + str(
            num) + ".dat"  # inserire nome del file

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

s = Optimize()

x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in
     range(n_items + 1)]  # x[k][i][j] == True : route (i->j) is used by courier k | set of Archs
v = [[Bool(f"v_{i}_{k}") for k in range(n_couriers)] for i in range(n_items)]  # vehicle k is assigned to node i

len_bin = int(log2(n_items)) + 1
u = [[[Bool(f"u_{i}_{k}_{b}") for b in range(len_bin)] for k in range(n_couriers)] for i in range(n_items)]  # encoding in binary of order index of node i

# - - - - - - - - - - - - - - - -  CONSTRAINTS - - - - - - - - - - - - -  - - - #

# implicit constraint: no routes from any node to itself
for k in range(n_couriers):
    s.add([x[i][i][k] == False for i in range(n_items + 1)])

# To ensure that each node (i, j) is visited only once, for each node there is exactly one arc entering and leaving
# from it
for i in range(1, n_items + 1):  # start from 1 to exclude the depot
    s.add(PbEq([(x[i][j][k], 1) for j in range(n_items + 1) for k in range(n_couriers)],
               1))  # each node is left exactly once by each courier

for j in range(1, n_items + 1):  # start from 1 to exclude the depot
    s.add(PbEq([(x[i][j][k], 1) for i in range(n_items + 1) for k in range(n_couriers)],
               1))  # each node is entered exactly once by each courier

# each courier return from the depot
for k in range(n_couriers):
    s.add(PbEq([(x[i][0][k], 1) for i in range(n_items + 1)], 1))

# each courier depart from the depot
for k in range(n_couriers):
    s.add(PbEq([(x[0][j][k], 1) for j in range(n_items + 1)], 1))

# for each vehicle, the total load over its route must be smaller than its max load size
for k in range(n_couriers):
    s.add(PbLe([(v[i][k], size_item[i]) for i in range(n_items)], max_load[k]))

# - - - - - - - - - - - - - - - - - NO SUBTOURS PROBLEM - - - - - - - - - - - - - - - - - - - - - - #

for j in range(n_items+1):
    for i in range(n_items):
        if i != j:
            s.add([Implies(x[i][j][k], bin_plus_1(u[i][k], u[j][k])) for k in range(n_couriers)])

"""#for each vehicle if x[k][i][j] = True, (i, j) -> (j, l)
for k in range(n_couriers):
    for j in range(n_items + 1):
        for i in range(n_items + 1):
            s.add([Implies(x[i][j][k], x[j][h][k]) for h in range(n_items+1)])"""

"""# For each courier, the number of nodes it travels to must be equal to the number of nodes it departs from
for k in range(n_couriers):
    s.add(Sum([If(x[i][j][k], 1, 0) for j in range(n_items + 1) for i in range(n_items + 1)]) ==
          Sum([If(x[j][i][k], 1, 0) for i in range(n_items + 1) for j in range(n_items + 1)]))"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


total_distance = Sum(
    [If(x[i][j][k], int(all_distances[i][j]), 0) for k in range(n_couriers) for i in range(n_items + 1) for j in
     range(n_items + 1)])

s.minimize(total_distance)

if s.check() == sat:
    model = s.model()
    r = [[[model.evaluate(x[i][j][k]) for k in range(n_couriers)] for j in range(n_items + 1)] for i in
         range(n_items + 1)]
    ut = [[model.evaluate(v[i][k]) for k in range(n_couriers)] for i in range(n_items)]

    for k in range(n_couriers):
        for i in range(n_items + 1):
            for j in range(n_items + 1):
                if model.evaluate(x[i][j][k]):
                    temp = str(x[i][j][k]).split('_')
                    print(f"({temp[1]}-{temp[2]}) <- [{temp[3]}]")
        print("\n")

    # total distance traveled minimized
    print(f"Total Distance: {model.evaluate(total_distance)}")
    print("\n")


else:
    print("UNSATISFIABLE")
