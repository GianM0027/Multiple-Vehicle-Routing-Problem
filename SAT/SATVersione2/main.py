from z3 import *
import numpy as np
from math import log2
from itertools import combinations

numero = 3


def at_least_one_np(bool_vars):
    return Or(bool_vars)


def at_most_one_np(bool_vars, name=""):
    return And([Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)])


def exactly_one_np(bool_vars, name=""):
    return And(at_least_one_np(bool_vars), at_most_one_np(bool_vars, name))


def binary_increment(a, b):
    constraints = []
    carry = {}
    num_digits = len(a)
    print(f"\nnum_digits: {num_digits}")
    print(f"b[0]: {b[0]}")
    print(f"a[0]: {a[0]}")
    print(f"b[1]: {b[1]}")
    print(f"a[1]: {a[1]}")
    constraints.append(b[0] == Not(a[0]))
    constraints.append(b[1] == Or(And(a[1], Not(a[0])), And(Not(a[1]), a[0])))
    carry[1] = a[0]

    for i in range(2, num_digits):
        carry[i] = And(a[i - 1], carry[i - 1])
        print(f"\ncarry[{i}]: {carry[i]}")
        print(f"a[i-1] / a[{i}-1]: {a[i - 1]}")
        print(f"carry[{i - 1}]: {carry[i - 1]}")
        constraints.append(b[i] == Or(And(a[i], Not(carry[i])), And(Not(a[i]), carry[i])))

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

# s = Solver()
s = Optimize()

x = [[[Bool(f"x_{i}_{j}_{k}") for k in range(n_couriers)] for j in range(n_items + 1)] for i in
     range(n_items + 1)]  # x[k][i][j] == True : route (i->j) is used by courier k | set of Archs
v = [[Bool(f"v_{i}_{k}") for k in range(n_couriers)] for i in range(n_items)]  # vehicle k is assigned to node i

len_bin = int(log2(n_items)) + 1
u = [[[Bool(f"u_{i}_{k}_{b}") for b in range(len_bin)] for k in range(n_couriers)] for i in
     range(n_items)]  # encoding in binary of order index of node i

# - - - - - - - - - - - - - - - -  CONSTRAINTS - - - - - - - - - - - - -  - - - #

# No routes from any node to itself
for k in range(n_couriers):
    s.add([Not(x[i][i][k]) for i in range(n_items + 1)])

# - - - - - - - - - - - - - - - - -
# Each node (i, j) is visited only once
# for each node there is exactly one arc entering and leaving from it
"""for i in range(1, n_items + 1):  # start from 1 to exclude the depot
    s.add(PbEq([(x[i][j][k], 1) for j in range(n_items + 1) for k in range(n_couriers)],
               1))  # each node is left exactly once by each courier

for j in range(1, n_items + 1):  # start from 1 to exclude the depot
    s.add(PbEq([(x[i][j][k], 1) for i in range(n_items + 1) for k in range(n_couriers)],
               1))  # each node is entered exactly once by each courier"""

# For each node there is exactly one arc entering and leaving from it

for i in range(1, n_items + 1):
    s.add(exactly_one_np([x[i][j][k] for j in range(n_items + 1) for k in range(n_couriers)], f"arc_leave{i}"))

# For each node there is exactly one arc entering and leaving from it
for j in range(1, n_items + 1):
    s.add(exactly_one_np([x[i][j][k] for i in range(n_items + 1) for k in range(n_couriers)], f"arc_enter{j}"))
# - - - - - - - - - - - - - - - - -

# Each courier ends at the depot
for k in range(n_couriers):
    # s.add(PbEq([(x[j][0][k], 1) for j in range(1, n_items + 1)], 1))
    s.add(exactly_one_np([x[j][0][k] for j in range(1, n_items + 1)], f"courier_ends_{k}"))

# Each courier depart from the depot
for k in range(n_couriers):
    # s.add(PbEq([(x[0][j][k], 1) for j in range(n_items + 1)], 1))
    s.add(exactly_one_np([x[0][j][k] for j in range(1, n_items + 1)], f"courier_starts_{k}"))

# For each vehicle, the total load over its route must be smaller than its max load size
for k in range(n_couriers):
    s.add(PbLe([(v[i][k], size_item[i]) for i in range(n_items)], max_load[k]))

# Each item is carried by exactly one courier
for i in range(n_items):
    # s.add(PbEq([(v[i][k], 1) for k in range(n_couriers)], 1))
    s.add(exactly_one_np([v[i][k] for k in range(n_couriers)], f"item_carried_{i}"))


# If courier k goes to location (i, j), then courier k must carry item i, j
for k in range(n_couriers):
    for i in range(n_items):
        for j in range(n_items):
            s.add([Implies(x[i][j][k], And(v[i][k], v[j][k]))])

# - - - - - - - - - - - - - - - - - NO SUBTOURS PROBLEM - - - - - - - - - - - - - - - - - - - - - - #

"""def funzione_brutta(a, b):
    return And(b[0] == Not(a[0]), b[1] == Or(And(a[1], Not(a[0])), And(Not(a[1]), a[0])), b[2] == Or(And(a[2], Not(And(a[1], a[0]))), And(Not(a[2]), And(a[1], a[0]))))

#The order of visiting locations must be consistent with the binary representations
for k in range(n_couriers):
    for j in range(n_items):
        for i in range(n_items):
            if i != j:
                print(f"u[{i}]: {u[i]} \n u[{j}]: {u[j]} \n")
                print(f"u[{i}_{k}]: {u[i][k]} \n u[{j}_{k}]: {u[j][k]} \n")
                s.add(Implies(x[i][j][k], funzione_brutta(u[i][k], u[j][k])))"""

"""for k in range(n_couriers):
    for i in range(1, n_items + 1):
        for j in range(1, n_items + 1):
            s.add(Implies(x[i][j][k], And(Or([x[j][f][k] for f in range(n_items + 1)]),
                                          Or([x[m][i][k] for m in range(n_items + 1)]))))"""

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

    # print("R: " + str(r) + "\n")
    # print("UT: " + str(ut) + "\n")

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
