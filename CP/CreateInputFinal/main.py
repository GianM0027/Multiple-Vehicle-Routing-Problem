from minizinc import Instance, Model, Solver
import numpy as np
import datetime


def main(argv):
    # Create a MiniZinc model
    gecode = Solver.lookup("gecode", executable="C:\\Users\gianm\AppData\Local\Programs\Python\Python311\Lib\site-packages")

    model = Model()
    model.add_file("mainCode.mzn")

    # Transform Model into a instance
    inst = Instance(gecode, model)

    # Instantiate variables from file
    instances_path = "Instances_CP_blank/" + str(argv) + ".dzn"
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

    inst["n_couriers"] = n_couriers
    inst["n_items"] = n_items
    inst["max_load"] = max_load
    inst["size_item"] = size_item
    inst["all_distances"] = dist

    print("n_couriers = " + str(inst["n_couriers"]) + ";")
    print("n_items = " + str(inst["n_items"]) + ";")
    print("max_load = " + str(inst["max_load"]) + ";")
    print("size_item = " + str(inst["size_item"]) + ";")
    dist_str = str(dist)
    dist_str = dist_str[0] + dist_str[1:-1].replace("[", "|").replace("]", "|") + dist_str[-1]
    dist_str = dist_str.replace("\n", "")
    dist_str = dist_str[0] + dist_str[1:-1].replace("| |", "|") + dist_str[-1]
    dist_str = dist_str[0] + dist_str[1:-1].replace(" ", ", ") + dist_str[-1]

    print("all_distances = " + str(dist_str) + ";")

    # Output
    result = inst.solve(timeout=datetime.timedelta(minutes=5))
    print("\n\n" + "--> " + str(result.solution))


#instance_number = input("Inserire numero Instance [1 -5]")
main("01")