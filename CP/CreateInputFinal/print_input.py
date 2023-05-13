import numpy as np
import datetime


def main():

    # Instantiate variables from file
    instances_path = "Instances_CP/sorted_test.dzn"   #inserire nome del file
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
    dist_str = dist_str[0] + dist_str[1:-1].replace("| |", "|") + dist_str[-1]
    dist_str = dist_str[0] + dist_str[1:-1].replace(" ", ", ") + dist_str[-1]

    print("n_couriers = " + str(n_couriers) + ";")
    print("n_items = " + str(n_items) + ";")
    print("max_load = " + str(max_load) + ";")
    print("size_item = " + str(size_item) + ";")
    print("all_distances = " + str(dist_str) + ";")


# instance_number = input("Inserire numero Instance [1 -5]")
# main(instance_number)
main()
