import numpy as np

def sort_rows_by_sum(all_distances):
    size = len(all_distances[:, 1])

    for row in range(1,size):
        min_idx = row

        for i in range(row + 1, size):
            if sum(all_distances[i, :]) < sum(all_distances[min_idx, :]):
                min_idx = i

        # put min at the correct position
        all_distances[[row, min_idx]] = all_distances[[min_idx, row]]


    #checking position of zeros
    for column in range(size):
        zero_pos = np.where(all_distances[:, column] == 0)

        while zero_pos[0] != column:
            all_distances[:, column] = np.roll(all_distances[:, column], 1)
            zero_pos = np.where(all_distances[:, column] == 0)


n_couriers = int(input("n_couriers: "))
n_items = int(input("n_items: "))

size_item = np.random.randint(4, 12, size=n_items)
max_load = np.random.randint(4*n_couriers, 12*n_couriers, size=n_couriers)

while np.sum(max_load) < np.sum(size_item):
    max_load = np.random.randint(4*n_couriers, 12*n_couriers, size=n_couriers)

#matrix
matrix = np.random.randint(3, 16, size=(n_items+1, n_items+1))
# Fill the diagonal with zeros
np.fill_diagonal(matrix, 0)
all_distances = matrix + matrix.T
sort_rows_by_sum(all_distances)


#print
print(n_couriers)
print(n_items)

# Print the array without brackets
for i in range(n_couriers):
    print(max_load[i], end=" ")
print("")

for i in range(n_items):
    print(size_item[i], end=" ")
print("")

for j in range(n_items+1):
    for i in range(n_items+1):
        print(all_distances[i, j], end=" ")
    print("")