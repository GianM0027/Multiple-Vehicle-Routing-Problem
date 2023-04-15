% Define the input data
int: m; % number of couriers
int: n; % number of items
array[1..m] of int: li; % maximum load size for each courier
array[1..n] of int: sj; % size of each item
array[1..n] of int: dj; % distance between the origin point and each distribution point

% Define the decision variables
array[1..m, 1..n] of var 0..1: xi; % binary variable for item assignment
array[1..m, 1..n] of var 0..1: rij; % binary variable for point visitation
var int: z; % maximum distance travelled by any courier

