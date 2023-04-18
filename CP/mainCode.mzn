% Input data
int: n_couriers; % number of couriers
int: n_items; % number of items

array[1..n_couriers] of int: max_load; % maximum load size for each courier
array[1..n_items] of int: size_item; % size of each item
array[1..n_items] of int: distance_to_origin; % distance between each distribution point and the origin point
array[1..n_items] of int: distance_from_origin; % distance between the origin point and each distribution point
array[1..n_items,1..n_items] of int: distance_between_destinations; % distance between each distribution point



% Decision variables
array[1..n_items, 1..n_couriers] of var 0..1: item_assignment; % binary variable -> a courier deliver a package or not
array[1..n_items, 1..n_couriers] of var 1..n_items: delivery_order; % correspondence between courier and items (by chronological order)
var int: z; % maximum distance travelled by any courier




% Constraints 




% Objective function



solve satisfy;