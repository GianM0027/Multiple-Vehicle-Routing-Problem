include "alldifferent.mzn";
include "globals.mzn";

% Input data
int: n_couriers; % number of couriers
int: n_items; % number of items

array[1..n_couriers] of int: max_load; % maximum load size for each courier
array[1..n_items] of int: size_item; % size of each item
array[1..n_items] of int: distance_to_origin; % distance between each distribution point and the origin point
array[1..n_items] of int: distance_from_origin; % distance between the origin point and each distribution point
array[1..n_items,1..n_items] of int: distance_between_destinations; % distance between each distribution point



% Decision variables
%array[1..n_items, 1..n_couriers] of var 0..1: item_assignment; % binary variable -> a courier deliver a package or not
array[1..n_items, 1..n_couriers] of var 0..n_items: delivery_order; % correspondence between courier and items (by chronological order)
var int: max_distance; % maximum distance travelled by any courier



% Constraints 
/*
- Tutti i corrieri partono e tornano all'origine
- Ogni pacco deve essere consegnato una sola volta
- Ogni pacco deve essere consegnato
- Ogni corriere non deve superare il suo carico massimo
- 
*/

%Tutti i corrieri partono e tornano all'origine
set of int: COURIERS = 1..n_couriers;
constraint forall(i in COURIERS)
                  (delivery_order[1,i] == 0 /\ delivery_order[n_items,i] == 0);

%Ogni pacco deve essere consegnato una sola volta
constraint alldifferent_except_0(delivery_order);

%Ogni pacco deve essere consegnato
set of int: ITEMS = 1..n_items;
int: size = n_couriers*n_items;
array[1..size] of int: DELIVERY_ORDER = delivery_order;
constraint at_least(1, DELIVERY_ORDER, ITEMS);

/*
constraint forall(i in ITEMS)
                  (i in delivery_order);
*/

% Objective function
% minimizzare la distanza


%solve -> minimizzare la objective
solve satisfy;