include "alldifferent.mzn";
include "globals.mzn";
/*
% PARAMETERS DECLARATION
%------------------------------
% number of couriers
int: n_couriers = 3; 

% number of items
int: n_items = 7; 

% Maximum load size for each courier
array[1..n_couriers] of int: max_load = [14, 10, 8];

% Size of each item (AGGIUSTA INDICI ARRAY)
array[1..n_items] of int: size_item = [3, 2, 6, 8, 5, 4, 4];

% Distance between each distribution point
array[1..n_items] of int: distance_to_origin = [2, 3, 3, 4, 3, 4, 4];

% Distance between the origin point and each distribution point
array[1..n_items] of int: distance_from_origin = [2, 3, 3, 4, 3, 4, 4];

% Array of all distances
array[1..n_items+1,1..n_items+1] of int: all_distances = [|0, 2, 3, 3, 4, 3, 4, 4| 
                                                           2, 0, 3, 3, 6, 5, 6, 6 |  
                                                           3, 3, 0, 4, 3, 4, 7, 7 | 
                                                           3, 3, 4, 0, 7, 6, 3, 5 | 
                                                           4, 6, 3, 7, 0, 3, 6, 6 | 
                                                           3, 5, 4, 6, 3, 0, 3, 3 | 
                                                           4, 6, 7, 3, 6, 3, 0, 2 | 
                                                           4, 6, 7, 5, 6, 3, 2, 0 |];

set of int: COURIERS = 1..n_couriers;
set of int: ITEMS = 1..n_items;
set of int: STEPS = 1..(n_items+2);
var float: sum_items = n_items*(n_items+1)/2;
*/


% PARAMETERS DECLARATION
%------------------------------

include "input.dzn";

% number of couriers
int: n_couriers; 

% number of items
int: n_items; 

% Maximum load size for each courier
array[1..n_couriers] of int: max_load;

% Size of each item (AGGIUSTA INDICI ARRAY)
array[1..n_items] of int: size_item;

% Array of all distances
array[1..n_items+1,1..n_items+1] of int: all_distances;

set of int: COURIERS = 1..n_couriers;
set of int: ITEMS = 1..n_items;
set of int: STEPS = 1..(n_items+2);
var float: sum_items = n_items*(n_items+1)/2;



% DECISION VARIABLES
%------------------------------

% Correspondence between courier and items (chronological order)
array[STEPS, 1..n_couriers] of var 0..n_items: delivery_order;



% CONSTRAINTS
%------------------------------

%Tutti i corrieri partono e tornano all'origine
constraint forall(i in COURIERS)
                  (delivery_order[1,i] == 0 /\ delivery_order[n_items,i] == 0
);

%Ogni pacco deve essere consegnato una sola volta
constraint alldifferent_except_0(delivery_order)::domain;

%Ogni pacco deve essere consegnato
constraint sum(delivery_order) == sum_items;

% Ogni corriere non deve superare il suo carico massimo
constraint forall(c in COURIERS) (
  sum([size_item[delivery_order[i,c]] | i in STEPS where delivery_order[i,c] != 0]) <= max_load[c]
);
                  
% Rimozione soluzioni con 'reload' dai corrieri
constraint forall(c in COURIERS, i in 2..(n_items+1))        
                 (delivery_order[i,c] != 0 \/ sum(delivery_order[i..,c]) == 0);
                 



% OBJECTIVE FUNCTION - MINIMIZE THE DISTANCE (max_dist)
%------------------------------
set of int: STEPS_NO_LAST = 1..(n_items+1);
array[COURIERS] of var int: a_max_dist; 
constraint forall(c in COURIERS)
                  (a_max_dist[c] = sum(s in STEPS_NO_LAST)
                  (all_distances[delivery_order[s,c] + 1,delivery_order[s + 1,c] + 1]));
var int: max_dist = max(a_max_dist);


solve minimize max_dist;