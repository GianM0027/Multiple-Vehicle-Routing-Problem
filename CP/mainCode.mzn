include "alldifferent.mzn";


% CHOOSE THE INPUT FOR THE MODEL
%------------------------------
%include "prof_example_input.dzn"; % 3 couriers, 7 items / Solution -> a_max_dist = [12, 10, 10];
%include "complex_input.dzn"; % 5 couriers, 9 items / Solution -> a_max_dist = ????????????
include "simple_input.dzn"; % 4 couriers, 47 items / Solution -> a_max_dist = [10, 10, 9, 8, 10];



% PARAMETERS DECLARATION
%------------------------------

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
set of int: STEPS_NO_LAST = 1..(n_items+1);
set of int: STEPS_NO_FIRST_NO_LAST = 2..(n_items+1);
var float: sum_items = sum(ITEMS);



% DECISION VARIABLES
%------------------------------

% Correspondence between courier and items (chronological order)
array[STEPS, COURIERS] of var 0..n_items: delivery_order;

% array of max distance for each courier
array[COURIERS] of var int: a_max_dist; 




% CONSTRAINTS per risoluzione problema
%------------------------------

%Tutti i corrieri partono e tornano all'origine
constraint forall(i in COURIERS)
                  (delivery_order[1,i] == 0 /\ delivery_order[n_items+2,i] == 0);

%Ogni pacco deve essere consegnato una sola volta
constraint alldifferent_except_0(delivery_order);

%Ogni pacco deve essere consegnato
constraint sum(delivery_order) == sum_items;

% Ogni corriere non deve superare il suo carico massimo
constraint forall(c in COURIERS) (
  sum([size_item[delivery_order[i,c]] | i in STEPS where delivery_order[i,c] != 0]) <= max_load[c]);
                  
% Rimozione soluzioni con 'reload' dai corrieri
constraint forall(c in COURIERS, i in STEPS_NO_FIRST_NO_LAST)        
                 (delivery_order[i,c] != 0 \/ sum(delivery_order[i..,c]) == 0);
                 
         
% CONSTRAINTS per far scendere i tempi di risoluzione del problema
%------------------------------



                  


% OBJECTIVE FUNCTION - MINIMIZE THE DISTANCE (max_dist)
%------------------------------
constraint forall(c in COURIERS)
                  (a_max_dist[c] = sum(s in STEPS_NO_LAST)
                  (all_distances[delivery_order[s,c] + 1,delivery_order[s + 1,c] + 1]));
var int: max_dist = max(a_max_dist);


solve minimize max_dist;
