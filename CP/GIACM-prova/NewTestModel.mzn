include "alldifferent.mzn";

% CHOOSE THE INPUT FOR THE MODEL
%------------------------------

% 3 couriers, 7 items / Solution -> a_max_dist = [12, 10, 10];
include "prof_example_input.dzn"; 

% 5 couriers, 9 items / Solution -> a_max_dist = [10, 10, 9, 8, 10]; (a_max_dist = [10, 9, 8, 8, 8] with the new objective f)
%include "simple_input.dzn"; 

% 5 couriers, 11 items / Solution -> a_max_dist = [13, 15, 15, 14, 13];
%include "middle_complexity_input.dzn";

% 4 couriers, 47 items / Solution -> a_max_dist = ????????????
%include "complex_input.dzn";


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
int : steps = n_items+(2*n_couriers);
set of int: STEPS = 1..steps;
set of int: ITEMS_0 = 0..n_items;
var int: sum_items = sum(ITEMS);

array[STEPS] of var int: items_w_zeros;
constraint forall(c in COURIERS)(items_w_zeros[c] = 0);
constraint forall(i in ITEMS)(items_w_zeros[i+n_couriers] = i);
constraint forall(s in n_items+n_couriers+1..steps)(items_w_zeros[s] = 0);
% DECISION VARIABLES
%------------------------------
% Order of delivery of the item given the courier
array[STEPS] of var ITEMS_0: delivery_order;

% Correspondence between the items and the relative courier
array[STEPS] of var COURIERS: courier_assignment;

%array[COURIERS] of var int: current_load; %it may not be needed

% array of max distance for each courier
array[COURIERS] of var int: a_max_dist;

%For test
array[1..n_items+2,COURIERS] of var ITEMS_0: decode_matrix;


% CONSTRAINTS per risoluzione problema
%------------------------------

%Tutti i corrieri partono all'origine
constraint forall(c in COURIERS)(delivery_order[c] == 0);

%La partenza ed il ritorno relativa all'origine avviene una volta per corriere
constraint forall(c in COURIERS)(courier_assignment[c] = c);
constraint forall(s in n_items+n_couriers+1..steps)(courier_assignment[s] = s-(n_items+n_couriers));

%Tutti i corrieri torano all'origine solo terminate le consegne
constraint forall(c in COURIERS)(delivery_order[c + n_items+n_couriers] > 
                  max([delivery_order[s] | s in 1..n_items+n_couriers 
                  where courier_assignment[s] == c]));

% Ogni corriere deve consegnare uno ed un solo item ogni time step
% Ogni combinazione corriere/timestep può esserci solo una volta
constraint forall(c in COURIERS, s in 1..steps-1, i in ITEMS_0)
                  (if delivery_order[s] == i /\ courier_assignment[s] == c then
                  forall(s2 in s+1..steps)(delivery_order[s2] != i \/ courier_assignment[s2] != c)
                  endif);

%Ogni pacco deve essere consegnato
% Vero di default in questo modello

% Ogni corriere non deve superare il suo carico massimo
constraint forall(c in COURIERS) (
  sum([size_item[i] | i in ITEMS where courier_assignment[i+n_couriers] == c]) <= max_load[c]);
                 
% Rimozione soluzioni con 'reload' dai corrieri
% Vero di default in questo modello
                 
% i valori dell'array delivery_order devono essere interi successivi dato un determinato corriere
constraint forall(s in STEPS, c in COURIERS)(if courier_assignment[s] == c then
                  delivery_order[s] < count(courier_assignment,c) endif);

% CONSTRAINTS per far scendere i tempi di risoluzione del problema
%------------------------------
%TEST CONSTRAINT -> each courier deliver at least one item
/*
constraint forall(c in COURIERS)(sum(delivery_order[n_items+2-n_couriers..,c]) == 0);
constraint forall(c in COURIERS)(delivery_order[2,c]!= 0);

constraint all_equal(delivery_order[n_items+2-n_couriers..,1..n_couriers]);


/*
var float: avg_dist = sum(all_distances)/((n_items+1)*n_items);
set of int: TEMP = 5..0;
constraint forall(c in COURIERS, s in STEPS_NO_LAST, t in TEMP)
                  (all_distances[delivery_order[s,c] + 1,delivery_order[s + 1,c] + 1] <= avg_dist+t);
*/    
%With those new constraints is faster / the second one may give unoptimal solutions but its an idea we can try to use


% OBJECTIVE FUNCTION - MINIMIZE THE DISTANCE (max_dist)
%------------------------------

constraint forall(s in STEPS)
                  (decode_matrix[delivery_order[s] + 1, courier_assignment[s]] = items_w_zeros[s]);

constraint sum(decode_matrix) == sum_items;

constraint forall(c in COURIERS)
                  (a_max_dist[c] = sum(s in 1..n_items+1)
                  (all_distances[decode_matrix[s,c] + 1,decode_matrix[s + 1,c] + 1] ));

var int: max_dist = max(a_max_dist);

solve satisfy; %:: int_search(delivery_order, dom_w_deg, indomain_min) 

%minimize max_dist;