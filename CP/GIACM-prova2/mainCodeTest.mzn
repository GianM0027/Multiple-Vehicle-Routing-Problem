include "alldifferent.mzn";
include "globals.mzn";

% CHOOSE THE INPUT FOR THE MODEL
%------------------------------

% 3 couriers, 7 items / Solution -> a_max_dist = [12, 10, 10];
%include "prof_example_input.dzn"; 

% 5 couriers, 9 items / Solution -> a_max_dist = [10, 10, 9, 8, 10]; (a_max_dist = [10, 9, 8, 8, 8] with the new objective f)
%include "simple_input.dzn"; 

% 5 couriers, 11 items / Solution -> a_max_dist = [13, 15, 15, 14, 13];
% include "middle_complexity_input.dzn";

% 5 couriers, 22 items / Solution -> a_max_dist = ????????????
include "middle_complex_complexity_input.dzn";

% 4 couriers, 47 items / Solution -> a_max_dist = ????????????
%include "complex_input.dzn";


% PARAMETERS DECLARATION
%------------------------------

% number of couriers
int: n_couriers; 

% number of items
int: n_items; 

% Array of couriers, each i in couriers has value max_load for courier i
array[1..n_couriers] of int: max_load;

% Array of items, each i in items has value weight for item i
array[1..n_items] of int: size_item;

% Array of all distances between delivery points
array[1..n_items+1,1..n_items+1] of int: all_distances;


set of int: COURIERS = 1..n_couriers;
set of int: ITEMS = 1..n_items;
int: steps = n_items+2;
set of int: STEPS = 1..steps; 
set of int: STEPS_NO_LAST = 1..(n_items+1);
set of int: STEPS_NO_FIRST_NO_LAST = 2..(n_items+1);


% DECISION VARIABLES
%------------------------------
% each column represent an item, the first row contains the number of the courier that delivers that item, the second row contain the order in which it is delivered
array[ITEMS] of var COURIERS: couriers;
array[ITEMS] of var int: ordering;

% array of max distance for each courier
array[COURIERS] of var int: a_max_dist;

%specie di copia dell'originale delivery_order, ogni colonna è un corriere, ogni riga contienre l'index del pacco consegnato da quel corriere in ordine cronologico
array[STEPS, COURIERS] of var 0..n_items: delivery_order;


% CONSTRAINTS per risoluzione problema
%------------------------------

%pacchi diversi che vengono consegnati dallo stesso corriere non possono avere lo stesso numero di consegna (in fatto di ordine cronologico)
constraint forall(item1, item2 in ITEMS where item1 != item2 /\ couriers[item1] == couriers[item2])(
                    ordering[item1] != ordering[item2]
);

% Ogni corriere non deve superare il suo carico massimo
constraint forall(c in COURIERS)(
          sum(i in ITEMS where couriers[i] == c)(size_item[i]) <= max_load[c]
);
                  
%Se tutti i corrieri hanno capienza sufficiente a trasportare almeno un pacco, allora devono farlo
constraint if (max(size_item) <= min(max_load)) then
    forall(c in COURIERS) (
        exists(i in ITEMS) (
            couriers[i] == c
        )
    )
endif;

         
% CONSTRAINTS per far scendere i tempi di risoluzione del problema
%------------------------------
constraint forall(c in COURIERS, s in STEPS_NO_LAST)
                  (all_distances[delivery_order[s,c] + 1,delivery_order[s + 1,c] + 1] < max(all_distances));


% OBJECTIVE FUNCTION - MINIMIZE THE DISTANCE (max_dist)
%------------------------------

constraint forall(i in ITEMS)
                (delivery_order[ordering[i] + 1, couriers[i]] = i);
                                                                                                                                                                                                                                                      %CONSTRAINTS PER delivery_order                                                                                                                                                                                                                                                       
constraint forall(c in COURIERS)
                  (delivery_order[1,c] == 0 /\ delivery_order[n_items+2,c] == 0);

%Ogni pacco deve essere consegnato una sola volta
constraint alldifferent_except_0(delivery_order);

%Ogni pacco deve essere consegnato
constraint nvalue(n_items+1, delivery_order);

% Ogni corriere non deve superare il suo carico massimo
constraint forall(c in COURIERS) (
  sum([size_item[delivery_order[i,c]] | i in STEPS where delivery_order[i,c] != 0]) <= max_load[c]);
                  
% Rimozione soluzioni con 'reload' dai corrieri
constraint forall(c in COURIERS, i in STEPS_NO_FIRST_NO_LAST)        
                 (delivery_order[i,c] != 0 \/ sum(delivery_order[i..,c]) == 0);                                                                                                         
                 
                                                                                                                                  
% CONSTRAINTS per far scendere i tempi di risoluzione del problema
%------------------------------
%Each courier deliver at least one item
constraint if max(size_item) <= min(max_load) then forall(c in COURIERS)(delivery_order[2,c] != 0 /\
              sum(delivery_order[steps-n_couriers..,c]) == 0) endif;
% constraint if max(size_item) <= min(max_load) then alldifferent(delivery_order[2,1..]) /\ all_equal(delivery_order[steps-n_couriers..,1..]) endif; %Più o meno uguale

constraint forall(c,c2 in COURIERS, s in 3..steps-n_couriers)(if delivery_order [s,c] != 0 /\ c != c2
                  then sum(delivery_order[steps-n_couriers-(s-2)..,c2]) == 0 endif);

% OBJECTIVE FUNCTION - MINIMIZE THE DISTANCE (max_dist)
%------------------------------
constraint forall(c in COURIERS)
                  (a_max_dist[c] = sum(s in STEPS_NO_LAST)
                  (all_distances[delivery_order[s,c] + 1, delivery_order[s + 1,c] + 1] ));

constraint forall(c in COURIERS) (a_max_dist[c] <= max(a_max_dist)); % with this is a little faster
var int: max_dist = max(a_max_dist);

solve minimize max_dist;      