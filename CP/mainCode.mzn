include "alldifferent.mzn";
include "globals.mzn";

% CHOOSE THE INPUT FOR THE MODEL
%------------------------------

% 3 couriers, 7 items / Solution -> a_max_dist = [12, 10, 10];
% include "prof_example_input.dzn"; 

% 5 couriers, 9 items / Solution -> a_max_dist = [10, 10, 9, 8, 10]; (a_max_dist = [10, 9, 8, 8, 8] with the new objective f)
% include "simple_input.dzn"; 

% 5 couriers, 11 items / Solution -> a_max_dist = [13, 15, 15, 14, 13];
%include "middle_complexity_input.dzn";

% 5 couriers, 22 items / Solution -> a_max_dist = ????????????
%include "middle_complex_complexity_input.dzn";

% 4 couriers, 47 items / Solution -> a_max_dist = ????????????
include "complex_input.dzn";

include "sorted_test.dzn";


% PARAMETERS DECLARATION
%------------------------------

% number of couriers
int: n_couriers; 

% number of items
int: n_items; 

% Maximum load size for each courier
array[1..n_couriers] of int: max_load;

% Size of each item 
array[1..n_items] of int: size_item;

% Array of all distances
array[1..n_items+1,1..n_items+1] of int: all_distances;


set of int: COURIERS = 1..n_couriers;
set of int: ITEMS = 1..n_items;
int: steps = n_items+2;
set of int: STEPS = 1..(n_items+2); 
set of int: STEPS_NO_LAST = 1..(n_items+1);
set of int: STEPS_NO_FIRST_NO_LAST = 2..(n_items+1);


% DECISION VARIABLES
%------------------------------
% Correspondence between courier and items (chronological order)
array[STEPS, COURIERS] of var 0..n_items: delivery_order;

% array of max distance for each courier
array[COURIERS] of var int: a_max_dist;


% CONSTRAINTS per risoluzione problema





% CONSTRAINTS NECESSARIE per risoluzione problema
%------------------------------
%Tutti i corrieri partono e tornano all'origine
% constraint count(delivery_order[1,1..],0,n_couriers); %più lento di qualche ms
% constraint count(delivery_order[n_items+2,1..],0,n_couriers);
constraint forall(c in COURIERS)
                  (delivery_order[1,c] == 0 /\ delivery_order[n_items+2,c] == 0);

%Ogni pacco deve essere consegnato una sola volta
constraint alldifferent_except_0(delivery_order);

%Ogni pacco deve essere consegnato
% constraint sum(delivery_order) == sum_items; %old version
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
constraint if max(size_item) <= min(max_load) then 
              forall(c in COURIERS)(delivery_order[2,c] != 0 /\
              sum(delivery_order[steps-n_couriers..,c]) == 0) endif;
              
              
constraint forall(c,c2 in COURIERS, s in 3..steps-n_couriers)(if delivery_order [s,c] != 0 /\ c != c2
                  then sum(delivery_order[steps-n_couriers-(s-2)..,c2]) == 0 endif);


%se tutti i corrieri possono trasportare almeno un pacco, allora devono farlo (inutile ai fini dell'efficienza)
constraint if max(size_item) <= min(max_load) then forall(c in COURIERS)
                                      (delivery_order[2,c] != 0) endif;





/*
(CONSTRAINT di base sbagliate, ma da cui si può prendere spunto per qualche idea)

% vincolo con il quale si cerca di dare una logica alla scelta della meta successiva, sviluppare l'idea
constraint forall(c in COURIERS, s in STEPS_NO_LAST)
                  (all_distances[delivery_order[s,c] + 1,delivery_order[s + 1,c] + 1] <
                  max(all_distances));


%vincolo con il quale si cerca di assegnare meno peso a un corriere con max_load più piccolo 
constraint forall(c in COURIERS, c1 in COURIERS where max_load[c] <= max_load[c1])(
                  sum([size_item[delivery_order[i,c]] | i in STEPS where delivery_order[i,c] != 0])
                    <= sum([size_item[delivery_order[i,c1]] | i in STEPS where delivery_order[i,c1] != 0]));
*/





% OBJECTIVE FUNCTION - MINIMIZE THE DISTANCE (max_dist)
%------------------------------
constraint forall(c in COURIERS)
                  (a_max_dist[c] = sum(s in STEPS_NO_LAST)
                  (all_distances[delivery_order[s,c] + 1,delivery_order[s + 1,c] + 1] ));

constraint forall(c in COURIERS) (a_max_dist[c] <= max(a_max_dist)); % with this is a little faster
var int: max_dist = max(a_max_dist);



% euristica per la ricerca della soluzione

%provare questo metodo di risoluzione con le colonne ordinate in ordine crescente
solve :: int_search(delivery_order, dom_w_deg, indomain_min)
minimize max_dist;