include "alldifferent.mzn";
include "globals.mzn";

% Input data
int: n_couriers; % number of couriers
int: n_items; % number of items

array[1..n_couriers] of int: max_load; % maximum load size for each courier
array[1..n_items] of int: size_item; % size of each item (AGGIUSTA INDICI ARRAY)
array[1..n_items] of int: distance_to_origin; % distance between each distribution point and the origin point
array[1..n_items] of int: distance_from_origin; % distance between the origin point and each distribution point
array[1..n_items,1..n_items] of int: distance_between_destinations; % distance between each distribution point
array[1..n_items+1,1..n_items+1] of int: all_distances; %array of all distances



% set of and auxiliary stuff
set of int: COURIERS = 1..n_couriers;
set of int: ITEMS = 1..n_items;
set of int: STEPS = 1..(2*n_items+1);
var float: sum_items = n_items*(n_items+1)/2;


% Decision variables
%array[1..n_items, 1..n_couriers] of var 0..1: item_assignment; % binary variable -> a courier deliver a package or not
array[1..(2*n_items+1), 1..n_couriers] of var 0..n_items: delivery_order; %correspondence between courier and items (by chronological order), 2*n_items+1 perchè se un pacco è pesante il corriere deve fare avanti e indietro passando di nuovo per l'origine
var int: max_distance = 0; % maximum distance travelled by any courier


%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT TEST %%%%%%%%%%%%%%%%%%%%%%%%%%
n_couriers = 3;
n_items = 7;
size_item = [3, 2, 6, 8, 5, 4, 4];
max_load = [15,10,7];

distance_between_destinations = [|0, 3, 3, 6, 5, 6, 6 | 
                                  3, 0, 4, 3, 4, 7, 7 | 
                                  3, 4, 0, 7, 6, 3, 5 | 
                                  6, 3, 7, 0, 3, 6, 6 | 
                                  5, 4, 6, 3, 0, 3, 3 | 
                                  6, 7, 3, 6, 3, 0, 2 | 
                                  6, 7, 5, 6, 3, 2, 0 |];

distance_to_origin = [2, 3, 3, 4, 3, 4, 4];

distance_from_origin  = [2, 3, 3, 4, 3, 4, 4];

all_distances = [|0, 2, 3, 3, 4, 3, 4, 4|
                  2, 0, 3, 3, 6, 5, 6, 6 | 
                  3, 3, 0, 4, 3, 4, 7, 7 | 
                  3, 3, 4, 0, 7, 6, 3, 5 | 
                  4, 6, 3, 7, 0, 3, 6, 6 | 
                  3, 5, 4, 6, 3, 0, 3, 3 | 
                  4, 6, 7, 3, 6, 3, 0, 2 | 
                  4, 6, 7, 5, 6, 3, 2, 0 |];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Constraints 
/*
- Tutti i corrieri partono e tornano all'origine
- Ogni pacco deve essere consegnato una sola volta
- Ogni pacco deve essere consegnato
- Ogni corriere non deve superare il suo carico massimo
*/

%Tutti i corrieri partono e tornano all'origine
constraint forall(i in COURIERS)
                  (delivery_order[1,i] == 0 /\ delivery_order[n_items,i] == 0);

%Ogni pacco deve essere consegnato una sola volta
constraint alldifferent_except_0(delivery_order)::domain;

%Ogni pacco deve essere consegnato
constraint sum(delivery_order) == sum_items;

% Ogni corriere non deve superare il suo carico massimo
constraint forall(c in COURIERS) (
  sum([size_item[delivery_order[i,c]] | i in STEPS where delivery_order[i,c] != 0]) <= max_load[c]
);

%constraint forall(c in COURIERS, i in STEPS)(size_item[delivery_order[i,c]] <= max_load[c]);
%array[COURIERS] of int: load_couriers = [sum(size_item[delivery_order[i,c]]) | i in STEPS, c in COURIERS];
%array[COURIERS] of int: load_couriers = [sum(c in COURIERS)(size_item[delivery_order[i,c]])];
/*
for(c in COURIERS, i in STEPS)(
                      if delivery_order[i,c] != 0 then
                        load_couriers[c] = load_couriers[c] + size_item[delivery_order[i,c]]
                      endif
                )
          );*/

% TEST CONSTRAINT FOR SYMMETRY BREAKING AND REDUCING THE SEARCH SPACE:
% Rimosse tutte le soluzioni che partono con zero alla seconda tappa
constraint forall(c in COURIERS, s in STEPS)
                 (if delivery_order[2,c] == 0 then delivery_order[s,c] = 0 endif);
% TERMPORANEA - Rimozione soluzioni con 'reload' dai corrieri
set of int: STEPS_NO_DEPOT = 2..(2*n_items);
constraint forall(c in COURIERS, snd in STEPS_NO_DEPOT)
                 (if delivery_order[snd,c] == 0 then delivery_order[snd + 1,c] = 0 endif);



% Objective function
% minimizzare la distanza
set of int: STEPS_NO_LAST = 1..(2*n_items);
array[COURIERS] of var int: a_max_dist; 
constraint forall(c in COURIERS)
                  (a_max_dist[c] = sum(s in STEPS_NO_LAST)
                  (all_distances[delivery_order[s,c] + 1,delivery_order[s + 1,c] + 1]));
var int: max_dist = max(a_max_dist);

%a_max_dist[c] = sum(delivery_order[i,c] | i in STEPS)(c in COURIERS);


%solve -> minimizzare la objective
%solve satisfy; 
solve minimize max_dist;