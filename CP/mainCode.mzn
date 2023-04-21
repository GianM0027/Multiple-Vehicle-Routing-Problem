include "alldifferent.mzn";
include "globals.mzn";

% Input data
int: n_couriers; % number of couriers
int: n_items; % number of items

array[1..n_couriers] of int: max_load; % maximum load size for each courier
array[0..n_items] of int: size_item; % size of each item (AGGIUSTA INDICI ARRAY)
array[1..n_items] of int: distance_to_origin; % distance between each distribution point and the origin point
array[1..n_items] of int: distance_from_origin; % distance between the origin point and each distribution point
array[1..n_items,1..n_items] of int: distance_between_destinations; % distance between each distribution point


% set of
set of int: COURIERS = 1..n_couriers;
set of int: ITEMS = 1..n_items;
set of int: STEPS = 1..(2*n_items+1);
var float: sum_items = n_items*(n_items+1)/2;
array[COURIERS] of int: load_couriers = [];


% Decision variables
%array[1..n_items, 1..n_couriers] of var 0..1: item_assignment; % binary variable -> a courier deliver a package or not
array[1..(2*n_items+1), 1..n_couriers] of var 0..n_items: delivery_order; %correspondence between courier and items (by chronological order), 2*n_items+1 perchè se un pacco è pesante il corriere deve fare avanti e indietro
var int: max_distance; % maximum distance travelled by any courier


%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT TEST %%%%%%%%%%%%%%%%%%%%%%%%%%
n_couriers = 3;
n_items = 7;
size_item = [0,3, 2, 6, 8, 5, 4, 4]; %ATTENTO ALLO ZERO ALL'INIZIO
max_load = [15,10,7];

distance_between_destinations = [|0, 3, 3, 6, 5, 6, 6 | 3, 0, 4, 3, 4, 7, 7 | 3, 4, 0, 7, 6, 3, 5 | 6, 3, 7, 0, 3, 6, 6 | 5, 4, 6, 3, 0, 3, 3 | 6, 7, 3, 6, 3, 0, 2 | 6, 7, 5, 6, 3, 2, 0 |];

distance_to_origin = [2, 3, 3, 4, 3, 4, 4];

distance_from_origin  = [2, 3, 3, 4, 3, 4, 4];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Constraints 
/*
- Tutti i corrieri partono e tornano all'origine
- Ogni pacco deve essere consegnato una sola volta
- Ogni pacco deve essere consegnato
- Ogni corriere non deve superare il suo carico massimo
- 
*/

%Tutti i corrieri partono e tornano all'origine
constraint forall(i in COURIERS)
                  (delivery_order[1,i] == 0 /\ delivery_order[n_items,i] == 0);

%Ogni pacco deve essere consegnato una sola volta
constraint alldifferent_except_0(delivery_order);

%Ogni pacco deve essere consegnato
constraint sum(delivery_order) == sum_items;

% Ogni corriere non deve superare il suo carico massimo
forall(c in COURIERS)(
                forall(i in ITEMS)(
                        load_couriers[c] = load_couriers[c] + size_item[delivery_order[i,c]]
                )
          );

constraint forall(c in COURIERS)(
                load_couriers[c] <= max_load[c]
           );


/*
constraint forall(i in ITEMS)(
            forall(c in COURIERS)(
                  i in delivery_order
            )
      );
*/

% Objective function
% minimizzare la distanza


%solve -> minimizzare la objective
solve satisfy;