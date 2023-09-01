include "alldifferent.mzn";
include "globals.mzn";

% CHOOSE THE INPUT FOR THE MODEL
%------------------------------

% include "inst01.dzn";    % 2 couriers, 6 items
% include "inst02.dzn";    % 6 couriers, 9 items
% include "inst03.dzn";    % 3 couriers, 7 items
% include "inst04.dzn";    % 8 couriers, 10 items
% include "inst05.dzn";    % 2 couriers, 3 items
% include "inst06.dzn";    % 6 couriers, 8 items
% include "inst07.dzn";    % 6 couriers, 17 items
% include "inst08.dzn";    % 8 couriers, 10 items
% include "inst09.dzn";    % 10 couriers, 13 items
% include "inst10.dzn";    % 10 couriers, 13 items
% include "inst11.dzn";    % 20 couriers, 143 items
% include "inst12.dzn";    % 20 couriers, 95 items
% include "inst13.dzn";    % 3 couriers, 47 items
% include "inst14.dzn";    % 20 couriers, 215 items
% include "inst15.dzn";    % 20 couriers, 239 items
% include "inst16.dzn";    % 20 couriers, 47 items
% include "inst17.dzn";    % 20 couriers, 287 items
% include "inst18.dzn";    % 20 couriers, 191 items
% include "inst19.dzn";    % 20 couriers, 71 items
% include "inst20.dzn";    % 20 couriers, 287 items
% include "inst21.dzn";    % 20 couriers, 143 items

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
set of int: STEPS = 1..steps; 
set of int: STEPS_NO_LAST = 1..steps-1;
set of int: STEPS_NO_FIRST_NO_LAST = 2..steps-1;

% int: lbound = max(all_distances[..,1])+ max(all_distances[1,..]);
int: ubound = sum(all_distances);

% float: lbound_f = max(all_distances[..,1])+ max(all_distances[1,..]);
float: ubound_f = sum(all_distances);
% DECISION VARIABLES
%------------------------------
% Correspondence between courier and items (chronological order)
array[STEPS, COURIERS] of var 0..n_items: delivery_order;

% array of the travelled distances for each courier
array[COURIERS] of var 0..sum(all_distances): courier_dist;


% CONSTRAINTS NECESSARY for problem resolution
%------------------------------
% All couriers depart and return to the origin
constraint forall(c in COURIERS)
                  (delivery_order[1,c] == 0 /\ delivery_order[n_items+2,c] == 0);

% Each package must be delivered only once
constraint alldifferent_except_0(delivery_order);

% Each package must be delivered
constraint count(delivery_order, 0, (steps*n_couriers)-n_items);

% Each carrier must not exceed its maximum load
constraint forall(c in COURIERS) (
  sum([size_item[delivery_order[i,c]] | i in STEPS where delivery_order[i,c] != 0]) <= max_load[c]);
                  
% Removing solutions with 'reload' from couriers
constraint forall(c in COURIERS, i in STEPS_NO_FIRST_NO_LAST)      
                 (delivery_order[i,c] != 0 \/ sum(delivery_order[i..,c]) == 0);

                 
% CONSTRAINTS to decrease problem resolution time
%------------------------------

% Constraint that guarantees an equal distrubution of the number of items among the couriers (maximum difference is two items)
% constraint forall(c in COURIERS,s in STEPS)
%                  (if c+((s-1)*n_couriers)> n_items+(2*n_couriers) then
%                  delivery_order[s,c] = 0 endif);

% Each courier deliver at least one item
constraint forall(c in COURIERS)(delivery_order[2,c] != 0 /\
              sum(delivery_order[steps-n_couriers+1..,c]) == 0);
        

% OBJECTIVE FUNCTION - MINIMIZE THE DISTANCE - MAXIMIZE FAIRNESS
%------------------------------
constraint forall(c in COURIERS)
                  (courier_dist[c] = sum(s in STEPS_NO_LAST)
                  (all_distances[delivery_order[s,c] + 1,delivery_order[s + 1,c] + 1] ));

% minimize the max of the distances 
% var lbound..ubound: max_dist = max(courier_dist);

% minimize the sum between the total length of all the couriers and the difference between the max and the min distance
% var lbound_f..ubound_f: obj_fun = max(courier_dist)+(sum(courier_dist)/(sum(courier_dist)+1));

% solve :: int_search(delivery_order, dom_w_deg, indomain_split)
% minimize obj_fun;

output 
    [show(obj_fun)] ++ [ "\n"] ++
    [show(delivery_order)] ++ [ "\n"] ++
    [show(courier_dist)] ++ [ "\n"]
;






