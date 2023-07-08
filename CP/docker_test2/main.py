from minizinc import Instance, Model, Solver
import numpy as np
import datetime
import json

def set_model(configuration):
    model = Model()
    model.add_file("mainCode.mzn")

    if configuration == 'impliedConsMaxDist' or configuration == 'impliedConsObjFun':
        model.add_string("""
        constraint forall(c in COURIERS,s in STEPS)(if c+((s-1)*n_couriers)> n_items+(2*n_couriers) then delivery_order[s,c] = 0 endif);
        """)

    if configuration == 'defaultModelMaxDist' or configuration == 'impliedConsMaxDist':
        model.add_string("""
        var int: obj_fun = max(courier_dist);
        solve :: int_search(delivery_order, dom_w_deg, indomain_split) minimize obj_fun;   
        """)

    if configuration == 'defaultModelObjFun' or configuration == 'impliedConsObjFun':
        model.add_string("""
        var int: obj_fun = sum(courier_dist)+ max(courier_dist)- min(courier_dist);
        solve :: int_search(delivery_order, dom_w_deg, indomain_split) minimize obj_fun;   
        """)

    return model

def get_results(result, n_couriers, n_items, timeout):

    lines = str(result.solution).split("\n")
    try: objective = int(lines[0])
    except: objective = lines[0]
    # if lines[0] == None:
    #     objective = lines[0]
    # else:
    #     objective = int(lines[0])
    if len(lines) > 1:

        delivery_order = lines[1].strip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ[]= ;\n").replace(",","")
        delivery_order = list(delivery_order.split(' '))
        delivery_order = [int(i) for i in delivery_order]
        delivery_order = np.array(delivery_order).reshape(n_items+2,n_couriers).T 

        sol = list()
        for i in range(n_couriers):
            c_sol = list()
            for j in range(n_items +2):
                if delivery_order[i][j] != 0:
                    c_sol.append(int(delivery_order[i][j]))
            sol.append(c_sol)
    else:
        sol = None

    try: runTime = int(result.statistics['time'].total_seconds())
    except: runTime = int(timeout.total_seconds())

    status = (result.status == result.status.OPTIMAL_SOLUTION)

    return(objective,sol,runTime,status)   


def solve_model(n_inst,model,solver_str,timeout):

    solver = Solver.lookup(solver_str)

    # Transform Model into a instance
    inst = Instance(solver, model)

    # Instantiate variables from file
    instances_path = "Instances_CP_blank/" + str(n_inst) + ".dzn"
    data_file = open(instances_path)
    lines = []
    for line in data_file:
        lines.append(line)
    data_file.close()
    n_couriers = int(lines[0].rstrip('\n'))
    n_items = int(lines[1].rstrip('\n'))
    max_load = list(map(int, lines[2].rstrip('\n').split()))
    size_item = list(map(int, lines[3].rstrip('\n').split()))
    for i in range(4, len(lines)):
        lines[i] = lines[i].rstrip('\n').split()

    for i in range(4, len(lines)):
        lines[i] = [lines[i][-1]] + lines[i]
        del lines[i][-1]

    dist = np.array([[lines[j][i] for i in range(len(lines[j]))] for j in range(4, len(lines))])

    last_row = dist[-1]
    dist = np.insert(dist, 0, last_row, axis=0)
    dist = np.delete(dist, -1, 0)

    dist = dist.astype(int)

    inst["n_couriers"] = n_couriers
    inst["n_items"] = n_items
    inst["max_load"] = max_load
    inst["size_item"] = size_item
    inst["all_distances"] = dist

    # Output
    result = inst.solve(timeout = timeout)
    return(result, n_couriers, n_items)


# Model confugurations
DEFAULT_MODEL_MAX_DIST = "defaultModelMaxDist"
DEFAULT_MODEL_OBJ_FUN = "defaultModelObjFun"
IMPLIED_CONS_MAX_DIST = "impliedConsMaxDist"
IMPLIED_CONS_OBJ_FUN = "impliedConsObjFun"

# every element in configurations corresponds to a specific configuration of the model
configurations = [DEFAULT_MODEL_MAX_DIST, DEFAULT_MODEL_OBJ_FUN, IMPLIED_CONS_MAX_DIST, IMPLIED_CONS_OBJ_FUN]

solvers = ["gecode","chuffed"]
timeout = datetime.timedelta(milliseconds= 300000)

valid_in = False
while not valid_in:
    n_inst = input("\nSelect the instance number (1-21) = ")
    if int(n_inst) > 0 and int(n_inst) <= 21:
        n_inst = int(n_inst)
        valid_in = True
    else: print("Please, insert a valid input\n")
valid_in = False

print("\n1:"+configurations[0])
print("2:"+configurations[1])
print("3:"+configurations[2])
print("4:"+configurations[3])
while not valid_in:
    n_conf = input("\nSelect the configuration (1-4) = ")
    if int(n_conf) > 0 and int(n_conf) <= 4:
        n_conf = int(n_conf)
        valid_in = True
    else: print("Please, insert a valid input\n")
valid_in = False

print("1:"+solvers[0])
print("2:"+solvers[1])
while not valid_in:
    n_solv = input("\nSelect the solver (1-2) = ")
    if int(n_solv) > 0 and int(n_solv) <= 2:
        n_solv = int(n_solv)
        valid_in = True
    else: print("Please, insert a valid input\n")
valid_in = False

if int(n_inst) < 10:
    inst = '0'+str(n_inst)
else:
    inst = str(n_inst)

model = set_model(configurations[n_conf-1])
result, n_couriers, n_items = solve_model(inst,model,solvers[n_solv-1],timeout)
obj,solution,runTime,status = get_results(result, n_couriers, n_items, timeout)
print("\n###### RESULTS ######\n")
print("Time = ", runTime)
print("Optimal = ", status)
print("Objective = ", obj)
print("Solution = ", solution)