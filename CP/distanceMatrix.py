import csv
import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

temp = "inst01"
instances = {}
MPC_instances_path = "instances/"
with os.scandir(MPC_instances_path) as inst_list:
    for inst in inst_list:
        inst_path = MPC_instances_path + inst.name
        with open(inst_path) as f:
            reader = csv.reader(f, delimiter=' ')
            data = list(reader)
            instances[inst.name] = {}
            M = int(data[0][0])
            N = int(data[1][0])
            Capacities = [int(n) for n in data[2]]
            Weights = [int(n) for n in data[3]]
            x = [int(n) for n in data[4]]
            y = [int(n) for n in data[5]]
            instances[inst.name]['M'] = M
            instances[inst.name]['N'] = N
            instances[inst.name]['Capacities'] = Capacities
            instances[inst.name]['Weights'] = Weights
            instances[inst.name]['dx']= x
            instances[inst.name]['dy']= y
            instances[inst.name]['distance_matrix'] = np.array(
                [[abs(x[i]-x[j])+abs(y[i]-y[j]) for i in range(N+1)] for j in range(N+1)])

for key in instances[temp].keys():
    if key in ["N","M","Capacities","Weights"]  :
        print(key,"=", instances[temp][key])

def To_minizinc_format(obj:str):
    obj =  obj[0] + '|' + obj[2:]
    obj = obj[0:-2] + '|' + obj[-1]
    obj = obj.replace("\n","")
    obj = obj[0] + obj[1:-1].replace("[","|") + obj[-1]
    obj = obj[0] + obj[1:-1].replace("]","|") + obj[-1]
    obj = obj[0] + obj[1:-1].replace("| |","|") + obj[-1]
    obj = obj[0] + obj[1:-1].replace("| ","|") + obj[-1]
    obj = obj[0] + obj[1:-1].replace("| ","|") + obj[-1]
    obj = obj[0] + obj[1:-1].replace("     ",",") + obj[-1]
    obj = obj[0] + obj[1:-1].replace("    ",",") + obj[-1]
    obj = obj[0] + obj[1:-1].replace("   ",",") + obj[-1]
    obj = obj[0] + obj[1:-1].replace("  ",",") + obj[-1]
    obj = obj[0] + obj[1:-1].replace(" ",",") + obj[-1]
    obj = "Distance = " + obj
    return obj

with open('Distance Matrix.txt', 'w') as f:
    f.write(To_minizinc_format(str(instances[temp]["distance_matrix"])))
