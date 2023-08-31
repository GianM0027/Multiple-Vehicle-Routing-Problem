import numpy as np
from z3 import *

# returns True if two lists of booleans are equal
def equal_counts(a, b):
    if len(a) != len(b):
        return False
    return And([ai == bi for ai, bi in zip(a, b)])

def sum_binary_digits(l):
    boolean = [0]

    for value in l:
        boolean = sum_binary(boolean, [value])

    return boolean

#sum of two binary numbers a and b represented as a list of booleans
def sum_binary(x, y):
    length = max(len(x), len(y))

    s = Solver()

    # both addends x and y must have same lenght

    a = [Bool(f"a_{i}") for i in range(length)]
    b = [Bool(f"b_{i}") for i in range(length)]


    for i in range(len(x)):
        if type(x[i]) != BoolRef:
            s.add(If(bool(x[i]), a[i], Not(a[i])))
        else:
            s.add(Implies(x[i], a[i]))
            s.add(Implies(Not(x[i]), Not(a[i])))

    for i in range(len(y)):
        if type(y[i]) != BoolRef:
            s.add(If(bool(y[i]), b[i], Not(b[i])))
        else:
            s.add(If(y[i], b[i], Not(b[i])))


    solution = [Bool(f"sol_{i}") for i in range(length+1)]
    carry = [Bool(f"carry_{i}") for i in range(length+1)]

    # sum constraints
    s.add(Not(carry[length]))

    for i in range(-1, -length-1, -1):
        # 0 + 1 + 1 = 0 with carry = 1  ///  0 + 1 + 0 = 1 with carry = 0
        s.add(Implies(Xor(a[i], b[i]), If(carry[i], Not(solution[i]), solution[i])))
        s.add(Implies(Xor(a[i], b[i]), If(carry[i], carry[i - 1], Not(carry[i - 1]))))

        # 1 + 1 + 1 = 1 with carry = 1  ///  0 + 1 + 1 = 0 with carry = 1
        s.add(Implies(And(a[i], b[i]), If(carry[i], solution[i], Not(solution[i]))))
        s.add(Implies(And(a[i], b[i]), carry[i - 1]))

        # 0 + 0 + 0 = 0 with carry 0   ///  0 + 0 + 1 = 1 with carry 0
        s.add(Implies(And(Not(a[i]), Not(b[i])), If(carry[i], solution[i], Not(solution[i]))))
        s.add(Implies(And(Not(a[i]), Not(b[i])), Not(carry[i - 1])))

    s.add(Implies(carry[0], solution[0]))

    if s.check() == sat:
        model = s.model()
        print("a -> ", [1 if model.evaluate(a[i]) == True else 0 for i in range(l_a)])
        print("b -> ", [1 if model.evaluate(b[i]) == True else 0 for i in range(l_b)])
        solution = [1 if model.evaluate(solution[i]) == True else 0 for i in range(length+1)]
        while solution and solution[0] == 0:
            solution.pop(0)
        print(solution)
        return solution
    else:
        return False



#sum of two lists of decision variables
def sum_bool_ref(a, b, sol):
    length = max(len(a), len(b))

    s = Solver()

    solution = [Bool(f"sol_{i}") for i in range(length+1)]
    carry = [Bool(f"carry_{i}") for i in range(length+1)]

    # sum constraints
    s.add(Not(carry[length]))

    for i in range(-1, -length-1, -1):
        # 0 + 1 + 1 = 0 with carry = 1  ///  0 + 1 + 0 = 1 with carry = 0
        s.add(Implies(Xor(a[i], b[i]), If(carry[i], Not(solution[i]), solution[i])))
        s.add(Implies(Xor(a[i], b[i]), If(carry[i], carry[i - 1], Not(carry[i - 1]))))

        # 1 + 1 + 1 = 1 with carry = 1  ///  0 + 1 + 1 = 0 with carry = 1
        s.add(Implies(And(a[i], b[i]), If(carry[i], solution[i], Not(solution[i]))))
        s.add(Implies(And(a[i], b[i]), carry[i - 1]))

        # 0 + 0 + 0 = 0 with carry 0   ///  0 + 0 + 1 = 1 with carry 0
        s.add(Implies(And(Not(a[i]), Not(b[i])), If(carry[i], solution[i], Not(solution[i]))))
        s.add(Implies(And(Not(a[i]), Not(b[i])), Not(carry[i - 1])))

    s.add(Implies(carry[0], solution[0]))

    if s.check() == sat:
        model = s.model()
        print("a -> ", [1 if model.evaluate(a[i]) == True else 0 for i in range(l_a)])
        print("b -> ", [1 if model.evaluate(b[i]) == True else 0 for i in range(l_b)])
        solution = [1 if model.evaluate(solution[i]) == True else 0 for i in range(length+1)]
        while solution and solution[0] == 0:
            solution.pop(0)
        print(solution)
        return solution
    else:
        return False


####### TEST SUM
s = Solver()
l_a = 3
l_b = 3

a = [Bool(f"a_{i}") for i in range(l_a)] # 010
b = [Bool(f"b_{i}") for i in range(l_b)] # 010

s.add(Not(a[0]))
s.add(a[1])
s.add(Not(a[2]))

s.add(Not(b[0]))
s.add(b[1])
s.add(Not(b[2]))

print("Sum function")
#s.add(sum_bool_ref(a,b,[1,0,0]))

if s.check() == sat:
    model = s.model()
    print("\nActual values")
    print("a -> ", [1 if model.evaluate(a[i]) == True else 0 for i in range(l_a)])
    print("b -> ", [1 if model.evaluate(b[i]) == True else 0 for i in range(l_b)])
    print("[1,0,0]")
else:
    print("UNSAT")