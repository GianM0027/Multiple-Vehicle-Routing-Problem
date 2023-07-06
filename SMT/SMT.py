from z3 import *

# Create the solver
solver = Solver()

# Create the variables
x = Int('x')
y = Int('y')

# Add constraints to the solver
solver.add(x >= 0)
solver.add(y >= 0)
solver.add(x + y == 10)

# Check if the constraints are satisfiable
if solver.check() == sat:
    # If satisfiable, get the model
    model = solver.model()
    # Print the values of the variables
    print("x =", model[x])
    print("y =", model[y])
else:
    print("Constraints are unsatisfiable.")