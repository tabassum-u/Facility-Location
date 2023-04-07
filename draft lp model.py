# https://www.supplychaindataanalytics.com/multi-objective-linear-optimization-with-pulp-in-python/#:~:text=A%20multi%2Dobjective%20linear%20optimization%20problem%20is%20a%20linear%20optimization,or%20multi%2Dgoal%20linear%20programming.

from io import StringIO
import numpy as np
import gurobipy as grb

from pulp import *

# variables
warehouse_limits = "warehouse_limits"
customer_demands = "customer_demands"
fixed_costs = "fixed_costs"
cost_matrix = "cost_matrix"


w_1 = 0.3
w_2 = 0.7
Weight = [w_1, w_2]
num_of_obj = [0, 1]

# if uncertain demand then following can be used

"""""
def calling_sum_alpha(sum_inside):
    sum_alpha = 0.3 * 9  # 0.3 and demand areas are 9
    return sum_inside <= sum_alpha


check_alpha = False

while check_alpha == False:
    sum_inside = 0
    alpha = []
    uncertain_demand = []
    for i in range(len(demand)):
        generate_random = random.uniform(0, 1)

        alpha.append(generate_random)

        sum_inside += alpha[i]

        uncertain_demand.append(math.ceil(demand[i] + alpha[i] * 0.2 * demand[i]))

    check_alpha = calling_sum_alpha(sum_inside)

print('uncertain demand', uncertain_demand)
up_bound_for_var = max(uncertain_demand)
"""

demand = np.loadtxt("./probability_multiply_demand.csv", delimiter=",", dtype=int)
print('demand', demand)

def get_test_cases():
    a = np.loadtxt("./capacity.csv", delimiter=",", dtype=int)
    pd = demand
    f = np.loadtxt("./fixed_cost.csv", delimiter=",", dtype=int)
    c = np.loadtxt("./cost_matrix.csv", delimiter=",", dtype=float)

    return a, pd, f, c


# multi-objective in linear
def create_lp_file(a, pd, f, c):
    name = "creating_lp.lp"
    n = len(a)  # warehouse size
    m = len(pd)  # customer size

    file_str = StringIO("")
    file_str.write("Minimize multi-objectives\n")

    # create objective function "obj"

    for k in range(0, 2):
        file_str.write("\n")
        file_str.write("OBJ")
        file_str.write(str(k))
        file_str.write(": ")
        file_str.write("Priority=1 ")
        file_str.write("Weight= ")
        file_str.write(str(Weight[k]))
        file_str.write(" AbsTol=0 ")
        file_str.write("RelTol=0 ")
        file_str.write("\n")

        if k == 0:
            for i in range(0, n):  # first objective function
                file_str.write(str(f[i]))
                file_str.write(" y")
                file_str.write(str(i))
                file_str.write(" + ")

            for i in range(0, n):
                for j in range(0, m):
                    file_str.write(str(c[i][j]))
                    file_str.write(" x")
                    file_str.write(str(i))
                    file_str.write("_")
                    file_str.write(str(j))
                    if not (i == n - 1 and j == m - 1):
                        file_str.write(" + ")

        else:  # second objective function
            file_str.write("\n")
            for i in range(0, n):

                file_str.write(" y")
                file_str.write(str(i))

                if not (i == n - 1):
                    file_str.write(" + ")

            file_str.write(" - ")
            for i in range(0, n):
                for j in range(0, m):

                    file_str.write(" x")
                    file_str.write(str(i))
                    file_str.write("_")
                    file_str.write(str(j))
                    if not (i == n - 1 and j == m - 1):
                        file_str.write(" - ")

    file_str.write("\nSubject To\n")

    # create constraints for customer demands
    for j in range(m):
        file_str.write("d")
        file_str.write(str(j))
        file_str.write(": ")
        for i in range(n):
            file_str.write("x")
            file_str.write(str(i))
            file_str.write("_")
            file_str.write(str(j))
            if (i != n - 1):
                file_str.write(" + ")
        file_str.write(" = ")
        file_str.write(str(pd[j]))
        file_str.write("\n")

    # create constraints for warehouse limits
    for i in range(n):
        file_str.write("a")
        file_str.write(str(i))
        file_str.write(": ")
        for j in range(m):
            file_str.write("x")
            file_str.write(str(i))
            file_str.write("_")
            file_str.write(str(j))
            if (j != m - 1):
                file_str.write(" + ")
        file_str.write(" - ")
        file_str.write(str(a[i]))
        file_str.write(" y")
        file_str.write(str(i))
        file_str.write(" <= 0\n")

    # constraint for domino location not to select (here 4)
    file_str.write(" y")
    file_str.write(str(4))
    file_str.write(" = 0\n")

    # constraint
    for i in range(n):

        for j in range(m):
            file_str.write("x")
            file_str.write(str(i))
            file_str.write("_")
            file_str.write(str(j))
            file_str.write(" - ")
            file_str.write(str(pd[j]))
            file_str.write(" y")
            file_str.write(str(i))
            file_str.write(" <= 0\n")

    file_str.write("Integers\n")  # integer variables
    for i in range(n):
        for j in range(m):
            file_str.write("x")
            file_str.write(str(i))
            file_str.write("_")
            file_str.write(str(j))
            file_str.write(" ")

    file_str.write("\nBinaries\n")  # binary variables
    for i in range(n):
        file_str.write("y")
        file_str.write(str(i))
        file_str.write(" ")

    f = open(name, "w+")
    f.write(file_str.getvalue())
    f.close()


def create_all_lp_files():
    a, pd, f, c = get_test_cases()
    create_lp_file(a, pd, f, c)


"""""
# For Gurobi Solver
"""""


def solve_lp_problem_own_write():
    lp = "creating_lp.lp"
    sol = "creating_sol.sol"

    model = grb.read(lp)
    model.setParam("TimeLimit", 10 * 60)  # time limit : 10 minutes
    model.optimize()
    f = open(sol, "w")
    f.write("objVal {0}\n".format(model.objNVal))
    f.write("RunTime {0}\n".format(round(model.runtime, 2)))
    if model.runtime > 10 * 60:
        f.write("is Optimal False\n")
    else:
        f.write("is Optimal True\n")
    for var in model.getVars():
        if (var.varName):
            f.write("{0} {1}\n".format(var.varName, abs(var.X)))


def evaluate_fitness():
    a, pd, f, c = get_test_cases()
    vars = []
    vars.append(a)
    vars.append(pd)
    vars.append(f)
    vars.append(c)

    SetObjPriority = [1, 1]
    SetObjWeight = []
    w = 0
    for i in range(101):
        weightage = [w, 1-w]
        SetObjWeight.append(weightage)
        w += 0.01
    print('weightage combination ', SetObjWeight)


    n = len(a)  # warehouse
    m = len(pd)  # customer
    sol_evaluate_fit = 'creating_ev_fit_sol.sol'

    warehouses = range(n)
    customers = range(m)

    # model
    model = grb.Model("multiobj")

    # decision variables
    demand_transfer = model.addVars(warehouses, customers, vtype=grb.GRB.INTEGER, name="demand_transfer")
    open_or_close = model.addVars(warehouses, vtype=grb.GRB.BINARY, name="open_or_close")

    # demand constraints
    model.addConstrs((demand_transfer.sum('*', j) == pd[j] for j in customers), "Demand")

    # capacity constraints
    model.addConstrs((demand_transfer.sum(i) <= a[i] * open_or_close[i] for i in warehouses), "Capacity")

    # domino constraints
    model.addConstrs(open_or_close[i] == 0 for i in warehouses if i == 4)
    model.addConstrs(demand_transfer[i, j] <= pd[j] * open_or_close[i] for i in warehouses for j in customers)

    # this is minimization
    model.ModelSense = grb.GRB.MINIMIZE

    # Limit how many solutions to collect
    model.setParam(grb.GRB.Param.PoolSolutions, 100)

    # multi-objective
    # Set and configure p-th objective

    objective_storing = []
    for wei in range(len(SetObjWeight)):

        decision_variable = []
        for p in range(2):
            if p == 0:
                objective = sum([f[i] * open_or_close[i] for i in warehouses]) + \
                            (grb.quicksum(c[i, j] * demand_transfer[i, j] for i in warehouses for j in customers))

            elif p == 1:
                objective = sum(a[i] * open_or_close[i] for i in warehouses) - \
                            (grb.quicksum(demand_transfer[i, j] for i in warehouses for j in customers))

            model.setObjectiveN(objective, p, SetObjPriority[p], SetObjWeight[wei][p], 1.0 + p, 0.01)

        # Optimize
        model.optimize()

        # Save problem
        model.write('multiobj.lp')

        model.setParam(grb.GRB.Param.OutputFlag, 0)

        nSolutions = model.SolCount
        nObjectives = model.NumObj
        nVariables = model.numVars
        print('Problem has', nObjectives, 'objectives for weightage ', SetObjWeight[wei] )
        print('Gurobi found', nSolutions, 'solutions for weightage ', SetObjWeight[wei] )
        print('found variables', nVariables, 'variables for weightage ', SetObjWeight[wei] )

        #f = open(sol_evaluate_fit, "w")
        for i in range(model.NumObj):
            model.setParam(grb.GRB.Param.ObjNumber, i)

            #f.write("objVal {0}\n".format(model.ObjNVal))
            #f.write("RunTime {0}\n".format(round(model.runtime, 2)))

            objective_storing.append(model.ObjNVal)

            print(f"Obj {i + 1} = {model.ObjNVal}")


        for var in model.getVars():
            if var.varName:
                decision_variable.append(abs(var.X))
        print_decision = np.array(decision_variable)
        #print("decision variable array for weightage \n", SetObjWeight[wei] , print_decision.reshape(10, 9))
        # last row is binary variable

        # Status checking
        status = model.Status
        if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
            print("The model cannot be solved because it is infeasible or "
                  "unbounded")
            sys.exit(1)

        if status != grb.GRB.OPTIMAL:
            print('Optimization was stopped with status ' + str(status))
            sys.exit(1)

    print_objective_all = np.array(objective_storing)
    print("all objective ", print_objective_all.reshape(len(SetObjWeight), 2))

    return

def main():
    create_all_lp_files()
    solve_lp_problem_own_write()
    evaluate_fitness()


if __name__ == "__main__":
    main()

