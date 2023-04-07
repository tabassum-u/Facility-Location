# reference help
# https://www.supplychaindataanalytics.com/multi-objective-linear-optimization-with-pulp-in-python/#:~:text=A%20multi%2Dobjective%20linear%20optimization%20problem%20is%20a%20linear%20optimization,or%20multi%2Dgoal%20linear%20programming.
# https://web.stanford.edu/group/sisl/k12/optimization/MO-unit5-pdfs/5.8Pareto.pdf

# this is rough idea of a problem
# please recheck your own problem


import pandas as pand
import numpy as np
import gurobipy as grb
import matplotlib.pyplot as plt
from pulp import *

# variables
warehouse_limits = "warehouse_limits"
customer_demands = "customer_demands"
fixed_costs = "fixed_costs"
cost_matrix = "cost_matrix"

demand = np.loadtxt("./probability_multiply_demand.csv", delimiter=",", dtype=int)
print('demand', demand)


def get_test_cases():
    a = np.loadtxt("./capacity.csv", delimiter=",", dtype=int)
    pd = demand
    f = np.loadtxt("./fixed_cost.csv", delimiter=",", dtype=int)
    c = np.loadtxt("./cost_matrix.csv", delimiter=",", dtype=float)
    s = np.loadtxt("./storage_cost.csv", delimiter=",", dtype=float)

    return a, pd, f, c, s


"""""
# For Gurobi Solver
"""""

SetObjWeight = []
w = 0
for i in range(101):
    weightage = [w, 1 - w]
    SetObjWeight.append(weightage)
    w += 0.01
print('weightage combination ', SetObjWeight)


def evaluate_fitness():
    a, pd, f, c, s = get_test_cases()

    n = len(a)  # warehouse
    m = len(pd)  # customer
    warehouses = range(n)
    customers = range(m)

    # model
    model = grb.Model("multi obj")

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
                objective = SetObjWeight[wei][p]*(sum([f[i] * open_or_close[i] for i in warehouses]) + \
                            (grb.quicksum(c[i, j] * demand_transfer[i, j] for i in warehouses for j in customers)))

            elif p == 1:
                objective = 2*SetObjWeight[wei][p]*(sum(a[i] * open_or_close[i] for i in warehouses) - \
                            (grb.quicksum(demand_transfer[i, j] for i in warehouses for j in customers)))

            model.setObjectiveN(objective, index = p)

        # Optimize
        model.optimize()
        objective_storing.append(model.ObjNVal)
        print(f"Obj  = {model.ObjNVal}")

        # Save problem
        model.write('multi obj.lp')

        nSolutions = model.SolCount
        nObjectives = model.NumObj
        nVariables = model.numVars
        print('Problem has', nObjectives, 'objectives for weightage ', SetObjWeight[wei] )
        print('Gurobi found', nSolutions, 'solutions for weightage ', SetObjWeight[wei] )
        print('found variables', nVariables, 'variables for weightage ', SetObjWeight[wei] )

        for var in model.getVars():
            if var.varName:
                decision_variable.append(abs(var.X))
        print_decision = np.array(decision_variable)
        print("decision variable array for weightage \n", SetObjWeight[wei], print_decision.reshape(10, 9))
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
    print("all objective ", print_objective_all)
    # print("all objective ", print_objective_all.reshape(len(SetObjWeight), 1))

    abc = np.array(SetObjWeight)
    bcd = np.array(print_objective_all)

    xy = abc * bcd[:, np.newaxis]
    x = [i[0] for i in xy]

    y = [i[1] for i in xy]

    print(y)
    plt.figure(figsize=(5,5), dpi=160)
    plt.title("")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.scatter(x, y, color="black", s=4.5)
    plt.axvline(x=305, linestyle='--', linewidth=0.5, color='black')
    plt.axhline(y=700, linestyle='--', linewidth=0.5, color='black')

    xx, yy = [min(x), max(x)], [max(y), min(y)]
    plt.plot(xx, yy, color='black')
    plt.rcParams.update({'font.size': 11})
    plt.show()

    return


"""""
#For PuLP Solver
"""""

a, pd, f, c, s = get_test_cases()
n = len(a)  # warehouse
m = len(pd)  # customer

warehouses = range(n)
customers = range(m)

facility = [0, 1, 2, 3, 4, 5, 6, 7, 8]
demand_area = [0, 1, 2, 3, 4, 5, 6, 7, 8]

nm = [(i, j) for i in warehouses for j in customers]

# define step-size
stepSize = 0.01

# initialize empty DataFrame for storing optimization outcomes
solutionTable = pand.DataFrame(columns=["weight", "obj_value"])

# iterate through alpha values from 0 to 1 with stepSize, and write PuLP solutions into solutionTable

pulp_objective = []

for w in range(0, 101, int(stepSize * 100)):

    # declare the problem again
    linearProblem = LpProblem("Multi-objective linear minimization", LpMinimize)

    # declare optimization variables, using PuLP
    open_or_close = LpVariable.dicts("open_or_close", facility, 0, 1, cat=LpBinary)
    demand_transfer = LpVariable.dicts("demand_transfer", nm, lowBound=0, cat=LpInteger)

    # add the objective function at sampled alpha
    linearProblem += ((w / 100) * (lpSum(f[i] * open_or_close[i] for i in facility) +
                                   (lpSum(c[i, j] * demand_transfer[i, j] for i in facility for j in demand_area)
                                    ))) + (2 * (1 - w / 100) * (lpSum(a[i] * open_or_close[i] for i in facility) -
                                                                  (lpSum(demand_transfer[i, j] for i in facility
                                                                   for j in demand_area))))

    # add the constraints

    for j in demand_area:  # demand constraint
        linearProblem += lpSum(demand_transfer[(i, j)] for i in facility) == pd[j]

    for i in facility:  # capacity constraint
        linearProblem += lpSum(demand_transfer[(i, j)] for j in demand_area) <= a[i] * open_or_close[i]

    for i in facility:  # domino location not select constraint
        if i == 4:
            linearProblem += open_or_close[i] == 0
        else:
            continue

    for j in demand_area:
        for i in facility:
            linearProblem += demand_transfer[(i, j)] <= pd[j] * open_or_close[i]

    # solve the problem
    linearProblem.solve()
    print("Status:", LpStatus[linearProblem.status])

    for v in linearProblem.variables():
        print(w, v.name, "=", v.varValue)

    pulp_objective.append(value(linearProblem.objective))
    # write solutions into DataFrame
    solutionTable.loc[int(w / (stepSize * 100))] = [w / 100,
                                                    value(linearProblem.objective)]

file_name = 'solution_data.csv'
for_csv_soln = solutionTable
for_csv_soln.to_csv(file_name)


print(solutionTable)
print("pulp obj \n ", pulp_objective)

dc = np.array(SetObjWeight)
gh = np.array(pulp_objective)

xyz=dc * gh[:, np.newaxis]

xa=[i[0] for i in xyz]

yb=[i[1] for i in xyz]


print(xyz)
plt.figure(figsize=(5, 5),dpi=170)
plt.title("")
plt.xlabel("F1")
plt.ylabel("F2")
plt.scatter(xa, yb, color ="black",s = 4)
plt.rcParams.update({'font.size': 11})
#ax = plt.gca()
#ax.set_ylim(ax.get_ylim()[::-1])

plt.axvline(x=305, linestyle = '--', linewidth = 0.5, color='black')
plt.axhline(y=700, linestyle = '--', linewidth = 0.5, color='black')

xxy, yyx = [min(xa), max(xa)], [max(yb), min(yb)]
plt.plot(xxy, yyx,  color='black')
plt.show()


def main():

    evaluate_fitness()


if __name__ == "__main__":
    main()

