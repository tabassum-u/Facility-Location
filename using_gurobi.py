from io import StringIO
import pandas as pand
import numpy as np
import gurobipy as grb
import matplotlib.path as mpath
import matplotlib.pyplot as plt

# variables

w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

a = np.loadtxt("./capacity.csv", delimiter=",", dtype=int)
pd = np.loadtxt("./probability_multiply_demand.csv", delimiter=",", dtype=int)
f = np.loadtxt("./fixed_cost.csv", delimiter=",", dtype=int)
c = np.loadtxt("./cost_matrix.csv", delimiter=",", dtype=float)
s = np.loadtxt("./storage_cost.csv", delimiter=",", dtype=float)
c_1_d = np.loadtxt("./cost_matrix_one_dim.csv", delimiter=",", dtype=float)
s_for_fun_2 = np.loadtxt("./storage_cost_fun2.csv", delimiter=",", dtype=float)

n = len(a)  # warehouse size
m = len(pd)  # customer size

function_1 = []
function_2 = []

def create_solve_lp(a, pd, f, c, s, n, m, k):

    x_value = []
    y_value = []

    name = "creating_lp" + str(k) + ".lp"

    file_str = StringIO("")
    file_str.write("Minimize\n")
    # create objective function "obj"

    for i in range(0, n):
        file_str.write(str(f[i] * w1))
        file_str.write(" y")
        file_str.write(str(i))

        file_str.write(" + ")

    for i in range(0, n):
        for j in range(0, m):

            file_str.write(str(c[i][j] * w1))

            file_str.write(" x")
            file_str.write(str(i))
            file_str.write("_")
            file_str.write(str(j))
            if not (i == n - 1 and j == m - 1):
                file_str.write(" + ")
    file_str.write(" + ")

    for i in range(0, n):
        file_str.write(str(a[i] * s[i] * w2))

        file_str.write(" y")
        file_str.write(str(i))

        if not (i == n - 1 and j == m - 1):
            file_str.write(" + ")

    file_str.write(" - ")
    for i in range(0, n):
        for j in range(0, m):

            file_str.write(str(s[i] * w2))

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

    # constraint for number of facilities
    for i in range(n):
        file_str.write(" y")
        file_str.write(str(i))
        if (i != n - 1):
            file_str.write(" + ")
    file_str.write(" <= 3\n")

    file_str.write("Bounds\n")
    # putting bound for domino location 4 not to select
    file_str.write("y")
    file_str.write(str(4))
    file_str.write(" = 0\n")

    file_str.write("Integers\n")
    # integer variables
    for i in range(n):
        for j in range(m):
            file_str.write("x")
            file_str.write(str(i))
            file_str.write("_")
            file_str.write(str(j))
            file_str.write(" ")

    file_str.write("\nBinaries\n")
    # binary variables
    for i in range(n):
        file_str.write("y")
        file_str.write(str(i))
        file_str.write(" ")

    f = open(name, "w+")
    f.write(file_str.getvalue())
    f.close()

    lp = "creating_lp" + str(k) + ".lp"
    sol = "creating_sol" + str(k) + ".sol"
    y_var = "creating_y" + str(k) + ".sol"
    x_var = "creating_x" + str(k) + ".sol"
    model = grb.Model("CWLP")
    model = grb.read(lp)
    model.setParam("TimeLimit", 10 * 60)  # time limit : 10 minutes
    model.optimize()

    f = open(sol, "w")
    f.write("objVal {0}\n".format(model.objVal))
    f.write("RunTime {0}\n".format(round(model.runtime, 2)))
    if model.runtime > 10 * 60:
        f.write("isOptimal False\n")
    else:
        f.write("isOptimal True\n")

    g = open(y_var, "w")
    for var in model.getVars():
        if (var.varName.startswith("y")):
            y_value.append(abs(var.X))   #for int value skip format part
            g.write("{0}\n".format(abs(var.X)))

    h = open(x_var, "w")
    for var in model.getVars():
        if (var.varName.startswith("x")):
            x_value.append(abs(var.X))
            h.write("{0}\n".format(abs(var.X)))

    return y_value, x_value

for k in range(len(w)):
    w1 = w[k]
    w2 = 1 - w1
    create_solve_lp(a, pd, f, c, s, n, m, k)

    y_in_use = create_solve_lp(a, pd, f, c, s, n, m, k)[0]
    x_in_use = create_solve_lp(a, pd, f, c, s, n, m, k)[1]
    print("y in use", y_in_use)

    sum_f_1 = 0
    for i in range(n):
        sum_f_1 += f[i] * y_in_use[i]

    for i in range(81): # range 81
        sum_f_1 += c_1_d[i] * x_in_use[i]

    print("sum_f_1", sum_f_1)

    function_1.append(sum_f_1)

    sum_f_2 = 0
    for i in range(n):
        sum_f_2 += a[i] * s[i] * y_in_use[i]

    for i in range(81): # range 81
        sum_f_2 -= (s_for_fun_2[i] * x_in_use[i])

    print("sum_f_2", sum_f_2)

    function_2.append(sum_f_2)

print(function_1)
print(function_2)

# plotting
x = function_1
y = function_2
fig, ax = plt.subplots()
ax.set_xlabel('Cost')
ax.set_ylabel('Unutilized Capacity')
ax.set_title('Pareto Optimal Front (Weighted Sum Method)')

star = mpath.Path.unit_regular_star(4)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)
plt.plot(x, y, '--b', marker=cut_star, markersize=9)
plt.show()

