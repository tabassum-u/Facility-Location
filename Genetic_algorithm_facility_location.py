import random as rnd
import numpy as np
import operator
import matplotlib.path as mpath
import time
import matplotlib.pyplot as plt

# variables
size = 30
best_sample = 10
lucky_few = 10
number_of_child = 2
number_of_generation = 50
chance_of_mutation = 10

# weightage
w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

function_1 = []
function_2 = []

# a is capacity of facility
# pd is customer demands multiplied with probability
# f is fixed cost for facility, c is cost matrix for transportation, s is storage cost in facility
# n is warehouse count, m is customer count


def genetic_algorithm():
    for k in range(len(w)):
        w_1 = w[k]
        w_2 = 1 - w_1
        solution_dic = {}

        def get_test_cases():
            a = np.loadtxt("./capacity.csv", delimiter=",", dtype=int)
            pd = np.loadtxt("./probability_multiply_demand.csv", delimiter=",", dtype=int)
            f = np.loadtxt("./fixed_cost.csv", delimiter=",", dtype=int)
            c = np.loadtxt("./cost_matrix.csv", delimiter=",", dtype=float)
            s = np.loadtxt("./storage_cost.csv", delimiter=",", dtype=float)
            return a, pd, f, c, s

        def is_feasible(vars, y):
            a = vars[0]
            pd = vars[1]
            demands = sum(pd)
            y = [int(s) for s in y]
            capacity = sum(a * y)
            sum_y = 0
            for i in range(9):
                sum_y += int(y[i])

            # checking whether capacity is greater than demand and total facility number will be less than or equal 3
            return capacity >= demands and sum_y <= 3

        def generate_first_random_feasible_sol(vars):
            a = vars[0]
            pd = vars[1]
            n = len(a)
            sol_feasible = False

            while sol_feasible == False:
                y = [str(rnd.randint(0, 1)) for i in range(n)]

                # for excluding the location where domino probability is high. here location index [4]
                if y[4] == str(1):
                    y[4] = str(0)

                sol_feasible = is_feasible(vars, y)

            return ''.join(y)

        def fitness(vars, y):
            a = vars[0].copy()
            pd = vars[1].copy()
            f = vars[2]
            c = vars[3].copy()
            s = vars[4].copy()
            n = len(a)  # facility number
            m = len(pd)  # customer number

            decision_var_array = np.zeros(shape=(9, 9))
            total = 0
            for i in range(n):

                if y[i] == '0':
                    a[i] = 0
                    c[i, :] = 2.0
                else:
                    total += (w_1 * f[i]) + (w_2 * a[i] * s[i])

            for i in range(m):
                demand_satisfied = False
                while demand_satisfied == False:
                    ind = np.argmin(c[:, i]) # all row of column i
                    if pd[i] <= a[ind]:
                        total += ((w_1 * c[ind][i] * pd[i]) - (w_2 * s[ind] * pd[i]))
                        a[ind] -= pd[i]
                        demand_satisfied = True
                        decision_var_array[ind][i] = pd[i]
                    else:
                        total += ((w_1 * c[ind][i] * a[ind]) - (w_2 * s[ind] * a[ind]))
                        pd[i] -= a[ind]
                        decision_var_array[ind][i] = a[ind]
                        a[ind] = 0
                        c[ind, :] = 2
                        demand_satisfied = pd[i] == 0

            return total, decision_var_array

        def generate_first_population(size, vars):  # this size is for pop size, size is 30
            population = []

            # here size is 30. and also in range(size) is range(30)
            for i in range(size):
                # always try to create a unique random solution
                sol_exists = True
                while sol_exists:
                    new_sol = generate_first_random_feasible_sol(vars)
                    sol_exists = new_sol in population

                population.append(new_sol)

            return population

        def compute_fitness_population(population, vars):
            population_fitness = {}
            for individual in population:
                if individual in solution_dic:

                    population_fitness[individual] = solution_dic[individual]

                else:
                    population_fitness[individual] = fitness(vars, individual)
                    print("fitness", population_fitness[individual])
                    solution_dic[individual] = population_fitness[individual]

            return sorted(population_fitness.items(), key=operator.itemgetter(1), reverse=False)  # best is first

        def select_from_population(population_sorted, best_sample, lucky_few):
            next_generation = []
            population_array = np.array(population_sorted)[:, 0].tolist()
            population_array.reverse()

            for i in range(best_sample):
                next_generation.append(population_array.pop())
            for i in range(lucky_few):
                selected_sol = rnd.choice(population_array)
                next_generation.append(selected_sol)
                population_array.remove(selected_sol)
            rnd.shuffle(next_generation)
            return next_generation

        def create_child(indivudual1, indivudual2, vars):
            child_is_feasible = False
            while child_is_feasible == False:
                child = ''
                for i in range(len(indivudual1)):
                    if int(100 * rnd.random()) < 50:
                        child += indivudual1[i]
                    else:
                        child += indivudual2[i]
                child_is_feasible = is_feasible(vars, child)
            return child

        def create_children(breeders, number_of_child, vars):
            next_population = []

            next_population.extend(breeders[:best_sample])

            for i in range(int(len(breeders) / 2)):
                for j in range(number_of_child):
                    child_exists = True
                    while child_exists:
                        random_breeders = rnd.choices(breeders, k=2)

                        new_child = create_child(random_breeders[0], random_breeders[1], vars)
                        child_exists = new_child in next_population

                    next_population.append(new_child)

            return next_population

        def mutate_solution(y, vars):
            mutated_sol_feasible = False
            while mutated_sol_feasible == False:
                y0 = list(y)
                for i in range(len(y0)):

                    y0[i] = str(rnd.randint(0, 1))
                    if y0[4] == str(1):
                        y0[4] = str(0)

                y0 = ''.join(y0)

                mutated_sol_feasible = is_feasible(vars, y0)
            return y0

        def mutate_population(population, chance_of_mutation, vars):
            for i in range(len(population)):
                if rnd.random() * 100 < chance_of_mutation:
                    new_sol = mutate_solution(population[i], vars)
                    mutated_sol_exist = new_sol in population
                    if mutated_sol_exist == False:
                        population[i] = new_sol
            return population

        def next_generation(first_generation, vars, best_sample, lucky_few, number_of_child, chance_of_mutation):
            population_sorted = compute_fitness_population(first_generation, vars)
            next_breeders = select_from_population(population_sorted, best_sample, lucky_few)
            next_population = create_children(next_breeders, number_of_child, vars)
            next_generation = mutate_population(next_population, chance_of_mutation, vars)
            return next_generation

        def multiple_generation(number_of_generation, vars, size, best_sample, lucky_few, number_of_child,
                                chance_of_mutation):
            historic = []
            historic.append(generate_first_population(size, vars))
            for i in range(number_of_generation):
                historic.append(
                    next_generation(historic[i], vars, best_sample, lucky_few, number_of_child, chance_of_mutation))
            return historic

        def get_best_individual_from_population(population, vars):
            
            return compute_fitness_population(population, vars)[0]

        def get_list_best_individual_from_history(historic, vars):
            best_individuals = []

            for population in historic:
                best_individuals.append(get_best_individual_from_population(population, vars))

                # here it is 30, population size and this is occurring for number of generation times

            return best_individuals

        def evolution_best_fitness(historic, best_sol):
            evolution_fitness = []
            for population in historic:
                evolution_fitness.append(get_best_individual_from_population(population, vars)[1][0])
            plt.title("Best solution : " + best_sol[0])
            evolution_fitness.sort(reverse=True)
            plt.plot(evolution_fitness)
            plt.ylabel("fitness best individual")
            plt.xlabel("generation")
            plt.show()

        def print_simple_result(historic, number_of_generation, vars):
            a, pd, f, c, s = get_test_cases()

            result_all = np.array(get_list_best_individual_from_history(historic, vars), dtype=object)
            index = np.argmin(result_all[:, 1])
            print("index for min result:", index)
            result = result_all[index]
            print("result", result)

            y_str = list(result[0])
            print("y str", y_str)
            y_genetic = []
            for i in range(len(y_str)):
                y_genetic.append(int(y_str[i]))

            transfer_amount_str = np.array(result[1][1])
            transfer_amount = transfer_amount_str
            print("y gene", y_genetic)
            print("transfer amount", transfer_amount)

            sum_f_1 = 0
            for i in range(9):
                sum_f_1 += (f[i] * y_genetic[i])

            for i in range(9):
                for j in range(9):
                    sum_f_1 += c[i][j] * transfer_amount[i][j]

            print("sum_f_1", sum_f_1)

            function_1.append(sum_f_1)

            sum_f_2 = 0
            for i in range(9):
                sum_f_2 += a[i] * s[i] * y_genetic[i]

            for i in range(9):
                for j in range(9):
                    sum_f_2 -= (s[i] * transfer_amount[i][j])

            print("sum_f_2", sum_f_2)

            function_2.append(sum_f_2)
            print(function_1)
            print(function_2)

            print("solution: " + str(result[0]) + " fitness: " + str(result[1][0]) + " decision variable: " + str(
                result[1][1]))

            return result

        a, pd, f, c, s = get_test_cases()
        vars = []
        vars.append(a)
        vars.append(pd)
        vars.append(f)
        vars.append(c)
        vars.append(s)
        t0 = time.time()
        historic = multiple_generation(number_of_generation, vars, size, best_sample, lucky_few, number_of_child,
                                       chance_of_mutation)
        t1 = time.time()
        best_sol = print_simple_result(historic, number_of_generation, vars)

        runtime = t1 - t0
        print("time : ", runtime)
        print("best sol : ", best_sol)
        sol = "creating_gen" + str(k) + ".sol"
        f = open(sol, "w")
        f.write("objVal {0}\n".format(best_sol[1][0]))
        f.write("transferredAmount {0}\n".format(best_sol[1][1]))
        f.write("RunTime {0}\n".format(round(runtime, 2)))

        for i in range(len(best_sol[0])):
            f.write("y{0} {1}\n".format(str(i), best_sol[0][i]))
        f.close()

        evolution_best_fitness(historic, best_sol)


def plotting_pareto():
    # plotting
    x = function_1
    y = function_2
    fig, ax = plt.subplots()
    ax.set_xlabel('Cost')
    ax.set_ylabel('Unutilized Capacity')
    ax.set_title('Pareto Optimal Front')

    star = mpath.Path.unit_regular_star(4)
    circle = mpath.Path.unit_circle()
    # concatenate the circle with an internal cutout of the star
    verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
    codes = np.concatenate([circle.codes, star.codes])
    cut_star = mpath.Path(verts, codes)
    plt.plot(x, y, '--b', marker=cut_star, markersize=9)
    plt.show()


def main():
    genetic_algorithm()
    plotting_pareto()


if __name__ == "__main__":
    main()