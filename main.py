import itertools
import random
from matplotlib import pyplot as plt


# x=[a h] , 0<a<=30 , 0<h<=30


# function to get the keys with the same value from a dictionary
def get_keys(val, dictionary):
    keys = []
    for key, value in dictionary.items():
        if val == value:
            keys.append(key)
    return keys


# parameters class
class Parameters:
    pop_size = 50  # population size
    max_gen = 300  # number of generations
    objectives = [lambda x: 2 * x[0] * ((x[0] / 2) ** 2 + x[1] ** 2) ** 0.5,
                  lambda x: 1 / (2 * x[0] * ((x[0] / 2) ** 2 + x[1] ** 2) ** 0.5 + x[0] ** 2)]  # objective functions
    problem_size = len(objectives)
    infinity = 10000000000
    min_volume = 3000  # minimum value of volume
    p_crossover = 0.9
    p_mutation = 0.05
    upper_limit = 30  # upper limit for a and h
    number_of_sets = 1  # number of solution sets to be compared on the same plot


# define the constraint for the specific problem
def constraint(x):
    return x[0] ** 2 * x[1] > Parameters.min_volume * 3


# create one chromosome with respect to the constraint
def create_one_chromosome():
    x = [0, 0]
    while not constraint(x):
        a, h = random.random() * Parameters.upper_limit, random.random() * Parameters.upper_limit
        x[0] = a
        x[1] = h
    return x


# generate the population
def generate_population(population_size):
    return [create_one_chromosome() for _ in range(population_size)]


# returns the indexes list based on the descending order of chromosomes objective values in a front
# objective_values=[fi(x1), fi(x2), ... ,fi(xn)]
# list of indexes will be the indexes of elements in one front
def sort_indexes(front, objective_values):
    values_for_indexes = {key: val for key, val in enumerate(objective_values) if key in front}
    return list(
        itertools.chain(*[get_keys(val, values_for_indexes) for val in set(sorted(values_for_indexes.values()))]))[::-1]


# if function returns true, then p dominates q (we want to minimize all objectives)
# objective_values = [[f1(x1), f1(x2), ... ,f1(xn)],[f2(x1), f2(x2), ... ,f2(xn)],...,[fm(x1), fm(x2), ... ,fm(xn)]]
# first_condition -  all should be <=
# second_condition - at least one should be <
# p,q - index of chromosomes in range (1..n)
def dominate(p, q, objective_values):
    first_condition = 0
    second_condition = 0
    for i in range(len(objective_values)):
        if objective_values[i][p] <= objective_values[i][q]:
            first_condition += 1
        if objective_values[i][p] < objective_values[i][q]:
            second_condition += 1
    return True if (first_condition == len(objective_values) and second_condition >= 1) else False


def fast_non_dominated_sort_algorithm(objective_values):
    population_size = len(objective_values[0])
    S = [[] for _ in range(population_size)]
    n = [0 for _ in range(population_size)]
    rank = [0 for _ in range(population_size)]
    F = [[]]
    for p in range(population_size):
        for q in range(population_size):
            if dominate(p, q, objective_values):
                if q not in S[p]:
                    S[p].append(q)
            elif dominate(q, p, objective_values):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            F[0].append(p)
    i = 0
    while len(F[i]) != 0:
        Q = []
        for p in F[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i = i + 1
        F.append(Q)
    del (F[-1])
    return F


# I is a list of lists: each list will be the sorted indexes for the specific front
def crowding_distance(front, objective_values):
    front_size = len(front)
    number_of_objectives = len(objective_values)
    distances = [[f, 0] for f in front]
    I = [sort_indexes(front, objective_values[i]) for i in range(number_of_objectives)]
    for i in range(number_of_objectives):
        distances = sorted(distances, key=lambda pair: I[i].index(pair[0]))
        distances[0][1] = distances[-1][1] = Parameters.infinity
        for j in range(1, front_size - 1):
            distances[j][1] = distances[j][1] + (
                    objective_values[i][I[i][j + 1]] - objective_values[i][I[i][j - 1]]) / (
                                      max(objective_values[i]) - min(objective_values[i]))
    return [pair[0] for pair in sorted(distances, key=lambda pair: pair[1])[::-1]]


def crossover(mother, father, p_crossover):
    child = [0, 0]
    while not constraint(child):
        r = random.random()
        if r < p_crossover:
            a = random.random()
            child = [mother[i] * a + father[i] * (1 - a) for i in range(len(mother))]
        else:
            child = mother
    return child


def mutation(child, p_mutation):
    while not constraint(child):
        for i in range(len(child)):
            r = random.random()
            if r < p_mutation:
                child[i] = random.random()*Parameters.upper_limit
    return child


# tournament selection with k=2
def tournament_selection(population_size, objective_values, fronts):
    index_c1 = random.randint(0, population_size - 1)
    index_c2 = random.randint(0, population_size - 1)
    front_c1 = 0
    front_c2 = 0
    for i in range(len(fronts)):
        if index_c1 in fronts[i]:
            front_c1 = i
        if index_c2 in fronts[i]:
            front_c2 = i
    if front_c1 < front_c2:
        return index_c1
    elif front_c2 < front_c1:
        return index_c2
    else:
        indexes = crowding_distance(fronts[front_c1], objective_values)
        c1_loc = indexes.index(index_c1)
        c2_loc = indexes.index(index_c2)
    return index_c1 if c1_loc <= c2_loc else index_c2


def nsga2_algorithm(pop_size=Parameters.pop_size, p_crossover=Parameters.p_crossover,
                    p_mutation=Parameters.p_mutation):
    gen_no = 0
    solution = generate_population(pop_size)
    objective_values = [[Parameters.objectives[i](x) for x in solution] for i in range(Parameters.problem_size)]
    while gen_no < Parameters.max_gen:
        objective_values = [[Parameters.objectives[i](x) for x in solution] for i in range(Parameters.problem_size)]
        fronts = fast_non_dominated_sort_algorithm(objective_values)
        print(f"Best front for generation number {gen_no + 1} : {[solution[i] for i in fronts[0]]}")
        auxiliary_solution = []
        while len(auxiliary_solution) != pop_size:
            mother = solution[tournament_selection(pop_size, objective_values, fronts)]
            father = solution[tournament_selection(pop_size, objective_values, fronts)]
            child = crossover(mother, father, p_crossover)
            child = mutation(child, p_mutation)
            auxiliary_solution.append(child)
        solution = solution + auxiliary_solution
        objective_values2 = [[Parameters.objectives[i](x) for x in solution] for i in range(Parameters.problem_size)]
        fronts2 = fast_non_dominated_sort_algorithm(objective_values2)
        new_solution_indexes = []
        for i in range(len(fronts2)):
            if len(new_solution_indexes) + len(fronts2[i]) <= pop_size:
                new_solution_indexes.extend(fronts2[i])
            else:
                number_of_chromosomes_from_i_front = pop_size - len(new_solution_indexes)
                crowding_distances_indexes = crowding_distance(fronts2[i], objective_values2)
                new_solution_indexes.extend(crowding_distances_indexes[:number_of_chromosomes_from_i_front])
        solution = [solution[i] for i in new_solution_indexes]
        gen_no += 1
    return objective_values


def plot_solution(objective_values_list):
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(15, 8))
    plt.title("NSGA-II")
    plt.xlabel('Aria laterala', fontsize=15)
    plt.ylabel('Aria totala', fontsize=15)
    for s in range(Parameters.number_of_sets):
        values1 = [j for j in objective_values_list[s][0]]
        values2 = [1 / j for j in objective_values_list[s][1]]
        plt.scatter(values1, values2, c=colors[s], marker='o', s=50, alpha=0.7)
    plt.show()


if __name__ == '__main__':
    objective_values_list = []
    for i in range(Parameters.number_of_sets):
        objective_values_list.append(nsga2_algorithm())
    print(objective_values_list)
    plot_solution(objective_values_list)
