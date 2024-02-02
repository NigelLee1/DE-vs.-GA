import random

def generate_vector(bounds):
    dim = len(bounds)
    return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)]

def generate_population_de(n_pop, bounds):
    return [generate_vector(bounds) for _ in range(n_pop)]

def mutation_de(population, F):
    U = []
    n_pop = len(population)
    dim = len(population[0])
    for i in range(n_pop):
        si = []
        individual = population[i]
        temp = population.copy()
        temp.remove(individual)
        D = []
        for j in range(3):
            a = random.choice(temp)
            si.append(a)
            temp.remove(a)
        for d in range(dim):
            D.append(si[0][d] + F*(si[1][d] - si[2][d]))
        U.append(D)
    return U

def crossover_de(population, U, r_cross):
    V = []
    n_pop = len(population)
    dim = len(population[0])
    indexs = [i for i in range(dim)]
    for i in range(n_pop):
        randj = [random.random() for i in range(dim)]
        rj = random.choice(indexs)
        V.append([])
        for j in range(dim):
            if randj[j] <= r_cross or j == rj:
                V[i].append(U[i][j])
            else:
                V[i].append(population[i][j])
    return V

def selection_de(objective,population, V):
    selected = []
    n_pop = len(population)
    for i in range(n_pop):
        if objective(V[i]) < objective(population[i]):
            selected.append(V[i])
        else:
            selected.append(population[i])
    return  selected

def run_de(objective, bounds, n_pop, r_cross, F, n_iter):
    population = generate_population_de(n_pop,bounds)
    # keep track of evolution process
    evolution_scores = []
    evolution_population = []
    for i in range(n_iter):
        scores = [objective(vector) for vector in population]
        min_score = scores[0]
        min_j = 0
        for j in range(len(population)):
            if scores[j] <= min_score:
                min_score = scores[j]
                min_j = j
        evolution_scores.append(min_score)
        evolution_population.append(population[min_j])
        U = mutation_de(population, F)
        V = crossover_de(population, U, r_cross)
        population = selection_de(objective,population,V)

    return evolution_scores, evolution_population