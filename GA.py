from numpy.random import randint
from numpy.random import rand

def generate_bitstring(length):
    return randint(0,2,length).tolist()

def generate_population_ga(n_pop, bitstring_length):
    return [generate_bitstring(bitstring_length) for _ in range(n_pop)]

# tournament selection
def selection_ga(population, fitness_func, k=3):
    # first random selection
    selection_ix = randint(len(population))
    for ix in randint(0, len(population), k-1):
        # check if better (e.g. perform a tournament)
        if fitness_func[ix] < fitness_func[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

# crossover two parents to create two children
def crossover_ga(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# mutation operator
def mutation_ga(bitstring, probability = 0.5):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < probability:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded


def run_ga(objective, bounds, n_bits=8, n_pop=10, r_cross=0.9, r_mut=0.01, n_iter=31):
    dim = len(bounds)
    population = generate_population_ga(n_pop, dim * n_bits)
    # keep track of best solution
    # best, best_eval = 0, objective(decode(bounds, n_bits, population[0]))

    # keep track of best solution
    evolution_scores = []
    evolution_population = []
    # enumerate generations
    for gen in range(n_iter):
        # decode population
        decoded = [decode(bounds, n_bits, p) for p in population]
        # evaluate all candidates in the population
        scores = [objective(d) for d in decoded]
        min_score = scores[0]
        min_j = 0
        for j in range(len(population)):
            if scores[j] <= min_score:
                min_score = scores[j]
                min_j = j

        # evolution_scores(min(scores))
        evolution_scores.append(min_score)
        evolution_population.append(decoded[min_j])
        # select parents
        selected = [selection_ga(population, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover_ga(p1, p2, r_cross):
                # mutation
                mutation_ga(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        population = children
    return evolution_scores, evolution_population