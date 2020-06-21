import numpy as np

    
def perturb_individual(x0, perturb_type, perturb):
    assert perturb_type in ['relative', 'absolute']
    if perturb_type == 'absolute':
        x = x0 + np.random.uniform(-perturb, perturb, size=len(x0))
    else:
        x = x0 * (1 + np.random.uniform(-perturb, perturb, size=len(x0)))
    return x

def generate_first_population(x0, population_size, perturb_type, perturb):
    population = []
    i = 0
    while i < population_size:
        population.append(perturb_individual(x0, perturb_type, perturb))
        i += 1
    return np.array(population)

def evaluate_population(population, func):
    performance = []
    for individual in population:
        perf = func(individual)
        performance.append(perf)
    return np.array(performance)

def select_best(population, n_best, func):
    performance = evaluate_population(population, func)
    performance[np.isnan(performance)] = -np.inf
    ranking = np.flip(np.argsort(performance), axis=0)
    ranked_population = population[ranking]
    return ranked_population[:n_best], np.max(performance)

def select_random(population, n_random):
    ind = range(population.shape[0])
    return population[np.random.choice(ind, size=n_random)]

def select(population, n_best, n_random, func):
    best, best_perf = select_best(population, n_best, func)
    best_individual = best[0]
    random = select_random(population, n_random)
    selected = np.vstack((best, random))
    np.random.shuffle(selected)
    return selected, best_perf, best_individual

def create_child(individual1, individual2):
    # Crossover
    child = np.zeros_like(individual1)
    ind = np.random.binomial(1, 0.5, size=len(child))
    ind = ind.astype(bool)
    child[ind] = individual1[ind]
    child[np.logical_not(ind)] = individual2[np.logical_not(ind)]
    return child

def create_children(breeders, n_children):
    next_population = []
    for i in range(int(breeders.shape[0]/2)):
        for j in range(n_children):
            next_population.append(create_child(breeders[i], breeders[-i-1]))
    return np.array(next_population)

def mutate_individual(individual, perturb_type, perturb):
    mutate_idx = np.random.choice(len(individual))
    if perturb_type == 'absolute':
        if np.isscalar(perturb):
            individual[mutate_idx] += np.random.uniform(-perturb, perturb)
        else:
            individual[mutate_idx] += np.random.uniform(-perturb[mutate_idx], perturb[mutate_idx])
    else:
        if np.isscalar(perturb):
            individual[mutate_idx] *= (1 + np.random.uniform(-perturb, perturb))
        else:
            individual[mutate_idx] *= (1 + np.random.uniform(-perturb[mutate_idx], perturb[mutate_idx]))
    return individual
	
def mutate_population(population, chance_of_mutation, perturb_type, perturb):
    for i, individual in enumerate(population):
        if np.random.rand(1) < chance_of_mutation:
            population[i] = mutate_individual(individual, perturb_type, perturb)
    return population

def optimize(func, perturb_type, perturb, n_eval):
    # Optimize using GA
    n_best = 30
    n_random = 10
    n_children = 5
    chance_of_mutation = 0.1
    population_size = int((n_best+n_random)/2*n_children)
    population = generate_first_population(func.alpha0, population_size, perturb_type, perturb)
    best_inds = []
    best_perfs = []
    opt_perfs = [0]
    i = 0
    while 1:
        breeders, best_perf, best_individual = select(population, n_best, n_random, func)
        best_inds.append(best_individual)
        best_perfs.append(best_perf)
        opt_perfs += [np.max(best_perfs)] * population_size # Best performance so far
        print('%d: fittest %.2f' % (i+1, best_perf))
        # No need to create next generation for the last generation
        if i < n_eval/population_size-1:
            next_generation = create_children(breeders, n_children)
            population = mutate_population(next_generation, chance_of_mutation, perturb_type, perturb)
            i += 1
        else:
            break
    
    opt_x = best_inds[np.argmax(best_perfs)]
    opt_airfoil = func.synthesize(opt_x)
    print('Optimal CL/CD: {}'.format(opt_perfs[-1]))
    
    return opt_x, opt_airfoil, opt_perfs

