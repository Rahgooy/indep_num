"""
A simple implementation of genetic algorithm.
"""
import numpy as np


class GA(object):
    """A generic class which provides the basic functions
    and data structures for GA.

    Args:
        fit: fitness function, a function that takes an individual and
        returns a real value indicating the fitness of that individual.

        mu: mutation function, a function that takes an individual peforms
        a mutation on it and returns the new individual

        cr: cross over function, a function that takes two individuals
        performs cross over and returns the new childs

        p_elite: the proportion of elites

        p_cr: proportion of cross overs, 1-p_cr will be the
        proportion of mutations
    """

    def __init__(self, fit, mu, cr, p_cr, p_elite, log_func=print):
        super(GA, self).__init__()
        self.fit = fit
        self.mu = mu
        self.cr = cr
        self.p_elite = p_elite
        self.p_cr = p_cr
        self.fitness = []
        self.log = log_func

    def run(self, pop, iter, gt):
        """Runs the genetic algorithm and returns the results.

        Args:
            pop(list): initial population
            iter(int): number of iterations
            gt: good individual threshold. The threshold of fitness where an
            individual considered to be good enough.

        Returns:
            a list of good individuals found throughout the search
        """
        self.n = n = len(pop)
        elites = int(n * self.p_elite)
        self.pop = [i for i in pop]
        good = []
        for i in range(1, iter+1):
            self.log(f"Iteration {i} started ...")
            # 1. Selection
            self._select()
            # save the good ones
            for j in range(n):
                if self.fitness[j] >= gt:
                    good.append((self.pop[j], self.fitness[j]))
                    self.log(f"Found a good individual with fitness : {self.fitness[j]}")
                else:
                    break

            # 2. generate the new population
            # 2.1 Elitisism
            new_pop = []
            new_pop.extend(self.pop[:elites])

            # 2.2 use cross over and mutation to generate the remaining individuals
            for j in range(n - elites):
                r = np.random.rand()

                if r < self.p_cr:
                    # 2.2.1 cross over
                    ind1 = np.random.randint(0, n)
                    ind2 = np.random.randint(0, n)
                    while(ind1 == ind2):
                        ind2 = np.random.randint(0, n)
                    new_pop.append(self.cr(self.pop[ind1], self.pop[ind2]))
                else:
                    # 2.2.2 Mutation
                    ind = np.random.randint(0, n)
                    new_pop.append(self.mu(self.pop[ind]))

            # 3. Update the population
            self.pop = new_pop

        return good

    def _select(self):
        """
        Samples the population according to their fitness values then updates
        the population and fitness lists and sorts them using their fitness
        values.
        """
        self.fitness = []
        for i in range(self.n):
            self.fitness.append(self.fit(self.pop[i]))

        # roulette wheel selection
        cdf = np.cumsum(self.fitness)
        cdf = cdf / cdf[-1]
        new_pop = []
        new_fit = []
        for i in range(self.n):
            r = np.random.rand()
            sample = sum(r > cdf)
            new_pop.append(self.pop[sample])
            new_fit.append(self.fitness[sample])

        self.pop = new_pop
        self.fitness = new_fit

        # Sort by fitness decreasing
        idx = np.argsort(self.fitness)[::-1]
        self.pop = [self.pop[i] for i in idx]
        self.fitness = [self.fitness[i] for i in idx]
