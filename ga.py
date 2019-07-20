"""
A simple implementation of genetic algorithm.
"""
import numpy as np
from logger import global_logger, wrap_with_log
def _no_duplicates(value, list,epsilon=0.001):
    for l in list:
        if abs(value - l)<epsilon:
            return False
    return True

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

    def __init__(self, fit, mu, cr, p_cr, p_elite, pop_size=None):
        super(GA, self).__init__()
        self.fit = fit
        self.mu = mu
        self.cr = cr
        self.p_elite = p_elite
        self.p_cr = p_cr
        self.fitness = []
        self.log = global_logger
        self.pop_size = pop_size

    @wrap_with_log
    def run(self, pop, iter, good_size = 10):
        """Runs the genetic algorithm and returns the results.

        Args:
            pop(list): initial population
            iter(int): number of iterations
            good_size: the number of graphs considered "good"
            pop_size: the maximal size of the population (We allow mutations to create more individuals.)
        Returns:
            a list of good individuals found throughout the search
        """
        if self.pop_size is None:
            self.n = n = len(pop)
        else:
            self.n = n = self.pop_size
        elites = int(n * self.p_elite)
        self.pop = [i.copy() for i in pop]
        good = []
        gt = 0 #the good threshold. This will be the smallest fitnees in the set good.
        for i in range(1, iter + 1):
            if i % 10 == 1:
                self.log("Iteration " + str(i) + "/" + str(iter) + " ...")
            # 1. Selection
            self._select()
            # save the good ones
            for j in range(n):
                if ((len(good)<good_size or self.fitness[j] > gt)
                                and
                   _no_duplicates(self.fitness[j],[g[1] for g in good])):
                    good.append((self.pop[j].copy(), self.fitness[j]))
                    self.log(
                        f"Found a good individual with fitness :{self.fitness[j]:0.4f}")
                    good.sort(key = lambda g: g[1], reverse = True)#sort by fitness
                    if len(good)>good_size:
                        good = good[:-1]
                        gt = good[-1][1]
                # else:
                #     break

            # 2. generate the new population
            # 2.1 Elitisism
            new_pop = []
            new_pop.extend([x.copy() for x in self.pop[:elites]])
            additional_pop = []
            # 2.2 use cross over and mutation to generate the remaining individuals
            for j in range(n - elites):
                r = np.random.rand()

                if r < self.p_cr:
                    # 2.2.1 cross over
                    ind1 = np.random.randint(0, n)
                    ind2 = np.random.randint(0, n)
                    while (ind1 == ind2):
                        ind2 = np.random.randint(0, n)
                    additional_pop.append(self.cr(self.pop[ind1], self.pop[ind2]))
                else:
                    # 2.2.2 Mutation
                    ind = np.random.randint(0, n)
                    # if iter == 1:
                    #     print(i)
                    #     print(j)
                    #     print(n-elites)
                    #     print(n)
                    additional_pop.append(self.mu(self.pop[ind]))

            # 3. Update the population
            if type(additional_pop[0]) is list:
                additional_pop = [item for sublist in additional_pop for item in sublist]
            new_pop.extend(additional_pop)
            self.pop = new_pop
        #TODO: {good} should be updated before the final return.
        return good

    @wrap_with_log
    def _select(self):
        """
        Samples the population according to their fitness values then updates
        the population and fitness lists and sorts them using their fitness
        values.
        """
        self.fitness = []
        for i in range(len(self.pop)):
            f = self.fit(self.pop[i])
            self.fitness.append(f)

        # roulette wheel selection
        vals = np.array(self.fitness)
        vals = np.exp(vals)
        cdf = np.cumsum(vals)
        cdf = cdf / cdf[-1]
        new_pop = []
        new_fit = []
        for i in range(self.n):
            r = np.random.rand()
            sample = sum(r > cdf)
            new_pop.append(self.pop[sample].copy())
            new_fit.append(self.fitness[sample])

        # Sort by fitness decreasing
        idx = np.argsort(new_fit)[::-1]
        self.pop = [new_pop[i].copy() for i in idx]
        self.fitness = [new_fit[i] for i in idx]
