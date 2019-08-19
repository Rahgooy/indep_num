"""A simple implementation of genetic algorithm."""
import numpy as np
from logger import global_logger, wrap_with_log
from caching_redis import clear_cache
from caching_redis import set_to_start_matrices, set_to_start_indep_sets
from caching_redis import calculate_fitness_in_batch
#from multiprocessing import Process
from threading import Thread
def _no_duplicates(value, list,epsilon=0.0001):
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

    def __init__(self, fit, mu, cr, p_cr, p_elite, pop_size=None, make_unique=False):
        super(GA, self).__init__()
        self.fit = fit
        self.mu = mu
        self.cr = cr
        self.p_elite = p_elite
        self.p_cr = p_cr
        self.fitness = []
        self.log = global_logger
        self.pop_size = pop_size
        self.make_unique = make_unique #mutates each element of the population once.
        self.new_graph_dictionary = {}

    @wrap_with_log
    def perform_mutations(self, start, end, results, index):
        elites = int(self.n * self.p_elite)
        new_pop = []
        #self.pop = sorted(self.pop, key = self.fit, reverse = True)
        if start ==0 and not self.make_unique:
            new_pop.extend([x.copy() for x in self.pop[:elites]])
            #new_fit.extend([x.copy() for x in self.fitness[:elites]])
        additional_pop = []
        # 2.2 use cross over and mutation to generate the remaining individuals
        for j in range(start,end):
            #print("j is ",j)
            r = np.random.rand()

            if r < self.p_cr:
                # 2.2.1 cross over
                ind1 = np.random.randint(0, len(self.pop))
                ind2 = np.random.randint(0, len(self.pop))
                while (ind1 == ind2):
                    ind2 = np.random.randint(0, len(self.pop))
                additional_pop.append(self.cr(self.pop[ind1], self.pop[ind2]))
            else:
                # 2.2.2 Mutation
                if self.make_unique:
                    #assert len(self.pop)==n
                    additional_pop.append(self.mu(self.pop[j]))
                else:
                    ind = np.random.randint(0, len(self.pop))
                    additional_pop.append(self.mu(self.pop[ind]))

        # 3. Update the population
        if len(additional_pop)==0:
            print("empty")
            return []
        if type(additional_pop[0]) is list:
            additional_pop = [item for sublist in additional_pop for item in sublist]
        new_pop.extend(additional_pop)

        results[index] = new_pop
        return new_pop


    @wrap_with_log
    def run(self, pop, iter, good_size = 10, meta_select_proc=None):
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
        self.pop = [i.copy() for i in pop]
        self.fitness = calculate_fitness_in_batch(self.pop)
        good = []
        goodfit = []
        gt = 0 #the good threshold. This will be the smallest fitness in the set good.
        for i in range(1, iter + 1):
            if i % 10 == 1:
                self.log("Iteration " + str(i) + "/" + str(iter) + " ...")
            # 1. Selection
            """["make_unique_then_select", "only_add_elites",
                "make_extra_unique", "take_best", "take_best_unique"]"""
            if not meta_select_proc is None:
                if meta_select_proc == "make_unique_then_select": #standard procedure
                    self._remove_isomorphic()
                    self._select()
                elif meta_select_proc =="only_add_elites":#adds only the elites
                    self.pop = sorted(self.pop, key=self.fit, reverse=True)[:elites]
                    self._select()
                elif meta_select_proc =="make_extra_unique":
                    self._remove_isomorphic_better()
                    self._select()
                elif meta_select_proc =="take_best":
                    p = sorted(self.pop, key=self.fit, reverse=True)
                    self.pop = p[:self.n]
                elif meta_select_proc =="take_best_unique":
                    self._remove_isomorphic()
                    self._select()
                elif meta_select_proc=="take_best_very_unique":
                    self._remove_isomorphic_better()
                    #self.pop = sorted(self.pop, key=self.fit, reverse=True)[:self.pop_size]
                    self.pop = self.pop[:self.pop_size]
                    self.fitness = self.fitness[:self.pop_size]
                    assert(len(self.pop)==len(self.fitness))
                    #self._select()
                clear_cache()
                good = []
                goodfit = []
                gt = 0
                print("computed graphs with ", self.pop[0].order(), " vertices")
                print("current pop size ", len(self.pop))
                for g in self.pop:
                    #assert g.order()==self.pop[0].order()
                    set_to_start_matrices(g)
                    set_to_start_indep_sets(g)
            else:
                #self.pop=sorted(self.pop, key= self.fit, reverse=True)
                self._select()
            # save the good ones
            for j in range(len(self.pop)):
                #assert self.pop[j].order() == self.pop[0].order()
                if ((len(good)<good_size or self.fitness[j]) > gt and _no_duplicates(self.fitness[j],goodfit)):
                    #not (self.pop[j].adjacency_matrix() in [g.adjacency_matrix() for g in good])):
                    #_no_duplicates(self.fit(self.pop[j]),[g[1] for g in good])):
                    good.append(self.pop[j].copy())
                    goodfit.append(self.fitness[j])
                    self.log(
                        f"Found a good individual with fitness :{self.fitness[j]:0.4f}")
                    #good.sort(key = self.fit, reverse = True)#sort by fitness
                    gfg = zip(good, goodfit)
                    gfg = sorted(gfg, key = lambda x: x[1], reverse = True)
                    good = [g for g, f in gfg]
                    goodfit = [f for g, f in gfg]
                    if len(good)>good_size:
                        good = good[:-1]
                        goodfit = goodfit[:-1]
                        gt = goodfit[-1]
                # else:
                #     break

            # 2. generate the new population
            # 2.1 Elitisism
            if i!= iter:
                numthreads = min(8,len(self.pop))
                results = [None]*numthreads
                if self.make_unique and len(self.pop)>=numthreads:
                    self.new_graph_dictionary={}
                    threads = [Thread(target = self.perform_mutations, args = (t*len(self.pop)//numthreads, (t+1)*len(self.pop)//numthreads, results, t)) for t in range(numthreads)]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                else:
                    self.perform_mutations(0,len(self.pop), results, 0)
                new_graphs = []
                for r in results:
                    if r is not None:
                        new_graphs.extend(r)
                self.fitness = calculate_fitness_in_batch(new_graphs)
                #assert(len(self.fitness)== len(self.pop))
                popfit = zip(new_graphs, self.fitness)
                popfit= sorted(popfit, key = lambda x: x[1], reverse = True)
                self.pop = [g for g, f in popfit]
                self.fitness = [f for g, f in popfit]
                #self.pop = sorted(new_graphs, key= self.fit, reverse=True)
        return good

    @wrap_with_log
    def _select(self):
        """
        Samples the population according to their fitness values then updates
        the population and fitness lists and sorts them using their fitness
        values.
        """
        elites = int(self.n * self.p_elite)
        elite_pop = self.pop[:elites]
        elite_fit = self.fitness[:elites]
        #self.fitness = []
        # for i in range(len(self.pop)):
        #     f = self.fit(self.pop[i])
        #     self.fitness.append(f)

        # roulette wheel selection
        vals = np.array(self.fitness)
        vals = np.exp(vals)
        cdf = np.cumsum(vals)
        cdf = cdf / cdf[-1]
        new_pop = []
        new_fit = []
        for i in range(self.n - elites):
            r = np.random.rand()
            sample = sum(r > cdf)
            new_pop.append(self.pop[sample].copy())
            new_fit.append(self.fitness[sample])

        new_pop.extend(elite_pop)
        new_fit.extend(elite_fit)
        popfit = zip(new_pop, new_fit)
        #popfit.sort(key=lambda x: x[1], reverse = True)
        popfit= sorted(popfit,key=lambda x: x[1], reverse = True)
        self.pop = [p for p, f in popfit]
        self.fitness= [f for p, f in popfit]
        #self.pop=sorted(self.pop, key=self.fit, reverse = True)
        #self.fitness = [self.fit(g) for g in self.pop]

    def _select_unique(self):
        all_individuals = self._remove_isomorphic()
        self._select()
        self._remove_isomorphic()
        remaining_individuals = [g for g in all_individuals if not g in self.pop]
        extra = self.pop_size - len(self.pop)
        if extra > 0:
            self.pop.extend(sorted(remaining_individuals, key = self.fit, reverse=True)[:extra])


    @wrap_with_log
    def _remove_isomorphic(self):
        """Removes the isomorphic copies from the population."""
        #pop = sorted(self.pop, key=self.fit)#sorts the population by fitness, increasing.
        self.pop.reverse()
        pop = self.pop
        to_remove = []
        maximum_removal = len(self.pop)-self.pop_size
        if maximum_removal <=0:
            return self.pop
        for index in range(len(pop)-1):
            if abs(self.fitness[index] - self.fitness[index+1]) <0.001:
                if pop[index].isomorphic(pop[index+1]):
                    to_remove.append(index)
                    if len(to_remove)>maximum_removal:
                        print("maximum removal")
                        break
        self.pop = [g for index, g in enumerate(pop) if index not in to_remove]
        return self.pop

    @wrap_with_log
    def _remove_isomorphic_better(self):
        #pop = sorted(self.pop, key=self.fit)#sorts the population by fitness, increasing.
        self.pop.reverse()
        self.fitness.reverse()
        pop = self.pop
        to_remove = []
        maximum_removal = len(self.pop)-self.pop_size
        # if maximum_removal <=0:
        #     return self.pop
        for index1 in range(len(pop)-1):
            for index2 in range(index1+1,len(pop)):
                if abs(self.fitness[index1] - self.fitness[index2] ) <0.001:
                    if pop[index1].isomorphic(pop[index2]):
                        to_remove.append(index1)
                        break
                else:
                    break
        self.pop = [g for index, g in enumerate(pop) if index not in to_remove]
        self.fitness = [f for index, f in enumerate(self.fitness) if index not in to_remove]
        #assert len(self.pop)>=self.pop_size
        self.pop.reverse()
        self.fitness.reverse()
        return self.pop
