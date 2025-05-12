import copy

from Environment import FroggerEnv
import random
import numpy as np
import time


class FroggerGA:
    def __init__(self, population_size, generations, mutation_rate, environment):

        self.random_crossover = False
        self.population_size = population_size
        self.population = []
        self.selection_pool = []
        self.generations = generations

        self.BP = [0.25, 0.75, 1]
        self.BW = [0.25, 0.75, 1]
        self.cross_weak = 0

        self.mutation_rate = mutation_rate
        self.mutation_random = False
        self.max_mutation = 3
        self.mutation_nr = 3
        self.crossover2_rate = 0.35

        self.parents_elite_number = 2
        self.children_elite_number = 2
        self.roulette_number = 50
        self.tour_number = 46
        self.tour_part = 2

        self.environment = environment  # Environment-ul Frogger

        self.weights = [1 / 3, 1 / 3, 1 / 3]  # Ponderi inițiale pentru selecție
        self.initialize_population()
        self.sort()


    def initialize_population(self):
        """Inițializează populația cu acțiuni random."""

        for _ in range(self.population_size):
            not_ok = True
            while not_ok is True:
                genome = np.random.randint(0, self.environment.action_space, size=200).tolist()
                individual = Genome(genome)
                individual.lane = self.fitness(individual)

                if individual.death > 9:
                    individual.fitness = individual.lane - individual.death / 200
                    not_ok = False
                    self.population.append(individual)

    def check(self, ind):
        for i in self.selection_pool:
            if i.val == ind.val:
                return False

        return True

    def sort(self):
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

    def fitness(self, genome):
        """Calculează fitness-ul unui individ rulându-l în mediu."""
        self.environment.reset()
        self.environment.gameApp.draw()
        fit = 1

        for i, action in enumerate(genome.val):
            _, reward, done, _ = self.environment.step(action)

            if reward >= 20:
                fit += 1
            elif reward == -5 and fit > 1:
                fit -= 1

            if done:
                genome.death = i
                break

        return fit

    def mutate(self):
        """Aplică mutație aleatorie pe genom."""
        for ind in reversed(self.population):
            if random.random() < self.mutation_rate:
                nr = random.randint(1, self.max_mutation) if self.mutation_random is True else self.mutation_nr
                mutations = []
                if random.random() < 0.7:
                    mutations.append(ind.death)
                    nr -= 1

                while nr > 0:
                    k = random.randint(0, len(ind.val) - 1)
                    if k not in mutations:
                        mutations.append(k)
                        nr -= 1

                child = copy.deepcopy(ind)

                for i in mutations:
                    not_ok = True
                    while not_ok is True:
                        action = random.randint(0, self.environment.action_space)
                        if action != child.val[i]:
                            child.val[i] = action
                            not_ok = False

                self.selection_pool.append(child)

    def crossover1(self):

        self.cross_weak = 0

        for i, parent1 in enumerate(reversed(self.population)):
            if self.random_crossover is True:
                subpopulation = self.population
                parent2 = random.choice(subpopulation)

            elif (random.random() < self.cross_weak and i > 0) or i == self.population_size - 1:
                subpopulation = self.population[:i]
                parent2 = random.choice(subpopulation)

            else:
                subpopulation = self.population[i + 1:]
                parent2 = random.choice(subpopulation)

            # Actual crossover starts here
            k = 0

            not_ok = True
            while not_ok is True:
                if parent1.death < 10:
                    if parent1.death != 0:
                        if parent1.death - 1 <= 1:
                            k = 1
                        else:
                            k = random.randint(1, parent1.death)

                    child = Genome(parent1.val[:k])
                    child.val.extend(parent2.val[k:])

                    if child.val != parent1.val and child.val != parent2.val and self.check(child) is True:
                        not_ok = False
                        self.selection_pool.append(child)
                        self.cross_weak += 2 ** (((i + 1) / self.population_size) * 10 - 10)
                    else:
                        parent2 = random.choice(self.population)

                else:
                    k = random.random()

                    for it, val in enumerate(self.BW):
                        if k < val:
                            if it > 0:
                                a = int(self.BP[it - 1] * parent1.death)
                            else:
                                a = 0

                            b = int(self.BP[it] * parent1.death)

                            if a == 0 and parent1.death != 0:
                                a += 1
                            if b >= parent1.death:
                                b = parent1.death

                            k = random.randint(a, b)

                            child = Genome(parent1.val[:k])
                            child.val.extend(parent2.val[k:])

                            if child.val != parent1.val and child.val != parent2.val and self.check(child) is True:
                                not_ok = False
                                self.selection_pool.append(child)
                                self.cross_weak += 2 ** (((i + 1) / self.population_size) * 10 - 10)
                            else:
                                parent2 = random.choice(self.population)

    def crossover2(self):
        for index, parent1 in enumerate(self.population):
            if random.random() < self.crossover2_rate:
                not_ok = True

                while not_ok is True:
                    subpopulation = self.population[:index]
                    subpopulation.extend(self.population[index + 1:])
                    parent2 = random.choice(subpopulation)

                    k1 = random.randint(1, parent1.death - 1)
                    while k1 >= len(parent1.val) - 1:
                        k1 = random.randint(1, parent1.death - 1)
                    if parent1.death + 1 < len(parent1.val) - 1:
                        k2 = random.randint(parent1.death + 1, len(parent1.val) - 1)

                    else:
                        k2 = random.randint(k1 + 1, len(parent1.val) - 1)

                    child = Genome(parent1.val[:k1])
                    child.val.extend(parent2.val[k1: k2])
                    child.val.extend(parent1.val[k2:])

                    if self.check(child) is True and child.val != parent1.val and child.val != parent2.val:
                        self.selection_pool.append(child)
                        not_ok = False

    def combined_selection(self):
        """Selecție combinată, elitism, ruleta si turneu."""
        total_fitness = 0

        for ind in self.selection_pool:
            ind.lane = self.fitness(ind)
            ind.fitness = ind.lane - ind.death / 200
            total_fitness += ind.fitness

        total_weight = 0

        for ind in self.selection_pool:
            ind.rate = ind.fitness / total_fitness
            ind.weight = total_weight + ind.rate
            total_weight = ind.weight

        # Elitism
        self.population[self.parents_elite_number:] = []

        self.selection_pool.sort(key=lambda ind: ind.fitness, reverse=True)
        self.population.extend(self.selection_pool[:self.children_elite_number])

        aux = self.selection_pool[self.children_elite_number:]
        self.selection_pool = aux

        # Roulette

        for i in range(self.roulette_number):
            no_new_member = True
            while no_new_member is True:
                k = random.random()

                for index, ind in enumerate(self.selection_pool):

                    if k < ind.weight and ind.selected is False:
                        self.population.append(ind)
                        ind.selected = True
                        no_new_member = False
                        break


        aux_selection = [id for id in self.selection_pool if id.selected is False]
        self.selection_pool = []
        self.selection_pool = aux_selection

        # Tournament

        for i in range(self.tour_number):
            # print(i)
            k = self.tour_part
            best_fit = -1
            best_ind = -1

            counter = k
            v = []
            while counter > 0:
                t = random.randint(0, len(self.selection_pool) - 1)
                if t in v:
                    continue
                else:
                    v.append(t)
                    counter -= 1

            while k:
                t = v[k - 1]

                if self.selection_pool[t].fitness > best_fit:
                    # print("Hello")
                    best_fit = self.selection_pool[t].fitness
                    best_ind = t

                k -= 1

            self.population.append(self.selection_pool[best_ind])
            del self.selection_pool[best_ind]

    def evolve(self):
        """Rulează algoritmul genetic."""
        for generation in range(self.generations):

            self.crossover1()
            self.crossover2()
            self.mutate()

            self.combined_selection()
            self.sort()

            print(f"Generația {generation + 1}: Cel mai bun fitness: {self.population[0].fitness} Lane: {self.population[0].lane}")
            self.selection_pool = []

            if generation == self.generations - 1:
                for ind in self.population:
                    print(ind.val, "cu fitness-ul: ", ind.fitness, " a murit la pozitia: ", ind.death, "Lane ", ind.lane)

        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        return self.population[0]


class Genome:
    def __init__(self, val):
        self.val = val
        self.fitness = None
        self.lane = None
        self.death = 199
        self.rate = None
        self.weight = None
        self.selected = False


def play(player):
    frogger_env.reset()
    frogger_env.gameApp.draw()
    frogger_env.gameApp.state = "PLAYING"
    total_reward = 0
    for i, action in enumerate(player.val):
        _, reward, done, _ = frogger_env.step(action)
        frogger_env.gameApp.draw()
        total_reward += reward

        if done:
            break

        time.sleep(0.5)

    print(total_reward)


if __name__ == "__main__":
    # Inițializează environment-ul
    frogger_env = FroggerEnv()

    # Creează și rulează algoritmul genetic
    ga = FroggerGA(
        population_size=100,
        generations=100,
        mutation_rate=0.3,
        environment=frogger_env
    )

    best_player = ga.evolve()

    play(best_player)
