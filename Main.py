from random import randint, randrange, seed, choice, random
from deap import base, creator, tools
import matplotlib.pyplot as plt


class ga_problem:
    def __init__(self):
        # Number of units
        self.nou = 7

        # The cost of each unit in 4 intervals [1,2,3,4]
        self.data_cost = {
            1: [100, 120, 110, 130],
            2: [90, 95, 100, 105],
            3: [80, 85, 90, 95],
            4: [150, 160, 155, 165],
            5: [70, 75, 80, 85],
            6: [60, 65, 70, 75],
            7: [50, 55, 60, 65],
        }
        # The capacity of each unit in 4 intervals [1,2,3,4]
        self.data_cap = {
            1: [130, 150, 100, 110],
            2: [80, 95, 110, 105],
            3: [50, 30, 120, 100],
            4: [130, 200, 170, 150],
            5: [90, 80, 70, 60],
            6: [40, 30, 50, 55],
            7: [60, 70, 60, 65],
        }
        # Demand for 4 intervals
        self.demand = [400, 350, 390, 430]

        # Total available capacity per interval (summing over units)
        self.capacity = []
        for i in range(4):
            tmp = 0
            for j in range(1, self.nou + 1):
                tmp += self.data_cap[j][i]
            self.capacity.append(tmp)

        # The number of intervals each unit needs for maintenance
        self.i_n = [2, 2, 1, 1, 1, 1, 1]

        # Number of generations
        self.iter = 100
        # Population size for the GA
        self.pop_size = 100

        # Validate input dimensions
        if not self.main_validate():
            print("Please control inputs!")
            return

        self.create_fitness()

    def main_validate(self):
        if len(self.data_cost) == len(self.data_cap) == self.nou == len(self.i_n) and len(self.capacity) == 4:
            return True
        return False

    def create_fitness(self):
        # Fitness: maximize minimum net reserve (F1), minimize total maintenance cost (F2)
        creator.create("FitMulti", base.Fitness, weights=(+1.0, -1.0))
        creator.create("ind", list, fitness=creator.FitMulti)

        self.tool = base.Toolbox()

        self.tool.register("ind", self.create_ind)
        self.tool.register("population", tools.initRepeat, list, self.tool.ind)
        self.tool.register("evaluate", self.evalMaintenance)
        self.tool.register("select", tools.selNSGA2)
        self.tool.register("corssovr", self.crossover)

        # Mutation function
        def mutate(ind):
            if random() > 0.001:
                return ind
            unit_index = randrange(len(ind))
            bit_index = randrange(len(ind[unit_index]))
            ind[unit_index][bit_index] = 1 - ind[unit_index][bit_index]

            # Enforce exactly one 1 per unit
            if ind[unit_index].count(1) != 1:
                ind[unit_index] = [0] * 4
                ind[unit_index][randint(0, 3)] = 1
            # Ensure units that need two intervals for maintenance do not get interval 4
            if self.i_n[unit_index] == 2 and ind[unit_index][3] == 1:
                ind[unit_index][3] = 0
                available_indices = [i for i in range(
                    3) if ind[unit_index][i] == 0]
                if available_indices:
                    ind[unit_index][choice(available_indices)] = 1

            return ind

        self.tool.register("mutate", mutate)

    def create_ind(self):
        ind = []
        for i in range(self.nou):
            gen = [0] * 4
            if self.i_n[i] == 2:
                gen[randint(0, 2)] = 1
            else:
                gen[randint(0, 3)] = 1
            ind.append(gen)
        return creator.ind(ind)

    def evalMaintenance(self, ind):
        # F1: Net reserve , F2: Total cost
        F1 = []
        F2 = 0

        for i in range(4):
            tmp = 0

            for j, gen in enumerate(ind):

                # If gene[j] == 1, maintenance is scheduled so subtract capacity cost
                if gen[i] == 1:
                    F2 += self.data_cost[j+1][i]
                else:
                    tmp += self.data_cap[j+1][i]

                if j > 0 and self.i_n[j] == 2 and gen[i-1] == 1:
                    tmp -= self.data_cap[j+1][i]
            F1.append(tmp)

        F1 = min(F1)
        return F1, F2

    def crossover(self, ind1, ind2):
        if random() > 0.7:
            return

        c_p = randint(1, len(ind1) - 1)
        ind1[:c_p], ind2[:c_p] = ind2[:c_p], ind1[:c_p]
        # Validate each gene in individuals
        for i in range(len(ind1)):
            if ind1[i].count(1) != 1:
                ind1[i] = [0] * 4
                ind1[i][randint(0, 3)] = 1

            if ind2[i].count(1) != 1:
                ind2[i] = [0] * 4
                ind2[i][randint(0, 3)] = 1

        del ind1.fitness.values
        del ind2.fitness.values
        return ind1, ind2

    def main(self):
        tool = self.tool
        pop = tool.population(n=self.pop_size)

        # Store fitness values for f1
        best_fitness_f1 = []
        avg_fitness_f1 = []
        worst_fitness_f1 = []

        best_fitness_f2 = []
        avg_fitness_f2 = []
        worst_fitness_f2 = []

        # Evaluate the initial population
        for ind in pop:
            ind.fitness.values = tool.evaluate(ind)

        fits = [ind.fitness.values[0] for ind in pop]
        best_fitness_f1.append(max(fits))
        avg_fitness_f1.append(sum(fits) / len(fits))
        worst_fitness_f1.append(min(fits))

        fits = [ind.fitness.values[1] for ind in pop]
        best_fitness_f2.append(max(fits))
        avg_fitness_f2.append(sum(fits) / len(fits))
        worst_fitness_f2.append(min(fits))

        # Run GA for the specified number of iterations
        for _ in range(1, self.iter):
            offspring = tools.selTournament(pop, len(pop), tournsize=3)
            offspring = [tool.clone(ind) for ind in offspring]

            # Crossover
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                tool.corssovr(ind1, ind2)
                if hasattr(ind1, "fitness"):
                    del ind1.fitness.values
                if hasattr(ind2, "fitness"):
                    del ind2.fitness.values

            # Mutation
            for mutant in offspring:
                tool.mutate(mutant)
                if hasattr(mutant, "fitness"):
                    del mutant.fitness.values

            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = tool.evaluate(ind)

            # Merge population with offspring and select the next generation
            pop = tool.select(pop + offspring, self.pop_size)

            # Track fitness stats
            fits = [ind.fitness.values[0] for ind in pop]
            best_fitness_f1.append(max(fits))
            avg_fitness_f1.append(sum(fits) / len(fits))
            worst_fitness_f1.append(min(fits))

            fits = [ind.fitness.values[1] for ind in pop]
            best_fitness_f2.append(max(fits))
            avg_fitness_f2.append(sum(fits) / len(fits))
            worst_fitness_f2.append(min(fits))

        # Plot the fitness evolution over generations
        plot_fitness_evolution(best_fitness_f1, avg_fitness_f1, worst_fitness_f1,
                               best_fitness_f2, avg_fitness_f2, worst_fitness_f2)

        return pop


def plot_fitness_evolution(bf1, af1, wf1, bf2, af2, wf2):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    g1 = range(len(bf1))
    axes[0].plot(g1, bf1, label="Best Fitness",
                 color='green', linewidth=2)
    axes[0].plot(g1, af1, label="Average Fitness",
                 color='blue', linestyle='dashed')
    axes[0].plot(g1, wf1, label="Worst Fitness",
                 color='red', linestyle='dotted')
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Fitness (F1: Net Reserve)")
    axes[0].legend()

    g2 = range(len(bf2))
    axes[1].plot(g2, bf2, label="Best Fitness",
                 color='green', linewidth=2)
    axes[1].plot(g2, af2, label="Average Fitness",
                 color='blue', linestyle='dashed')
    axes[1].plot(g2, wf2, label="Worst Fitness",
                 color='red', linestyle='dotted')
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Fitness (F2: Total cost)")
    axes[1].legend()
    fig.suptitle("Fitness Plot")
    plt.show()


def weighted(p_f, w1=2, w2=1):
    a_scores = [w1 * ind.fitness.values[0] -
                w2 * ind.fitness.values[1] for ind in p_f]
    best_index = a_scores.index(max(a_scores))
    best_solution = p_f[best_index]
    print("---------------------")
    print("Best solution from weighted sum approach:", best_solution)
    print("Objectives (F1, F2):", best_solution.fitness.values)
    print("Score:", a_scores[best_index])


# Run the GA and display results
ga = ga_problem()
f_p = ga.main()

# Get the first Pareto front from the final population and display weighted results
p_f = tools.sortNondominated(f_p, k=len(f_p), first_front_only=True)[0]
weighted(p_f, w1=3, w2=1)
