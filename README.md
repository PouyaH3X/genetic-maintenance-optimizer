Genetic Algorithm for Maintenance Scheduling Optimization

This project implements a Genetic Algorithm (GA) to solve a maintenance scheduling problem for multiple units over discrete time intervals. The goal is to optimize two conflicting objectives simultaneously:

    Maximize the minimum net reserve capacity over all intervals (F1)

    Minimize the total maintenance cost across all units and intervals (F2)

Problem Description

Given a set of units, each with associated maintenance costs and capacity reductions during maintenance, this algorithm schedules maintenance intervals for each unit while considering:

    The number of intervals each unit requires for maintenance (some require 2 intervals).

    Demand constraints and total capacity available.

    Ensuring valid schedules (only one maintenance interval per unit).

The problem is solved as a multi-objective optimization using NSGA-II (a popular evolutionary algorithm for multi-objective problems).
Features

    Uses DEAP library for the evolutionary algorithm framework.

    Implements custom crossover and mutation to respect problem constraints.

    Evaluates individuals based on net reserve capacity and maintenance cost.

    Tracks and plots fitness evolution over generations for both objectives.

    Provides a weighted sum method to select the best compromise solution from the Pareto front.

Requirements

    Python 3.x

    deap library (pip install deap)

    matplotlib for plotting (pip install matplotlib)

Usage

    Clone the repository or copy the script.

    Run the script with Python.

    Observe the plotted fitness over generations and the best found solution printed in the console.

Code Structure

    ga_problem class: Encapsulates the GA problem setup, fitness evaluation, genetic operators, and execution.

    main() method: Runs the GA for a specified number of generations and population size.

    plot_fitness_evolution() function: Plots the best, average, and worst fitness values for both objectives.

    weighted() function: Prints the best solution based on a weighted sum of objectives.

Example Output

---------------------
Best solution from weighted sum approach: [[0, 1, 0, 0], [1, 0, 0, 0], ...]
Objectives (F1, F2): (365, 850)
Score: 245

Fitness evolution graphs will also be displayed showing convergence trends.
Notes

    The code currently uses fixed example data for unit costs, capacities, demands, and maintenance requirements.

    You can customize these parameters inside the ga_problem class constructor.

    Mutation probability is low (0.1%) to preserve solution stability.

    The crossover probability is 30%.

License

This project is open-source and free to use under the MIT License.
