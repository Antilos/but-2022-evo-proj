import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygad

import os, random, time
import itertools
import csv, json
import logging

from graph import Graph


def random_graph(vs : int, edge_prob : float, random_state : int = 42) -> Graph:
    random.seed(random_state)
    edges = []
    for v1, v2 in itertools.product(range(vs), repeat=2):
        if v1 == v2:
            continue
        if random.random() < edge_prob:
            edges.append((v1, v2))

    g = Graph(vs, edges)
    return g

last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    change = ga_instance.best_solution()[1] - last_fitness
    # print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    if change > 0:
        logging.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
        logging.info("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
        logging.info("Change     = {change}".format(change=change))
    last_fitness = ga_instance.best_solution()[1]

def callback_stop(ga_instance, fitnesses):
    global last_fitness
    last_fitness = 0

def run_experiments(graph:Graph, experiment_name:str, args:list) -> None:
    for arg_i, arg in enumerate(args):
        logging.info(f"Experiment {arg_i}")
        ga_args = {
            "num_generations":arg["num_generations"],
            "num_parents_mating":arg["num_parents_mating"], 
            "fitness_func":arg["fitness_func"],
            "crossover_type":arg["crossover_type"],
            "mutation_type":arg["mutation_type"],
            "mutation_probability":arg["mutation_probability"],
            "sol_per_pop":arg["sol_per_pop"], 
            "num_genes":graph.num_vertices,
            "gene_type":int,
            "gene_space":range(0, 5),
            "on_generation":callback_generation,
            "on_stop":callback_stop,
            "stop_criteria":arg["stop_criteria"]
        }
        logging.info(f"Params:\n{ga_args}")

        EXP_T1 = time.time()
        for run in range(arg["runs"]):
            logging.info(f"Exp {arg_i}: run {run}")
            T1 = time.time()
            ga_instance = pygad.GA(**ga_args)
            ga_instance.run()
            T2 = time.time()

            best_solutions = []
            best_solutions_fitnesses = []
            fitness_developments_raw = []
            best_solution_generations = []
        
            solution, solution_fitness, solution_idx = ga_instance.best_solution()

            best_solutions.append(solution)
            best_solutions_fitnesses.append(solution_fitness)
            # print(len(ga_instance.best_solutions_fitness))
            # print(ga_instance.best_solutions_fitness)
            fitness_developments_raw.append(ga_instance.best_solutions_fitness)
            best_solution_generations.append(ga_instance.best_solution_generation)

            print(f"{run=} of version={arg_i} T={T2-T1:.2f} | Best Fitness={solution_fitness}")

        EXP_T2 = time.time()
        logging.info(f"Exp {arg_i} ({arg['runs']} runs) took {EXP_T2-EXP_T1:.2f} seconds")
        # Process data

        #pad fitnesses in runs that ended early
        from copy import deepcopy
        fitness_developments = deepcopy(fitness_developments_raw)
        for i, x in enumerate(fitness_developments):
            if (x_len := len(x)) < arg["num_generations"] + 1:
                x = x + [x[-1]] * (arg["num_generations"] + 1 - x_len)
                fitness_developments[i] = x

        def update_graph_count_conflicts (g, sol):
            g.set_colors(sol)
            return len(g.get_vertices_in_conflict())

        df = pd.DataFrame(
            {
                "generation":best_solution_generations,
                "fitness": best_solutions_fitnesses,
                "conflicts": [update_graph_count_conflicts(graph, solution) for solution in best_solutions]
            }
        )

        df["conflict_ratio"] = df["conflicts"] / graph.num_vertices
        df["solution_found"] = df["fitness"] == 1
        df["solution_not_in_init_pop"] = df["generation"] != 1

        # Save data
        experiment_dir = f"./experiments/{experiment_name}"
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)

        params_fout_name = f"{experiment_name}_{arg_i}_parameters.json"
        results_fout_name = f"{experiment_name}_{arg_i}_results.txt"
        df_fout_name = f"{experiment_name}_{arg_i}_data.csv"
        fit_fout_name = f"{experiment_name}_{arg_i}_fitness"
        fit_plot_fout_name = f"{experiment_name}_{arg_i}_mean_fit.png"

        with open(os.path.join(experiment_dir, params_fout_name), 'w') as fout:
            serializable_args = {k : str(v) for k, v in arg.items()}
            fout.write(json.dumps(serializable_args))
        with open(os.path.join(experiment_dir, results_fout_name), 'w') as fout:
            solution_count = df[df["solution_found"]].shape[0]
            fout.write(f"Solution found in {solution_count/arg['runs'] * 100}% runs\n")
            fout.write(f"Conflicts: mean={df['conflicts'].mean()}, std={df['conflicts'].std()}, mean ratio to vertices={df['conflict_ratio'].mean()}\n")
        with open(os.path.join(experiment_dir, df_fout_name), 'w') as fout:
            df.to_csv(fout)
        with open(os.path.join(experiment_dir, fit_fout_name), 'w') as fout:
            wr = csv.writer(fout)
            wr.writerows(fitness_developments)

        # Plots
        fig, ax = plt.subplots()

        fitness_developments = np.array(fitness_developments)
        ax.set(title='Mean fitness through runs', xlabel='Generation', ylabel='Fitness')
        _ = ax.plot(range(1, arg['num_generations']+1+1), fitness_developments.T.mean(axis=-1))
        
        fig.savefig(os.path.join(experiment_dir, fit_plot_fout_name))

if __name__ == '__main__':
    pass
