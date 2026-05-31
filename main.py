import json
from population import Population
from evaluator import evaluate_population
import concurrent.futures
import signal

with open("config.json", "r") as file:
    config = json.load(file)

def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == "__main__":

    load_backup = True

    executor = concurrent.futures.ProcessPoolExecutor(initializer=init_worker)

    if not load_backup:
        pop = Population(config)
    else: 
        pop = Population.load_json(f"{config['environment']}_training_backup.json", config)
    
    print(f"Starting Evolution...\n\n\n")
    
    try:
        winner = pop.run(lambda genomes, config, curr_gen: evaluate_population(genomes, config, executor, curr_gen), generations=300)
        print(f"\nTraining Complete! Best Fitness: {winner.fitness:.3f} / {config['max_fitness']}")
        
    except KeyboardInterrupt:
        print("\n\nTraining Interrupted by User!")
        print("Grabbing the best genome so far...")
        winner = max(pop.genomes, key=lambda g: g.fitness)
        print(f"Current Best Fitness: {winner.fitness:.3f} / {config['max_fitness']}")
        
    print("Saving population state to disk...")
    pop.save_json(f"{config['environment']}_training_backup.json")
    
    #print(f"\npreview of next gen:")
    #winner = pop.run(lambda genomes, config, curr_gen: evaluate_population(genomes, config, executor, curr_gen), generations=1)
    print(f"\n\n\n")
