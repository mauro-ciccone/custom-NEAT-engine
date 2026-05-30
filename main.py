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

    load_backup = False

    executor = concurrent.futures.ProcessPoolExecutor(initializer=init_worker)

    if not load_backup:
        pop = Population(config)
    else: 
        pop = Population.load_json(f"{config['environment']}_trained_backup.json", config)
    
    print("Starting Evolution...")
    
    try:
        winner = pop.run(lambda genomes, config: evaluate_population(genomes, config, executor), generations=201)
        print(f"\nTraining Complete! Best Fitness: {winner.fitness:.3f} / 100.0")
        
    except KeyboardInterrupt:
        print("\n\nTraining Interrupted by User!")
        print("Grabbing the best genome so far...")
        winner = max(pop.genomes, key=lambda g: g.fitness)
        print(f"Current Best Fitness: {winner.fitness:.3f} / 100.0")
        
    print("Saving population state to disk...")
    pop.save_json(f"{config['environment']}_trained_backup.json")
    print("Safely exited.")