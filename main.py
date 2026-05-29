import json
from population import Population
from evaluator import evaluate_population

with open("config.json", "r") as file:
    config = json.load(file)

if __name__ == "__main__":

    # 1. Spawn a brand new Gen 0
    pop = Population(config)
    #pop = Population.load_json("xor_trained_backup.json", config)
    
    print("Starting Evolution...")
    
    # 2. Run for up to 100 generations
    winner = pop.run(evaluate_population, generations=100)
    
    print(f"\nTraining Complete! Best Fitness: {winner.fitness:.3f} / 4.0")
    
    # 3. Save the resulting population so you don't lose the trained brains!
    pop.save_json("xor_trained_backup.json")