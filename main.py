import json
from population import Population
from evaluator import evaluate_population

with open("config.json", "r") as file:
    config = json.load(file)

if __name__ == "__main__":

    #pop = Population(config)
    pop = Population.load_json("xor_trained_backup.json", config)
    
    print("Starting Evolution...")
    
    winner = pop.run(evaluate_population, generations=100)
    
    print(f"\nTraining Complete! Best Fitness: {winner.fitness:.5f} / 4.0")
    
    pop.save_json("xor_trained_backup.json")