import json
from population import Population
from phenotype import NeuralNetwork
from evaluator import evaluate_population

with open("config.json", "r") as file:
    config = json.load(file)

if __name__ == "__main__":
    pop = Population(config)
    print(f"Spawned Gen 0 with {len(pop.genomes)} brains.")
    
    evaluate_population(pop.genomes, config)
        
    best_genome = max(pop.genomes, key=lambda g: g.fitness)
    print(f"Best Gen 0 Fitness: {best_genome.fitness:.3f} / 4.0")