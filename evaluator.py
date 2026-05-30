from phenotype import NeuralNetwork
import random
import concurrent.futures

class XOREvaluator:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        self.expected = [0.0, 1.0, 1.0, 0.0]

    def evaluate_genome(self, genome) -> float:
        from phenotype import NeuralNetwork
        network = NeuralNetwork(genome, self.config)
        fitness = 4.0 

        for i in range(4):

                for _ in range(20):
                    network.feed_forward(self.inputs[i])
                
                output = network.feed_forward(self.inputs[i])[0]
                fitness -= abs(self.expected[i] - output)
        
        return max(0.0, fitness)

class CircleEvaluator:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.test_cases = []
        for i in range(10):
            for j in range(10):
                x = -1.0 + (i * (2.0 / 9.0))
                y = -1.0 + (j * (2.0 / 9.0))
                expected = 1.0 if (x**2 + y**2) < 0.49 else 0.0
                self.test_cases.append(([x, y], expected))
                
    # It takes ONE genome, and returns ONE float. That's it!
    def evaluate_genome(self, genome) -> float:
        from phenotype import NeuralNetwork
        network = NeuralNetwork(genome, self.config)
        fitness = 100.0 

        for inputs, expected in self.test_cases:
            for _ in range(4):
                network.feed_forward(inputs)
            output = network.feed_forward(inputs)[0]
            fitness -= abs(expected - output)
        
        return max(0.0, fitness)
    
class CarEvaluator:
    def __init__(self, config: dict):
        self.config = config
        # Later, we will put Pygame screen initialization here!
        
    def evaluate_genome(self, genome) -> float:
        pass
        return 0.0

def evaluate_population(genomes: list, config: dict, executor):
    environment = config["environment"]
    
    if environment == "xor":
        evaluator = XOREvaluator(config)
    elif environment == "car":
        evaluator = CarEvaluator(config)
    elif environment == "circle":
        evaluator = CircleEvaluator(config)
    else:
        raise ValueError(f"Unknown environment in config: {environment}")
        
    results = list(executor.map(evaluator.evaluate_genome, genomes))
    
    for genome, calculated_fitness in zip(genomes, results):
        genome.fitness = calculated_fitness