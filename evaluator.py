from phenotype import NeuralNetwork
import random

class XOREvaluator:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        self.expected = [0.0, 1.0, 1.0, 0.0]
    
    def evaluate(self, genomes: list):
        for genome in genomes:
            network = NeuralNetwork(genome, self.config)
            fitness = 4.0

            for i in range(4):

                for _ in range(20):
                    network.feed_forward(self.inputs[i])
                
                output = network.feed_forward(self.inputs[i])[0]
                fitness -= abs(self.expected[i] - output)
            
            genome.fitness = fitness

class CircleEvaluator:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.test_cases = []
        
        # Generate 20 random (x, y) coordinates
        for _ in range(20):
            x = random.uniform(-1.0, 1.0)
            y = random.uniform(-1.0, 1.0)
            
            # Math: x^2 + y^2 < r^2. Let's use a radius of 0.7 (r^2 ≈ 0.49)
            expected = 1.0 if (x**2 + y**2) < 0.49 else 0.0
            self.test_cases.append(([x, y], expected))
    
    def evaluate(self, genomes: list):
        for genome in genomes:
            network = NeuralNetwork(genome, self.config)
            fitness = 20.0  # Max score is 20!

            for inputs, expected in self.test_cases:
                # Flush the network
                for _ in range(10):
                    network.feed_forward(inputs)
                
                output = network.feed_forward(inputs)[0]
                fitness -= abs(expected - output)
            
            # Prevent negative fitness
            genome.fitness = max(0.0, fitness)
    
class CarEvaluator:
    def __init__(self, config: dict):
        self.config = config
        # Later, we will put Pygame screen initialization here!
        
    def evaluate(self, genomes: list):
        # Later, we will put the Pygame physics while-loop here!
        pass

def evaluate_population(genomes: list, config: dict):
    environment = config["environment"]
    
    if environment == "xor":
        evaluator = XOREvaluator(config)
    elif environment == "car":
        evaluator = CarEvaluator(config)
    elif environment == "circle":
        evaluator = CircleEvaluator(config)
    else:
        raise ValueError(f"Unknown environment in config: {environment}")
        
    evaluator.evaluate(genomes)