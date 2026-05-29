from phenotype import NeuralNetwork

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
    
class CarEvaluator:
    def __init__(self, config: dict):
        self.config = config
        # Later, we will put Pygame screen initialization here!
        
    def evaluate(self, genomes: list):
        # Later, we will put the Pygame physics while-loop here!
        pass

def evaluate_population(genomes: list, config: dict):
    environment = config.get("environment", "no environment in config")
    
    if environment == "xor":
        evaluator = XOREvaluator(config)
    elif environment == "car":
        evaluator = CarEvaluator(config)
    else:
        raise ValueError(f"Unknown environment in config: {environment}")
        
    evaluator.evaluate(genomes)