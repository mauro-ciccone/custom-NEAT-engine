from genotype import Genome
import json
import random

class Species:
    def __init__(self, species_id: int, mascot: Genome):
        self.id = species_id
        
        self.mascot = mascot
        
        self.members: list[Genome] = [mascot]
        
        self.fitness: float = 0.0

class Population:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.hidden_neurons_history: dict[tuple[int, int], int] = {}
        self.synapses_history: dict[tuple[int, int], int] = {}

        self.global_neuron_counter = 0
        self.global_synapse_counter = 0

        self.species_list: list[Species] = []
        self.global_species_counter = 0

        self.compatibility_threshold = config.get("compatibility_threshold", 3.0)

        input_count = config["input_counts"]
        output_count = config["output_counts"]

        self.global_neuron_counter += input_count + output_count + 1

        inputs_and_bias = list(range(0, input_count + 1))
        outputs = list(range(input_count + 1, input_count + 1 + output_count))

        for in_node in inputs_and_bias:
            for out_node in outputs:
                self.synapses_history[(in_node, out_node)] = self.global_synapse_counter
                self.global_synapse_counter += 1
        
        self.genomes = [Genome(config) for _ in range(config["population_size"])]
    
    def save_json(self, filepath: str):
        save_data = {
            "global_neuron_counter": self.global_neuron_counter,
            "global_synapse_counter": self.global_synapse_counter,
            
            "synapses_history": {f"{k[0]},{k[1]}": v for k, v in self.synapses_history.items()},
            "hidden_neurons_history": {f"{k[0]},{k[1]}": v for k, v in self.hidden_neurons_history.items()},
            "compatibility_threshold": self.compatibility_threshold,
            "genomes": [g.to_dict() for g in self.genomes]
        }
        
        with open(filepath, "w") as file:
            json.dump(save_data, file, indent=4)

    @classmethod
    def load_json(cls, filepath: str, config: dict) -> 'Population':
        with open(filepath, "r") as file:
            data = json.load(file)
            
        pop = cls.__new__(cls)
        pop.config = config
        
        pop.global_neuron_counter = data["global_neuron_counter"]
        pop.global_synapse_counter = data["global_synapse_counter"]

        pop.compatibility_threshold = data.get("compatibility_threshold", config.get("compatibility_threshold", 3.0))
        
        pop.synapses_history = {
            (int(k.split(",")[0]), int(k.split(",")[1])): int(v)
            for k, v in data["synapses_history"].items()
        }
        
        pop.hidden_neurons_history = {
            (int(k.split(",")[0]), int(k.split(",")[1])): int(v)
            for k, v in data["hidden_neurons_history"].items()
        }
        
        pop.genomes = [Genome.from_dict(g_data, config) for g_data in data["genomes"]]
        
        return pop
    
    def run(self, evaluator_function, generations: int):
        for generation in range(generations):
            evaluator_function(self.genomes, self.config)
            best_genome = max(self.genomes, key=lambda g: g.fitness)
            print(f"Gen {generation} | Best Fitness: {best_genome.fitness:.3f} / 4.0")
    
            if best_genome.fitness >= 3.999999:
                print("Solution found!")
                return best_genome
                
            self.speciate() 
            
            # 4. MATING: Kill the weak, mutate the strong, spawn Gen N+1
            # self.reproduce()
            
        # Return the best genome we found after all generations
        return max(self.genomes, key=lambda g: g.fitness)
    
    def speciate(self):

        for species in self.species_list:
            if species.members:
                species.mascot = random.choice(species.members)
            
            species.members = []
        
        compatibility_threshold = self.config.get("compatibility_threshold", 3.0)

        for genome in self.genomes:
            found_species = False

            for species in self.species_list:
                distance = genome.distance_to(species.mascot, self.config)

                if distance < self.compatibility_threshold:
                    species.members.append(genome)
                    found_species = True
                    break
            
            if not found_species:
                new_species = Species(self.global_species_counter, genome)
                self.species_list.append(new_species)
                self.global_species_counter += 1
        
        self.species_list = [s for s in self.species_list if len(s.members) > 0]

        if self.config.get("dynamic_delta_threshold", False):
            target_species_count = self.config.get("target_species_count", 10)
            dynamic_threshold_shift = self.config.get("dynamic_threshold_shift", 0.1)

            num_species = len(self.species_list)

            if num_species < target_species_count:
                self.compatibility_threshold -= dynamic_threshold_shift
            elif num_species > target_species_count:
                self.compatibility_threshold += dynamic_threshold_shift

            if self.compatibility_threshold < 0.01:
                self.compatibility_threshold = 0.01