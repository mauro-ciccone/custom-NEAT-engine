from genotype import Genome
import json
import random

class Species:
    def __init__(self, species_id: int, mascot: Genome):
        self.id = species_id
        self.mascot = mascot
        self.members: list[Genome] = [mascot]
        self.fitness: float = 0.0

        self.allowed_children: int = 0
        self.max_fitness_ever: float = 0.0
        self.generations_since_improvement: int = 0

class Population:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.hidden_neurons_history: dict[tuple[int, int], int] = {}
        self.synapses_history: dict[tuple[int, int], int] = {}

        self.global_neuron_counter = 0
        self.global_synapse_counter = 0

        self.current_generation = 0

        self.species_list: list[Species] = []
        self.global_species_counter = 0

        self.compatibility_threshold = config["compatibility_threshold"]

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
            "current_generation": self.current_generation,
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

        pop.current_generation = data["current_generation"]

        pop.compatibility_threshold = data["compatibility_threshold"]

        pop.species_list = []
        pop.global_species_counter = 0
        
        pop.synapses_history = {
            (int(k.split(",")[0]), int(k.split(",")[1])): int(v)
            for k, v in data["synapses_history"].items()
        }
        
        pop.hidden_neurons_history = {
            (int(k.split(",")[0]), int(k.split(",")[1])): int(v)
            for k, v in data["hidden_neurons_history"].items()
        }
        
        pop.genomes = [Genome.from_dict(g_data) for g_data in data["genomes"]]
        
        return pop
    
    def run(self, evaluator_function, generations: int):

        start_gen = self.current_generation
        end_gen = start_gen + generations

        for generation in range(start_gen, end_gen):

            self.current_generation = generation

            evaluator_function(self.genomes, self.config)
            best_genome = max(self.genomes, key=lambda g: g.fitness)

            print(f"Gen {generation} | Best Fitness: {best_genome.fitness:.3f} / 100.0")

            if generation % self.config["extensive_log_per_gen"] == 0 or generation == 0:
                print(f"\n" + "="*40)
                print(f" 📊 GENERATION {generation} OVERVIEW")
                print("="*40)
                print(f" 🏆 Best Score : {best_genome.fitness:.3f}")
                print(f" 🧬 Species    : {len(self.species_list)} (Target: {self.config['target_species_count']})")
                print(f" 🎚️ Threshold  : {self.compatibility_threshold:.3f}")
                print(f" 🧠 Topo Size  : {self.global_neuron_counter} Nodes | {self.global_synapse_counter} Genes")
                print("="*40 + "\n")
    
            if best_genome.fitness >= self.config["premature_cutoff"]:
                print("Solution found!")
                return best_genome
                
            self.speciate()

            self.calculate_offspring_amounts()
            
            self.reproduce()
            
        return max(self.genomes, key=lambda g: g.fitness)
    
    def speciate(self):
        for species in self.species_list:
            if species.members:
                species.mascot = random.choice(species.members)
            
            species.members = []

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

        if self.config["dynamic_delta_threshold"]:
            target_species_count = self.config["target_species_count"]
            dynamic_threshold_shift = self.config["dynamic_threshold_shift"]

            num_species = len(self.species_list)

            if num_species < target_species_count:
                self.compatibility_threshold -= dynamic_threshold_shift
            elif num_species > target_species_count:
                self.compatibility_threshold += dynamic_threshold_shift

            if self.compatibility_threshold < 0.01:
                self.compatibility_threshold = 0.01
    
    def calculate_offspring_amounts(self):
        total_population_fitness = 0.0
        dropoff_age = self.config["species_dropoff_age"]

        global_best_fitness = max((g.fitness for g in self.genomes), default=0.0)

        for species in self.species_list:
            if len(species.members) > 0:
                current_max = max(g.fitness for g in species.members)
                if current_max > species.max_fitness_ever:
                    species.max_fitness_ever = current_max
                    species.generations_since_improvement = 0
                else:
                    species.generations_since_improvement += 1
                
                species.fitness = sum(g.fitness for g in species.members) / len(species.members)

                if species.generations_since_improvement >= dropoff_age and current_max < global_best_fitness:
                    species.fitness = 0.0

            else:
                species.fitness = 0.0
                
            total_population_fitness += species.fitness
        
        population_size = self.config["population_size"]
        total_assigned = 0

        for species in self.species_list:
            if total_population_fitness > 0:
                expected = (species.fitness / total_population_fitness) * population_size
            else:
                expected = population_size / len(self.species_list)
            
            species.allowed_children = int(expected)
            total_assigned += species.allowed_children

        leftovers = population_size - total_assigned
        if leftovers > 0:
            sorted_species = sorted(self.species_list, key=lambda s: s.fitness, reverse=True)
            for i in range(leftovers):
                sorted_species[i % len(sorted_species)].allowed_children += 1

    def reproduce(self):
        next_generation = []
        
        for species in self.species_list:
            if species.allowed_children <= 0:
                continue

            members_sorted = sorted(species.members, key=lambda g: g.fitness, reverse=True)
            survivor_threshold = max(1, int(len(members_sorted)*self.config["mate_best_percent"]))
            survivors = members_sorted[:survivor_threshold]
            children_spawned = 0

            if species.allowed_children >= self.config["species_champion_minsize"]:
                champion_clone = members_sorted[0].clone()
                next_generation.append(champion_clone)
                children_spawned += 1
            
            while children_spawned < species.allowed_children:
                if random.random() < self.config["offspring_from_mutation_percent"]:
                    parent1 = random.choice(survivors)
                    child = parent1.clone()
                else:
                    parent1 = random.choice(survivors)
                    if random.random() < self.config["interspecies_mating_rate"] and len(self.species_list) > 1:
                        random_species = random.choice([s for s in self.species_list if s.id != species.id])
                        parent2 = random.choice(random_species.members)
                    else:
                        parent2 = random.choice(survivors)
                    child = Genome.crossover(parent1, parent2, self.config)
                
                if random.random() < self.config["genome_mutate_weights_prob"]:
                    child.mutate_weights(self.config)
                
                if random.random() < self.config["new_node_mutation_prob"]:
                    self.global_neuron_counter, self.global_synapse_counter = child.mutate_add_neuron(
                        self.hidden_neurons_history, self.synapses_history, 
                        self.global_neuron_counter, self.global_synapse_counter
                    )

                if random.random() < self.config["new_link_mutation_prob"]:
                    self.global_synapse_counter = child.mutate_add_synapse(
                        self.synapses_history, self.global_synapse_counter, self.config
                    )
                
                child.synapses.sort(key=lambda x: x.innovation_id)
                
                next_generation.append(child)
                children_spawned += 1
        
        self.genomes = next_generation