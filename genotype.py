from enum import Enum
import random

class NeuronType(Enum):
    BIAS = "bias"
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"

class Neuron:
    def __init__(self, neuron_id: int, neuron_type: NeuronType) -> None:
        self.id = neuron_id

        if not isinstance(neuron_type, NeuronType):
            raise TypeError(f"type must be an instance of NeuronType, got {type(neuron_type)}")
        
        self.type = neuron_type

class Synapse:
    def __init__(self, input_node_id: int, output_node_id: int, weight: float, innovation_number: int) -> None:
        self.in_node_id = input_node_id
        self.out_node_id = output_node_id
        self.weight = weight
        self.is_enabled = True
        self.innovation_id = innovation_number

class Genome:
    def __init__(self, config: dict):
        self.fitness: float = 0.0

        self.neurons: dict[int, Neuron] = {}

        self.neuron_ids: list[int] = []
        self.inputs: list[int] = []
        self.outputs: list[int] = []
        self.inputs_and_bias: list[int] = []
        self.hidden: list[int] = []

        self.synapses: list[Synapse] = []

        input_count = config["input_counts"]
        output_count = config["output_counts"]

        self.neurons[0] = Neuron(0, NeuronType.BIAS)
        self.neuron_ids.append(0)
        self.inputs_and_bias.append(0)

        for i in range(1, input_count+1):
            self.neurons[i] = Neuron(i, NeuronType.INPUT)
            self.neuron_ids.append(i)
            self.inputs.append(i)
            self.inputs_and_bias.append(i)
        
        for i in range(input_count+1, input_count+1+output_count):
            self.neurons[i] = Neuron(i, NeuronType.OUTPUT)
            self.neuron_ids.append(i)
            self.outputs.append(i)

        innovation_counter = 0
        
        for startnode in self.inputs_and_bias:
            for output in self.outputs:
                initial_weight = random.uniform(config["weight_min_value"], config["weight_max_value"])
                self.synapses.append(Synapse(startnode, output, initial_weight, innovation_counter))
                innovation_counter += 1
    
    def to_dict(self) -> dict:
        return {
            "fitness": self.fitness,
            "neurons": [{"id": n.id, "type": n.type.value} for n in self.neurons.values()],
            "synapses": [
                {
                    "in": s.in_node_id, 
                    "out": s.out_node_id, 
                    "weight": s.weight, 
                    "innov": s.innovation_id, 
                    "enabled": s.is_enabled
                } for s in self.synapses
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Genome':
        genome = cls.__new__(cls)
        genome.fitness = data["fitness"]
        
        genome.neurons = {}
        genome.neuron_ids = []
        genome.inputs = []
        genome.outputs = []
        genome.inputs_and_bias = []
        genome.hidden = []
        
        for n_data in data["neurons"]:
            n_id = n_data["id"]
            n_type = NeuronType(n_data["type"]) # Converts string "hidden" back to Enum!
            
            genome.neurons[n_id] = Neuron(n_id, n_type)
            genome.neuron_ids.append(n_id)
            
            if n_type == NeuronType.BIAS:
                genome.inputs_and_bias.append(n_id)
            elif n_type == NeuronType.INPUT:
                genome.inputs.append(n_id)
                genome.inputs_and_bias.append(n_id)
            elif n_type == NeuronType.OUTPUT:
                genome.outputs.append(n_id)
            elif n_type == NeuronType.HIDDEN:
                genome.hidden.append(n_id)
                
        genome.synapses = []
        for s_data in data["synapses"]:
            syn = Synapse(s_data["in"], s_data["out"], s_data["weight"], s_data["innov"])
            syn.is_enabled = s_data["enabled"]
            genome.synapses.append(syn)
            
        return genome
    
    def mutate_weights(self, config: dict):
        for synapse in self.synapses:
            if random.random() <= config["mutate_weight_replace_prob"]:
                synapse.weight = random.uniform(config["weight_min_value"], config["weight_max_value"])
            else:
                shift = random.gauss(0.0, config["weight_mutate_power"])
                synapse.weight += shift
                synapse.weight = max(config["weight_min_value"], min(config["weight_max_value"], synapse.weight))

    def mutate_add_neuron(self, hidden_neurons_history: dict[tuple[int, int], int], synapses_history: dict[tuple[int, int], int], global_neuron_counter: int, global_synapse_counter: int) -> tuple[int, int]:
        enabled_synapses = [s for s in self.synapses if s.is_enabled]

        if len(enabled_synapses) == 0:
            return (global_neuron_counter, global_synapse_counter)
        
        target_synapse = random.choice(enabled_synapses)
        target_synapse.is_enabled = False

        in_node = target_synapse.in_node_id
        out_node = target_synapse.out_node_id

        mutation_key = (in_node, out_node)
        if mutation_key in hidden_neurons_history:
            new_node_id = hidden_neurons_history[mutation_key]
        else:
            new_node_id = global_neuron_counter
            hidden_neurons_history[mutation_key] = new_node_id
            global_neuron_counter += 1
        
        self.neuron_ids.append(new_node_id)
        self.hidden.append(new_node_id)
        self.neurons[new_node_id] = Neuron(new_node_id, NeuronType.HIDDEN)

        first_half_key = (in_node, new_node_id)
        second_half_key = (new_node_id, out_node)

        if first_half_key in synapses_history:
            first_half_id = synapses_history[first_half_key]
        else:
            first_half_id = global_synapse_counter
            synapses_history[first_half_key] = first_half_id
            global_synapse_counter += 1

        self.synapses.append(Synapse(in_node, new_node_id, 1, first_half_id))
        
        if second_half_key in synapses_history:
            second_half_id = synapses_history[second_half_key]
        else:
            second_half_id = global_synapse_counter
            synapses_history[second_half_key] = second_half_id
            global_synapse_counter += 1

        self.synapses.append(Synapse(new_node_id, out_node, target_synapse.weight, second_half_id))
        
        return (global_neuron_counter, global_synapse_counter)
    
    def mutate_add_synapse(self, synapses_history: dict[tuple[int, int], int], global_synapse_counter: int, config: dict) -> int:
        possible_in = self.neuron_ids
        possible_out = self.hidden + self.outputs

        existing_pairs = {(s.in_node_id, s.out_node_id) for s in self.synapses}

        available_pairs = [(i, o) for i in possible_in for o in possible_out if (i, o) not in existing_pairs]

        if not available_pairs:
            return global_synapse_counter
        
        new_pair = random.choice(available_pairs)
        in_node, out_node = new_pair

        if new_pair in synapses_history:
            innov_num = synapses_history[new_pair]
        else:
            innov_num = global_synapse_counter
            synapses_history[new_pair] = innov_num
            global_synapse_counter += 1
        
        initial_weight = random.uniform(config["weight_min_value"], config["weight_max_value"])

        new_synapse = Synapse(in_node, out_node, initial_weight, innov_num)
        self.synapses.append(new_synapse)

        return global_synapse_counter
    
    def mutate_toggle_connection(self):
        if not self.synapses:
            return

        synapse = random.choice(self.synapses)
        synapse.is_enabled = not synapse.is_enabled

    def distance_to(self, other_genome: 'Genome', config: dict) -> float:
        
        synapses1 = self.synapses
        synapses2 = other_genome.synapses
        i = 0
        j = 0

        matching = 0
        disjoint = 0
        weight_diff_sum = 0.0

        while i < len(synapses1) and j < len(synapses2):
            innov1 = synapses1[i].innovation_id
            innov2 = synapses2[j].innovation_id

            if innov1 == innov2:
                matching += 1
                weight_diff_sum += abs(synapses1[i].weight - synapses2[j].weight)
                i += 1
                j += 1
            elif innov1 < innov2:
                disjoint += 1
                i += 1
            else:
                disjoint += 1
                j += 1

        excess = (len(synapses1) - i) + (len(synapses2) - j)

        N = max(len(synapses1), len(synapses2))
        
        if N < config["small_genome_N"]:
            N = 1

        W = (weight_diff_sum / matching) if matching > 0 else 0.0

        c1 = config["c1"]
        c2 = config["c2"]
        c3 = config["c3"]

        distance = (c1 * excess / N) + (c2 * disjoint / N) + (c3 * W)
        
        return distance
    
    @classmethod
    def crossover(cls, parent1: 'Genome', parent2: 'Genome', config: dict) -> 'Genome':
        if parent1.fitness > parent2.fitness:
            better_parent, worse_parent = parent1, parent2
        elif parent2.fitness > parent1.fitness:
            better_parent, worse_parent = parent2, parent1
        else:
            better_parent, worse_parent = random.choice([(parent1, parent2), (parent2, parent1)])

        child = cls.__new__(cls)
        child.fitness = 0.0
        
        child.neurons = {}
        child.neuron_ids = []
        child.inputs = []
        child.outputs = []
        child.inputs_and_bias = []
        child.hidden = []
        
        for n_id, node in better_parent.neurons.items():
            child.neurons[n_id] = Neuron(n_id, node.type)
            child.neuron_ids.append(n_id)
            if node.type == NeuronType.BIAS:
                child.inputs_and_bias.append(n_id)
            elif node.type == NeuronType.INPUT:
                child.inputs.append(n_id)
                child.inputs_and_bias.append(n_id)
            elif node.type == NeuronType.OUTPUT:
                child.outputs.append(n_id)
            elif node.type == NeuronType.HIDDEN:
                child.hidden.append(n_id)

        child.synapses = []
        
        syn1 = sorted(better_parent.synapses, key=lambda x: x.innovation_id)
        syn2 = sorted(worse_parent.synapses, key=lambda x: x.innovation_id)
        
        i, j = 0, 0
        
        inherit_avg = config["inherit_average_weight"]
        disable_prob = config["disable_inherited_gene_prob"]
        
        while i < len(syn1) and j < len(syn2):
            s_better = syn1[i]
            s_worse = syn2[j]
            
            if s_better.innovation_id == s_worse.innovation_id:
                if inherit_avg:
                    new_weight = (s_better.weight + s_worse.weight) / 2.0
                else:
                    new_weight = s_better.weight if random.random() < 0.5 else s_worse.weight
                    
                new_synapse = Synapse(s_better.in_node_id, s_better.out_node_id, new_weight, s_better.innovation_id)
                
                if not s_better.is_enabled or not s_worse.is_enabled:
                    new_synapse.is_enabled = False if random.random() < disable_prob else True
                else:
                    new_synapse.is_enabled = True
                    
                child.synapses.append(new_synapse)
                i += 1
                j += 1
                
            elif s_better.innovation_id < s_worse.innovation_id:
                new_synapse = Synapse(s_better.in_node_id, s_better.out_node_id, s_better.weight, s_better.innovation_id)
                new_synapse.is_enabled = s_better.is_enabled
                child.synapses.append(new_synapse)
                i += 1
                
            else:
                j += 1
                
        while i < len(syn1):
            s_better = syn1[i]
            new_synapse = Synapse(s_better.in_node_id, s_better.out_node_id, s_better.weight, s_better.innovation_id)
            new_synapse.is_enabled = s_better.is_enabled
            child.synapses.append(new_synapse)
            i += 1
        
        return child
    
    def clone(self) -> 'Genome':
            new_genome = Genome.__new__(Genome)
            new_genome.fitness = self.fitness
        
            # Fast copy neurons
            new_genome.neurons = {n_id: Neuron(n_id, n.type) for n_id, n in self.neurons.items()}
            new_genome.neuron_ids = list(self.neuron_ids)
            new_genome.inputs = list(self.inputs)
            new_genome.outputs = list(self.outputs)
            new_genome.inputs_and_bias = list(self.inputs_and_bias)
            new_genome.hidden = list(self.hidden)
        
            # Fast copy synapses
            new_genome.synapses = []
            for s in self.synapses:
                new_syn = Synapse(s.in_node_id, s.out_node_id, s.weight, s.innovation_id)
                new_syn.is_enabled = s.is_enabled
                new_genome.synapses.append(new_syn)
            
            return new_genome