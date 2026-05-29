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

        self.synapses.append(Synapse(in_node, new_node_id, 1, global_synapse_counter))
        
        if second_half_key in synapses_history:
            second_half_id = synapses_history[second_half_key]
        else:
            second_half_id = global_synapse_counter
            synapses_history[second_half_key] = second_half_id
            global_synapse_counter += 1

        self.synapses.append(Synapse(new_node_id, out_node, target_synapse.weight, global_synapse_counter))
        
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