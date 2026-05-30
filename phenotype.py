from genotype import NeuronType, Genome
import math


class NeuralNetwork:
    def __init__(self, genome: Genome, config: dict) -> None:
        def fast_steep_sigmoid(x):
            try:
                return 1.0 / (1.0 + math.exp(-x * 4.9))
            except OverflowError:
                return 0.0 if x < 0 else 1.0

        self.activation_functions = {
            "steep_sigmoid": fast_steep_sigmoid,
            "weak_sigmoid": lambda x: 1.0 / (1.0 + math.exp(max(min(-x, 100), -100))),
            "tanh": math.tanh,
            "relu": lambda x: max(0.0, x),
            "binary": lambda x: 1.0 if x > 0.5 else 0.0
        }

        self.activation_function = self.activation_functions[config["activation_function"]]

        # Data-oriented flattening: Use lists instead of objects for CPU cache-friendly iterations
        num_nodes = len(genome.neurons)
        self.current_values = [0.0] * num_nodes
        self.next_values = [0.0] * num_nodes
        
        self.connections = []        # Will store fast tuples: (in_index, out_index, weight)
        self.activation_indices = [] # Indices of nodes that require activation

        # Create a mapping from Genome ID to a 0-based array index
        self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(genome.neurons.keys())}

        # Safely preserve exact input/output order using the lists from the genotype
        self.input_indices = [self.node_id_to_idx[n_id] for n_id in genome.inputs]
        self.output_indices = [self.node_id_to_idx[n_id] for n_id in genome.outputs]

        for node_id, physical_node in genome.neurons.items():
            idx = self.node_id_to_idx[node_id]
            
            if physical_node.type == NeuronType.BIAS:
                self.current_values[idx] = 1.0 
            elif physical_node.type in (NeuronType.HIDDEN, NeuronType.OUTPUT):
                self.activation_indices.append(idx)
        
        # Map synapses to index-based connections
        for synapse in genome.synapses:
            if synapse.is_enabled:
                in_idx = self.node_id_to_idx[synapse.in_node_id]
                out_idx = self.node_id_to_idx[synapse.out_node_id]
                self.connections.append((in_idx, out_idx, synapse.weight))
    
    def feed_forward(self, input_data: list[float]) -> list[float]:
        # 1. Feed input values into assigned array indices
        for i, input_value in enumerate(input_data):
            self.current_values[self.input_indices[i]] = input_value
        
        # 2. Accumulate connections using fast tuple unpacking
        for in_idx, out_idx, weight in self.connections:
            self.next_values[out_idx] += self.current_values[in_idx] * weight
        
        # 3. Apply activation function & clear next state
        for idx in self.activation_indices:
            self.current_values[idx] = self.activation_function(self.next_values[idx])
            self.next_values[idx] = 0.0
        
        # 4. Return output slice
        return [self.current_values[idx] for idx in self.output_indices]
    
    def reset(self):
        # We only need to zero out hidden/output nodes; bias stays 1.0 and inputs are overwritten
        for idx in self.activation_indices:
            self.current_values[idx] = 0.0
            self.next_values[idx] = 0.0