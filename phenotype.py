from genotype import NeuronType, Genome
import math


class Node:
    def __init__(self, node_id: int, node_type: NeuronType) -> None:
        self.id = node_id
        self.current_value: float = 0.0
        self.next_value: float = 0.0
        self.type = node_type

class Connection:
    def __init__(self, in_node: Node, out_node: Node, weight: float) -> None:
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight

class NeuralNetwork:
    def __init__(self, genome: Genome, config: dict) -> None:
        self.nodes: dict[int, Node] = {}
        self.connections: list[Connection] = []

        self.input_nodes: list[Node] = []
        self.output_nodes: list[Node] = []

        def fast_steep_sigmoid(x):
            try:
                return 1.0 / (1.0 + math.exp(-x * 4.9))
            except OverflowError:
                return 0.0 if x < 0 else 1.0

        self.activation_functions = {
            "steep_sigmoid": lambda x: fast_steep_sigmoid(x),
            "weak_sigmoid": lambda x: 1.0 / (1.0 + math.exp(max(min(-x, 100), -100))),
            "tanh": math.tanh,
            "relu": lambda x: max(0.0, x),
            "binary": lambda x: 1.0 if x > 0.5 else 0.0
        }

        self.activation_function = self.activation_functions[config["activation_function"]]

        for node_id, neuron_gene in genome.neurons.items():
            physical_node = Node(node_id, neuron_gene.type)
            self.nodes[node_id] = physical_node

            if physical_node.type == NeuronType.BIAS:
                physical_node.current_value = 1.0 
            elif physical_node.type == NeuronType.INPUT:
                self.input_nodes.append(physical_node)
            elif physical_node.type == NeuronType.OUTPUT:
                self.output_nodes.append(physical_node)
        
        for synapse in genome.synapses:
            if synapse.is_enabled:
                in_node = self.nodes[synapse.in_node_id]
                out_node = self.nodes[synapse.out_node_id]

                physical_conn = Connection(in_node, out_node, synapse.weight)
                self.connections.append(physical_conn)
    
    def feed_forward(self, input_data: list[float]) -> list[float]:
        for i, input_value in enumerate(input_data):
            self.input_nodes[i].current_value = input_value
        
        for connection in self.connections:
            connection.out_node.next_value += connection.in_node.current_value*connection.weight
        
        for node in self.nodes.values():
            if node.type in (NeuronType.HIDDEN, NeuronType.OUTPUT):
                node.current_value = self.activation_function(node.next_value)
                node.next_value = 0
        
        return [node.current_value for node in self.output_nodes]
    
    def reset(self):
        for node in self.nodes.values():
            if node.type in (NeuronType.HIDDEN, NeuronType.OUTPUT):
                node.current_value = 0.0
                node.next_value = 0.0