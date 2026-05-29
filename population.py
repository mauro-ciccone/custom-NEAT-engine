from genotype import Neuron, Synapse

class Population:
    def __init__(self, config: dict) -> None:
        self.hidden_neurons_history: dict[tuple[int, int], int] = {}
        self.synapses_history: dict[tuple[int, int], int] = {}

        self.global_neuron_counter = 0
        self.global_synapse_counter = 0

        input_count = config["input_counts"]
        output_count = config["output_counts"]

        self.global_neuron_counter += input_count + output_count + 1

        inputs_and_bias = list(range(0, input_count + 1))
        outputs = list(range(input_count + 1, input_count + 1 + output_count))

        for in_node in inputs_and_bias:
            for out_node in outputs:
                self.synapses_history[(in_node, out_node)] = self.global_innov_counter
                self.global_innov_counter += 1


