import random
import math

class Node:
    def __init__(self, value, name):
        self.value = value
        self.name = name
        self.out_connections = []
    
    def info (self):
        print (f"{self.name} value: {self.value: .2f}")

class Connection:
    innovation_counter = 0

    def __init__(self, input, output):
        self.in_node = input
        self.out_node = output
        self.weight = random.uniform(-1, 1)
        self.is_enabled = True
        self.innovation_number = Connection.innovation_counter
        Connection.innovation_counter += 1
    
    def fire (self):
        self.out_node.value += self.in_node.value*self.weight
        print(f"Fired! Multiplied {self.in_node.value} by weight {self.weight:.2f}")

class Graph:
    def __init__(self, inputs_count, outputs_count):
        self.nodes = []
        self.input_nodes = []
        self.output_nodes = []
        self.connections = []
        self.topo_order_nodes = []

        bias_node = Node(1, "BIAS")
        self.nodes.append(bias_node)
        
        for i in range(inputs_count):
            node = Node(0, f"Sensor_{i}")
            self.nodes.append(node)
            self.input_nodes.append(node)

        self.all_starting_nodes = [bias_node] + self.input_nodes
        
        for i in range(outputs_count):
            node = Node(0, f"Controlsignal_{i}")
            self.nodes.append(node)
            self.output_nodes.append(node)
        
        for start_node in self.all_starting_nodes:
            for out_node in self.output_nodes:
                connection = Connection(start_node, out_node)
                self.connections.append(connection)
                start_node.out_connections.append(connection)

        self.topological_sort()

    def topological_sort(self):
        topo_order = []
        visited = set()

        for start_node in self.all_starting_nodes:
            if start_node not in visited:
                self._topo_dfs(start_node, visited, topo_order)
        

        self.topo_order_nodes = topo_order[::-1]

    def _topo_dfs(self, current_node, visited, topo_order):
        visited.add(current_node)

        for connection in current_node.out_connections:
            next_node = connection.out_node
            if next_node not in visited:
                self._topo_dfs(next_node, visited, topo_order)
        
        topo_order.append(current_node)
    
    def feed_forward(self, sensor_data):

        for i in range(len(sensor_data)):
            self.input_nodes[i].value = sensor_data[i]
        
        not_start_nodes = [node for node in self.nodes if node not in self.all_starting_nodes]

        for node in not_start_nodes:
            node.value = 0

        for node in self.topo_order_nodes:
            if node not in self.all_starting_nodes:
                node.value = math.tanh(node.value)
            for connection in node.out_connections:
                if connection.is_enabled:
                    connection.fire()
        
        return [node.value for node in self.output_nodes]
    
    def mutate_weight(self):
        connection = random.choice(self.connections)
        connection.weight += random.normalvariate(0, 0.5)
        connection.weight = max(-1, min(1, connection.weight))
    
    def mutate_add_node(self):
        connection = random.choice(self.connections)
        connection.is_enabled = False
        between_node = Node(0, "between_node")
        first_half = Connection(connection.in_node, between_node)
        second_half = Connection(between_node, connection.out_node)
        self.connections.append(first_half)
        self.connections.append(second_half)
        self.nodes.append(between_node)
        connection.in_node.out_connections.append(first_half)
        between_node.out_connections.append(second_half)
        self.topological_sort()
    
    def mutate_add_connection(self):
        for i in range(100):
            valid_in_nodes = [node for node in self.nodes if node not in self.output_nodes]
            valid_out_nodes = [node for node in self.nodes if node not in self.all_starting_nodes]

            in_node = random.choice(valid_in_nodes)
            out_node = random.choice(valid_out_nodes)

            connection_exists = False
            for connection in self.connections:
                if connection.in_node == in_node and connection.out_node == out_node:
                    connection_exists = True
                    break
            if connection_exists:
                continue

            if self.would_cause_cycle(in_node, out_node):
                continue

            new_connection = Connection(in_node, out_node)
            self.connections.append(new_connection)
            in_node.out_connections.append(new_connection)
            break
        self.topological_sort()

    def mutate_toggle_connection(self):
        connection = random.choice(self.connections)
        connection.is_enabled = not connection.is_enabled

    def would_cause_cycle(self, in_node, out_node):
 
        visited = set()

        return self._has_path(out_node, in_node, visited)

    def _has_path(self, current_node, target_node, visited):
        if current_node == target_node:
            return True
        
        visited.add(current_node)
        
        for connection in current_node.out_connections:
            if connection.out_node not in visited:
                if self._has_path(connection.out_node, target_node, visited):
                    return True
        
        return False