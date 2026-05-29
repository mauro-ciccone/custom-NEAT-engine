

class Phenotype:
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