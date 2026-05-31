import math
import random
from phenotype import NeuralNetwork

class XOREvaluator:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        self.expected = [0.0, 1.0, 1.0, 0.0]

    def evaluate_genome(self, genome) -> float:
        network = NeuralNetwork(genome, self.config)
        fitness = 4.0 

        for i in range(4):
                network.reset()
                for _ in range(5):
                    network.feed_forward(self.inputs[i])
                
                output = network.feed_forward(self.inputs[i])[0]
                fitness -= abs(self.expected[i] - output)
        
        return max(0.0, fitness)

class CircleEvaluator:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.test_cases = []
        for i in range(10):
            for j in range(10):
                x = -1.0 + (i * (2.0 / 9.0))
                y = -1.0 + (j * (2.0 / 9.0))
                expected = 1.0 if (x**2 + y**2) < 0.49 else 0.0
                self.test_cases.append(([x, y], expected))
                
    # It takes ONE genome, and returns ONE float. That's it!
    def evaluate_genome(self, genome) -> float:
        network = NeuralNetwork(genome, self.config)
        fitness = 100.0 

        for inputs, expected in self.test_cases:
            network.reset()
            for _ in range(5):
                network.feed_forward(inputs)
            output = network.feed_forward(inputs)[0]
            fitness -= abs(expected - output)
        
        return max(0.0, fitness)
    
def get_line_intersection(p0, p1, p2, p3):
    """Calculates intersection between two line segments."""
    s1_x, s1_y = p1[0] - p0[0], p1[1] - p0[1]
    s2_x, s2_y = p3[0] - p2[0], p3[1] - p2[1]

    den = (-s2_x * s1_y + s1_x * s2_y)
    if den == 0: return None # Parallel

    s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / den
    t = ( s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / den

    if 0 <= s <= 1 and 0 <= t <= 1:
        i_x = p0[0] + (t * s1_x)
        i_y = p0[1] + (t * s1_y)
        return math.sqrt((i_x - p0[0])**2 + (i_y - p0[1])**2)
    return None

class CarEvaluator:
    def __init__(self, config: dict, gen_as_seed: int):
        self.config = config

        self.num_tracks = 5
        self.tracks = []
        
        for track_idx in range(self.num_tracks):
            random.seed((gen_as_seed * 1000) + track_idx) 
            
            walls = []
            checkpoints = []
            tunnel_width = 300
            gap_size = 100
            gate_spacing = 250
            num_gates = 20
            
            walls.append(((-tunnel_width, 0), (-tunnel_width, gate_spacing * num_gates)))
            walls.append(((tunnel_width, 0), (tunnel_width, gate_spacing * num_gates)))
            walls.append(((-tunnel_width, 0), (tunnel_width, 0)))
            
            for i in range(1, num_gates):
                y_pos = i * gate_spacing
                gap_center = random.uniform(-tunnel_width + gap_size, tunnel_width - gap_size)
                
                walls.append(((-tunnel_width, y_pos), (gap_center - gap_size/2, y_pos)))
                walls.append(((gap_center + gap_size/2, y_pos), (tunnel_width, y_pos)))
                checkpoints.append(((gap_center - gap_size/2, y_pos), (gap_center + gap_size/2, y_pos)))

            self.tracks.append({"walls": walls, "checkpoints": checkpoints})
       
        self.sensor_angles = [-math.pi/4, 0, math.pi/4]
        self.max_sensor_length = 800.0
        
    def evaluate_genome(self, genome) -> float:
        total_fitness = 0.0

        network = NeuralNetwork(genome, self.config)
        
        for track in self.tracks:
            walls = track["walls"]
            checkpoints = track["checkpoints"]
            
            network.reset() # CRITICAL: Wipe the RNN memory before starting a new track!
            
            x, y = 0.0, 50.0 
            angle = math.pi / 2 
            speed = 0.0
            
            fitness = 0.0
            current_checkpoint = 0
            frames_since_progress = 0
            max_frames_without_progress = 80 
            
            is_alive = True
            
            for _ in range(1600):
                if not is_alive: break

                sensor_distances = []
                for sensor_angle in self.sensor_angles:
                    ray_angle = angle + sensor_angle
                    ray_end = (x + math.cos(ray_angle) * self.max_sensor_length, y + math.sin(ray_angle) * self.max_sensor_length)
                    
                    closest_dist = self.max_sensor_length
                    for wall in walls:
                        dist = get_line_intersection((x, y), ray_end, wall[0], wall[1])
                        if dist is not None and dist < closest_dist:
                            closest_dist = dist
                    
                    sensor_distances.append(1.0 - (closest_dist / self.max_sensor_length))
                
                inputs = sensor_distances + [speed]
                outputs = network.feed_forward(inputs)

                accel, turn = outputs
                speed += accel

                air_drag = 0.015
                rolling_friction = 0.05
                speed -= (air_drag * speed * abs(speed)) + (rolling_friction * speed)

                if abs(speed) > 0.1:
                    desired_rotation = turn * speed * 0.05
                    max_grip = 3.0 / max(speed * speed, 1.0)
                    actual_rotation = max(-max_grip, min(max_grip, desired_rotation))
                    angle += actual_rotation
                    
                old_x, old_y = x, y
                x += math.cos(angle) * speed
                y += math.sin(angle) * speed
                
                for wall in walls:
                    if get_line_intersection((old_x, old_y), (x, y), wall[0], wall[1]):
                        is_alive = False
                        break
                
                if current_checkpoint < len(checkpoints):
                    target_cp = checkpoints[current_checkpoint]
                    if get_line_intersection((old_x, old_y), (x, y), target_cp[0], target_cp[1]):
                        fitness += 100.0 
                        current_checkpoint += 1
                        frames_since_progress = 0
                
                frames_since_progress += 1
                if frames_since_progress > max_frames_without_progress:
                    is_alive = False
            
            total_fitness += max(0.0, fitness + min(y * 0.05, 10))
        
        return total_fitness / self.num_tracks
    
def evaluate_genomes_batch(genomes_chunk: list, config: dict, curr_gen: int) -> list[float]:
    environment = config["environment"]
    if environment == "xor":
        evaluator = XOREvaluator(config)
    elif environment == "car":
        evaluator = CarEvaluator(config, curr_gen)
    elif environment == "circle":
        evaluator = CircleEvaluator(config)
    else:
        raise ValueError(f"Unknown environment in config: {environment}")
        
    results = []
    for genome in genomes_chunk:
        results.append(evaluator.evaluate_genome(genome))
        
    return results

def evaluate_population(genomes: list, config: dict, executor, curr_gen: int):
    # Determine chunk size so each worker gets exactly one thick batch
    num_workers = getattr(executor, '_max_workers', 10) 
    chunk_size = max(1, math.ceil(len(genomes) // num_workers))
    
    # Slice the population into chunks
    chunks = [genomes[i:i + chunk_size] for i in range(0, len(genomes), chunk_size)]
    
    # Submit batches instead of individual genomes
    futures = [executor.submit(evaluate_genomes_batch, chunk, config, curr_gen) for chunk in chunks]
    
    # Collect results chronologically
    results = []
    for future in futures:
        results.extend(future.result())
    
    # Map fitness back to genomes
    for genome, calculated_fitness in zip(genomes, results):
        genome.fitness = calculated_fitness