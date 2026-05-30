import pygame
import math
import json
import sys
from phenotype import NeuralNetwork
from population import Population

# --- Reusing the math from your evaluator ---
def get_line_intersection(p0, p1, p2, p3):
    s1_x, s1_y = p1[0] - p0[0], p1[1] - p0[1]
    s2_x, s2_y = p3[0] - p2[0], p3[1] - p2[1]

    den = (-s2_x * s1_y + s1_x * s2_y)
    if den == 0: return None

    s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / den
    t = ( s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / den

    if 0 <= s <= 1 and 0 <= t <= 1:
        i_x = p0[0] + (t * s1_x)
        i_y = p0[1] + (t * s1_y)
        return math.sqrt((i_x - p0[0])**2 + (i_y - p0[1])**2)
    return None

def build_track():
    import random
    random.seed(42) # MUST match the seed in evaluator.py so it's the exact same track!
    
    walls = []
    tunnel_width = 300
    gap_size = 100
    gate_spacing = 250
    num_gates = 20
    
    walls.append(((-tunnel_width, 0), (-tunnel_width, gate_spacing * num_gates)))
    walls.append(((tunnel_width, 0), (tunnel_width, gate_spacing * num_gates)))
    
    for i in range(1, num_gates):
        y_pos = i * gate_spacing
        gap_center = random.uniform(-tunnel_width + gap_size, tunnel_width - gap_size)
        walls.append(((-tunnel_width, y_pos), (gap_center - gap_size/2, y_pos)))
        walls.append(((gap_center + gap_size/2, y_pos), (tunnel_width, y_pos)))
        
    return walls

def main():
    # 1. Load Data
    with open("config.json", "r") as file:
        config = json.load(file)
        
    try:
        pop = Population.load_json(f"{config['environment']}_training_backup.json", config)
        best_genome = max(pop.genomes, key=lambda g: g.fitness)
        print(f"Loaded Champion with Fitness: {best_genome.fitness:.3f}")
    except FileNotFoundError:
        print("No backup found! Run main.py first to train the car.")
        return

    network = NeuralNetwork(best_genome, config)
    walls = build_track()

    # 2. Pygame Setup
    pygame.init()
    WIDTH, HEIGHT = 800, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT Car Visualizer")
    clock = pygame.time.Clock()

    # 3. Car Initial State (Matches evaluator exactly)
    x, y = 0.0, 50.0 
    angle = math.pi / 2 
    speed = 0.0
    
    sensor_angles = [-math.pi/4, 0, math.pi/4]
    max_sensor_length = 300.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- SENSORS ---
        sensor_distances = []
        hit_points = [] # For drawing the lasers
        
        for sensor_angle in sensor_angles:
            ray_angle = angle + sensor_angle
            ray_end = (
                x + math.cos(ray_angle) * max_sensor_length,
                y + math.sin(ray_angle) * max_sensor_length
            )
            
            closest_dist = max_sensor_length
            hit_point = ray_end
            
            for wall in walls:
                dist = get_line_intersection((x, y), ray_end, wall[0], wall[1])
                if dist is not None and dist < closest_dist:
                    closest_dist = dist
                    hit_point = (
                        x + math.cos(ray_angle) * dist,
                        y + math.sin(ray_angle) * dist
                    )
            
            hit_points.append(hit_point)
            sensor_distances.append(1.0 - (closest_dist / max_sensor_length))

        # --- BRAIN ---
        inputs = sensor_distances + [speed]
        outputs = network.feed_forward(inputs)
        accel, turn = outputs
        
        # --- PHYSICS ---
        speed += accel
        drag_force = (0.015 * speed * abs(speed)) + (0.05 * speed)
        speed -= drag_force
        
        if abs(speed) > 0.1:
            desired_rotation = turn * speed * 0.05
            max_grip = 1.5 / max(abs(speed), 1.0)
            actual_rotation = max(-max_grip, min(max_grip, desired_rotation))
            angle += actual_rotation
            
        x += math.cos(angle) * speed
        y += math.sin(angle) * speed

        # --- RENDER ---
        screen.fill((30, 30, 30)) # Dark gray background
        
        # Camera offset so the car stays in the lower-middle of the screen
        cam_x = x - WIDTH // 2
        cam_y = y - HEIGHT // 4 * 3 

        # Draw Walls
        for wall in walls:
            start_pos = (int(wall[0][0] - cam_x), int(wall[0][1] - cam_y))
            end_pos = (int(wall[1][0] - cam_x), int(wall[1][1] - cam_y))
            pygame.draw.line(screen, (255, 255, 255), start_pos, end_pos, 4)

        # Draw Sensor Lasers
        car_screen_pos = (int(x - cam_x), int(y - cam_y))
        for hit in hit_points:
            hit_screen_pos = (int(hit[0] - cam_x), int(hit[1] - cam_y))
            pygame.draw.line(screen, (255, 50, 50), car_screen_pos, hit_screen_pos, 1)
            pygame.draw.circle(screen, (255, 50, 50), hit_screen_pos, 4)

        # Draw Car (A simple triangle pointing in the direction of the angle)
        size = 15
        p1 = (x + math.cos(angle) * size * 2 - cam_x, y + math.sin(angle) * size * 2 - cam_y)
        p2 = (x + math.cos(angle + 2.5) * size - cam_x, y + math.sin(angle + 2.5) * size - cam_y)
        p3 = (x + math.cos(angle - 2.5) * size - cam_x, y + math.sin(angle - 2.5) * size - cam_y)
        pygame.draw.polygon(screen, (50, 255, 100), [p1, p2, p3])

        pygame.display.flip()
        clock.tick(60) # Lock to 60 FPS

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()