import pygame
import math
import json
import sys
from phenotype import NeuralNetwork
from population import Population

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

def build_track(seed_val):
    import random
    random.seed(seed_val) 
    
    walls = []
    tunnel_width = 300
    gap_size = 100
    gate_spacing = 250
    num_gates = 30
    
    walls.append(((-tunnel_width, 0), (-tunnel_width, gate_spacing * num_gates)))
    walls.append(((tunnel_width, 0), (tunnel_width, gate_spacing * num_gates)))
    walls.append(((-tunnel_width, 0), (tunnel_width, 0)))
    
    for i in range(1, num_gates):
        y_pos = i * gate_spacing
        gap_center = random.uniform(-tunnel_width + gap_size, tunnel_width - gap_size)
        walls.append(((-tunnel_width, y_pos), (gap_center - gap_size/2, y_pos)))
        walls.append(((gap_center + gap_size/2, y_pos), (tunnel_width, y_pos)))
        
    return walls

class Car:
    def __init__(self, genome, config):
        self.network = NeuralNetwork(genome, config)
        self.network.reset()
        
        self.x = 0.0
        self.y = 50.0
        self.angle = math.pi / 2
        self.speed = 0.0
        
        self.is_alive = True
        self.history = [(self.x, self.y)]
        self.hit_points = []

def main():
    # 1. Load Data
    with open("config.json", "r") as file:
        config = json.load(file)
        
    try:
        pop = Population.load_json(f"{config['environment']}_training_backup.json", config)
        print(f"Loaded Generation {pop.current_generation} with {len(pop.genomes)} cars.")
    except FileNotFoundError:
        print("No backup found! Run main.py first.")
        return

    # CRITICAL BUG CHECK: Ensure this seed perfectly matches the seed in your evaluator.py
    # If you are testing 1 track per generation:
    #track_seed = pop.current_generation
    
    # If you are using the 5-track generalist:
    track_seed = (pop.current_generation * 1000) + 0 
    
    walls = build_track(track_seed)
    cars = [Car(g, config) for g in pop.genomes]

    # 2. Pygame Setup
    pygame.init()
    screen = pygame.display.set_mode((1200, 800), pygame.RESIZABLE)
    pygame.display.set_caption(f"NEAT Visualizer - WASD to Move, Scroll to Zoom, Space to Snap")
    clock = pygame.time.Clock()
    
    sensor_angles = [-math.pi/4, 0, math.pi/4]
    max_sensor_length = 500.0

    # 3. Camera State
    cam_x = 0.0
    cam_y = 200.0
    zoom = 0.8
    free_cam = False # True when user manually moves WASD

    running = True
    frame_counter = 0

    while running:
        WIDTH, HEIGHT = screen.get_size()
        
        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom in/out via scroll wheel
                zoom += event.y * 0.1
                zoom = max(0.05, min(zoom, 5.0))

        # --- CAMERA CONTROLS ---
        keys = pygame.key.get_pressed()
        pan_speed = 20.0 / zoom
        
        if keys[pygame.K_w] or keys[pygame.K_UP]: cam_y += pan_speed; free_cam = True
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: cam_y -= pan_speed; free_cam = True
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: cam_x -= pan_speed; free_cam = True
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: cam_x += pan_speed; free_cam = True
        
        # Snap back to best car
        if keys[pygame.K_SPACE]:
            free_cam = False

        alive_cars = [c for c in cars if c.is_alive]
        
        # Auto-follow camera logic
        if not free_cam and alive_cars:
            best_car = max(alive_cars, key=lambda c: c.y)
            # Smoothly interpolate camera to the best car
            cam_x += (best_car.x - cam_x) * 0.1
            cam_y += (best_car.y - cam_y) * 0.1

        # --- PHYSICS LOOP ---
        for car in alive_cars:
            old_x, old_y = car.x, car.y
            car.hit_points = []
            sensor_distances = []
            
            # Sensors
            for sensor_angle in sensor_angles:
                ray_angle = car.angle + sensor_angle
                ray_end = (car.x + math.cos(ray_angle) * max_sensor_length, car.y + math.sin(ray_angle) * max_sensor_length)
                
                closest_dist = max_sensor_length
                hit_point = ray_end
                
                for wall in walls:
                    dist = get_line_intersection((car.x, car.y), ray_end, wall[0], wall[1])
                    if dist is not None and dist < closest_dist:
                        closest_dist = dist
                        hit_point = (car.x + math.cos(ray_angle) * dist, car.y + math.sin(ray_angle) * dist)
                
                car.hit_points.append(hit_point)
                sensor_distances.append(1.0 - (closest_dist / max_sensor_length))

            # Brain & Steer
            inputs = sensor_distances + [car.speed]
            accel, turn = car.network.feed_forward(inputs)
            
            car.speed += accel
            car.speed -= (0.015 * car.speed * abs(car.speed)) + (0.05 * car.speed)
            
            if abs(car.speed) > 0.1:
                desired_rotation = turn * car.speed * 0.05
                max_grip = 1.5 / max(abs(car.speed), 1.0)
                car.angle += max(-max_grip, min(max_grip, desired_rotation))
                
            car.x += math.cos(car.angle) * car.speed
            car.y += math.sin(car.angle) * car.speed

            # Death Detection
            for wall in walls:
                if get_line_intersection((old_x, old_y), (car.x, car.y), wall[0], wall[1]):
                    car.is_alive = False
                    break
            
            # Save trail
            if frame_counter % 3 == 0:
                car.history.append((car.x, car.y))

        frame_counter += 1

        # --- RENDERING ---
        screen.fill((20, 20, 25)) 

        def to_screen(math_x, math_y):
            # Maps math coordinates to screen space, keeping Y pointing UP and applying zoom/pan
            sx = int((math_x - cam_x) * zoom + (WIDTH / 2))
            sy = int((HEIGHT / 2) - (math_y - cam_y) * zoom)
            return sx, sy

        # Draw Walls
        for wall in walls:
            p1 = to_screen(wall[0][0], wall[0][1])
            p2 = to_screen(wall[1][0], wall[1][1])
            pygame.draw.line(screen, (200, 200, 255), p1, p2, max(1, int(4 * zoom)))

        # Draw Paths and Cars
        for car in cars:
            if len(car.history) > 1:
                path_points = [to_screen(px, py) for px, py in car.history]
                path_color = (100, 30, 30) if not car.is_alive else (50, 200, 255)
                pygame.draw.lines(screen, path_color, False, path_points, max(1, int(2 * zoom)))

            size = 12 * zoom
            # Don't bother drawing the polygon if they zoomed out so far it's less than 1 pixel
            if size > 1:
                p1 = to_screen(car.x + math.cos(car.angle) * 24, car.y + math.sin(car.angle) * 24)
                p2 = to_screen(car.x + math.cos(car.angle + 2.5) * 12, car.y + math.sin(car.angle + 2.5) * 12)
                p3 = to_screen(car.x + math.cos(car.angle - 2.5) * 12, car.y + math.sin(car.angle - 2.5) * 12)
                
                if car.is_alive:
                    pygame.draw.polygon(screen, (50, 255, 50), [p1, p2, p3])
                    
                    # Draw sensors
                    car_pos = to_screen(car.x, car.y)
                    for hit in car.hit_points:
                        hit_pos = to_screen(hit[0], hit[1])
                        pygame.draw.line(screen, (255, 50, 50), car_pos, hit_pos, 1)
                else:
                    pygame.draw.polygon(screen, (150, 0, 0), [p1, p2, p3])

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()