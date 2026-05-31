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
    checkpoints = []
    tunnel_width = 300
    gap_size = 100
    gate_spacing = 250
    num_gates = 30
    
    walls.append(((-tunnel_width, 0), (-tunnel_width, gate_spacing * num_gates)))
    walls.append(((tunnel_width, 0), (tunnel_width, gate_spacing * num_gates)))
    walls.append(((-tunnel_width, 0), (tunnel_width, 0))) # Garage Door
    
    for i in range(1, num_gates):
        y_pos = i * gate_spacing
        gap_center = random.uniform(-tunnel_width + gap_size, tunnel_width - gap_size)
        
        walls.append(((-tunnel_width, y_pos), (gap_center - gap_size/2, y_pos)))
        walls.append(((gap_center + gap_size/2, y_pos), (tunnel_width, y_pos)))
        
        checkpoints.append(((gap_center - gap_size/2, y_pos), (gap_center + gap_size/2, y_pos)))
        
    return walls, checkpoints

class Car:
    def __init__(self, genome, config, index):
        self.genome = genome
        self.config = config
        self.id = index
        self.network = NeuralNetwork(genome, config)
        self._live_display_score = 0.0
        
        self.total_fitness = 0.0 
        self.reset_for_track()
        
    def reset_for_track(self):
        self.network.reset()
        self.x = 0.0
        self.y = 50.0
        self.angle = math.pi / 2
        self.speed = 0.0
        self._live_display_score = 0.0
        
        self.is_alive = True
        self.history = [(self.x, self.y)]
        self.hit_points = []
        
        self.track_fitness = 0.0
        self.current_checkpoint = 0
        self.frames_since_progress = 0
        self.max_frames_without_progress = 80

def main():
    with open("config.json", "r") as file:
        config = json.load(file)
        
    try:
        pop = Population.load_json(f"{config['environment']}_training_backup.json", config)
        print(f"Loaded Generation {pop.current_generation} with {len(pop.genomes)} cars.")
    except FileNotFoundError:
        print("No backup found!")
        return

    pygame.init()
    screen = pygame.display.set_mode((1400, 800), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Courier", 18, bold=True)
    title_font = pygame.font.SysFont("Courier", 24, bold=True)

    cars = [Car(g, config, i) for i, g in enumerate(pop.genomes)]
    
    sensor_angles = [-math.pi/4, 0, math.pi/4]
    max_sensor_length = 800.0
    frames_per_track = 1600 

    for track_idx in range(5):
        track_seed = (pop.current_generation * 1000) + track_idx
        walls, checkpoints = build_track(track_seed)
        
        for car in cars:
            car.reset_for_track()

        cam_x, cam_y = 0.0, 200.0
        zoom = 0.8
        free_cam = False

        frame = 0
        track_finished = False
        proceed_to_next = False

        # --- THE NEW GAME LOOP ---
        while not proceed_to_next:
            WIDTH, HEIGHT = screen.get_size()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEWHEEL:
                    zoom = max(0.05, min(zoom + event.y * 0.1, 5.0))
                elif event.type == pygame.KEYDOWN:
                    # If the track is over, pressing ENTER moves to the next track!
                    if event.key == pygame.K_RETURN and track_finished:
                        proceed_to_next = True
                    
            keys = pygame.key.get_pressed()
            pan_speed = 20.0 / zoom
            if keys[pygame.K_w] or keys[pygame.K_UP]: cam_y += pan_speed; free_cam = True
            if keys[pygame.K_s] or keys[pygame.K_DOWN]: cam_y -= pan_speed; free_cam = True
            if keys[pygame.K_a] or keys[pygame.K_LEFT]: cam_x -= pan_speed; free_cam = True
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]: cam_x += pan_speed; free_cam = True
            if keys[pygame.K_SPACE]: free_cam = False

            alive_cars = [c for c in cars if c.is_alive]
            
            # --- PHYSICS --- (Only runs if the track is NOT finished)
            if not track_finished:
                if not free_cam and alive_cars:
                    best_car = max(alive_cars, key=lambda c: c.y)
                    cam_x += (best_car.x - cam_x) * 0.1
                    cam_y += (best_car.y - cam_y) * 0.1

                for car in alive_cars:
                    old_x, old_y = car.x, car.y
                    car.hit_points = []
                    sensor_distances = []
                    
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

                    inputs = sensor_distances + [car.speed]
                    accel, turn = car.network.feed_forward(inputs)
                    
                    car.speed = max(0.0, car.speed + accel) 
                    car.speed -= (0.015 * car.speed * abs(car.speed)) + (0.05 * car.speed)
                    
                    if abs(car.speed) > 0.1:
                        desired_rotation = turn * car.speed * 0.05
                        max_grip = 3.0 / max(car.speed * car.speed, 1.0)
                        car.angle += max(-max_grip, min(max_grip, desired_rotation))
                        
                    car.x += math.cos(car.angle) * car.speed
                    car.y += math.sin(car.angle) * car.speed

                    # Collisions
                    for wall in walls:
                        if get_line_intersection((old_x, old_y), (car.x, car.y), wall[0], wall[1]):
                            car.is_alive = False
                            break
                    
                    # Checkpoints
                    if car.current_checkpoint < len(checkpoints):
                        target_cp = checkpoints[car.current_checkpoint]
                        if get_line_intersection((old_x, old_y), (car.x, car.y), target_cp[0], target_cp[1]):
                            car.track_fitness += 100.0
                            car.current_checkpoint += 1
                            car.frames_since_progress = 0
                    
                    # Laziness Check
                    car.frames_since_progress += 1
                    if car.frames_since_progress > car.max_frames_without_progress:
                        car.is_alive = False

                    if frame % 3 == 0:
                        car.history.append((car.x, car.y))

                frame += 1
                
                # Check if the simulation for this track should end!
                if frame >= frames_per_track or not alive_cars:
                    track_finished = True

            # --- RENDER --- (Runs continuously so you can pan around while paused!)
            screen.fill((20, 20, 25)) 
            def to_screen(mx, my):
                return int((mx - cam_x) * zoom + (WIDTH / 2)), int((HEIGHT / 2) - (my - cam_y) * zoom)

            for wall in walls:
                pygame.draw.line(screen, (200, 200, 255), to_screen(*wall[0]), to_screen(*wall[1]), max(1, int(4 * zoom)))
                
            for cp in checkpoints:
                pygame.draw.line(screen, (255, 255, 0, 50), to_screen(*cp[0]), to_screen(*cp[1]), max(1, int(2 * zoom)))

            for car in cars:
                if len(car.history) > 1:
                    pygame.draw.lines(screen, (100, 30, 30) if not car.is_alive else (50, 200, 255), False, [to_screen(*p) for p in car.history], max(1, int(2 * zoom)))

                size = 12 * zoom
                if size > 1:
                    p1 = to_screen(car.x + math.cos(car.angle) * 12, car.y + math.sin(car.angle) * 12)
                    p2 = to_screen(car.x + math.cos(car.angle + 2.5) * 12, car.y + math.sin(car.angle + 2.5) * 12)
                    p3 = to_screen(car.x + math.cos(car.angle - 2.5) * 12, car.y + math.sin(car.angle - 2.5) * 12)
                    pygame.draw.polygon(screen, (50, 255, 50) if car.is_alive else (150, 0, 0), [p1, p2, p3])

                    car_pos = to_screen(car.x, car.y)
                    for hit in car.hit_points:
                        hit_pos = to_screen(hit[0], hit[1])
                        pygame.draw.line(screen, (255, 50, 50), car_pos, hit_pos, 1)

            # --- UI OVERLAYS ---
            for car in cars:
                car._live_display_score = max(0.0, car.track_fitness + min(car.y * 0.05, 10)) + car.total_fitness
            
            top_cars = sorted(cars, key=lambda c: c._live_display_score, reverse=True)[:10]
            
            pygame.draw.rect(screen, (10, 10, 10), (WIDTH - 350, 10, 310, 300), border_radius=10)
            
            status_text = "COMPLETED" if track_finished else f"{frame}/{frames_per_track}"
            title_text = title_font.render(f"TRACK {track_idx + 1}/5 | {status_text}", True, (255, 255, 255))
            screen.blit(title_text, (WIDTH - 330, 20))
            
            for i, car in enumerate(top_cars):
                color = (50, 255, 50) if car.is_alive else (150, 50, 50)
                text = font.render(f"#{i+1} Car {car.id:03d} - Score: {car._live_display_score:.1f}", True, color)
                screen.blit(text, (WIDTH - 330, 60 + i * 22))

            # --- THE PAUSE PROMPT ---
            if track_finished:
                prompt_text = title_font.render("TRACK FINISHED - PRESS [ENTER] FOR NEXT", True, (50, 255, 50))
                prompt_rect = prompt_text.get_rect(center=(WIDTH // 2, 50))
                # Draw a dark background box behind the text so it is easy to read
                pygame.draw.rect(screen, (20, 20, 20), prompt_rect.inflate(20, 20), border_radius=8)
                pygame.draw.rect(screen, (50, 255, 50), prompt_rect.inflate(20, 20), 2, border_radius=8)
                screen.blit(prompt_text, prompt_rect)

            pygame.display.flip()
            clock.tick(60) 
            
        # End of 'while not proceed_to_next' loop
        # Apply the final track score to their total average!
        for car in cars:
            car.total_fitness += max(0.0, car.track_fitness + min(car.y * 0.05, 10))
            
    # Show Final Average Scores in terminal once all 5 tracks are done
    print("\n--- 5-TRACK EVALUATION COMPLETE ---")
    for car in cars:
        car.total_fitness /= 5.0
    
    final_top = sorted(cars, key=lambda c: c.total_fitness, reverse=True)[:10]
    for i, car in enumerate(final_top):
        print(f"Rank {i+1}: Car {car.id:03d} - True Generalist Fitness: {car.total_fitness:.2f}")

if __name__ == "__main__":
    main()