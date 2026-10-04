import pygame
import sys
import json
from game_2048 import Game2048
from population import Population
from phenotype import NeuralNetwork

TILE_COLORS = {
    0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
    8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
    64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
    512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)
}

TEXT_COLORS = {2: (119, 110, 101), 4: (119, 110, 101)}

# --- ANIMATION TRACKER ---
class VisualTile:
    def __init__(self, r, c, val):
        self.r = float(r)
        self.c = float(c)
        self.target_r = float(r)
        self.target_c = float(c)
        self.val = val
        self.is_dead = False # Flags tiles that merged and should disappear

    def update(self, speed=0.4):
        self.r += (self.target_r - self.r) * speed
        self.c += (self.target_c - self.c) * speed

    def is_at_target(self):
        return abs(self.target_r - self.r) < 0.05 and abs(self.target_c - self.c) < 0.05

def get_slide_mapping(line):
    """Calculates where tiles on a single row/col will end up after sliding."""
    non_zeros = [(i, val) for i, val in enumerate(line) if val != 0]
    mapping = []
    target_idx = 0
    skip = False
    
    for idx in range(len(non_zeros)):
        if skip:
            skip = False
            continue
        orig_i, val = non_zeros[idx]
        if idx + 1 < len(non_zeros) and non_zeros[idx+1][1] == val:
            mapping.append((orig_i, target_idx, val, False))
            mapping.append((non_zeros[idx+1][0], target_idx, val, True)) # The merged tile dies
            skip = True
            target_idx += 1
        else:
            mapping.append((orig_i, target_idx, val, False))
            target_idx += 1
    return mapping

def calculate_visual_targets(visual_tiles, grid, direction):
    """Assigns new target coordinates to visual tiles based on the chosen move."""
    vt_map = {(int(round(t.r)), int(round(t.c))): t for t in visual_tiles if not t.is_dead}
    
    for idx in range(4):
        if direction == 0: # UP
            line = [grid[r][idx] for r in range(4)]
            mapping = get_slide_mapping(line)
            for orig_i, targ_i, _, is_merge in mapping:
                if (orig_i, idx) in vt_map:
                    t = vt_map[(orig_i, idx)]
                    t.target_r, t.target_c, t.is_dead = targ_i, idx, is_merge
                    
        elif direction == 1: # RIGHT
            line = [grid[idx][c] for c in reversed(range(4))]
            mapping = get_slide_mapping(line)
            for orig_i, targ_i, _, is_merge in mapping:
                if (idx, 3 - orig_i) in vt_map:
                    t = vt_map[(idx, 3 - orig_i)]
                    t.target_r, t.target_c, t.is_dead = idx, 3 - targ_i, is_merge
                    
        elif direction == 2: # DOWN
            line = [grid[r][idx] for r in reversed(range(4))]
            mapping = get_slide_mapping(line)
            for orig_i, targ_i, _, is_merge in mapping:
                if (3 - orig_i, idx) in vt_map:
                    t = vt_map[(3 - orig_i, idx)]
                    t.target_r, t.target_c, t.is_dead = 3 - targ_i, idx, is_merge
                    
        elif direction == 3: # LEFT
            line = [grid[idx][c] for c in range(4)]
            mapping = get_slide_mapping(line)
            for orig_i, targ_i, _, is_merge in mapping:
                if (idx, orig_i) in vt_map:
                    t = vt_map[(idx, orig_i)]
                    t.target_r, t.target_c, t.is_dead = idx, targ_i, is_merge

def sync_grid_to_visual(grid):
    """Resets the visual tracking to perfectly match the raw game grid."""
    tiles = []
    for r in range(4):
        for c in range(4):
            if grid[r][c] != 0:
                tiles.append(VisualTile(r, c, grid[r][c]))
    return tiles
# -----------------------------

def draw_board(screen, game, visual_tiles, font, score_font):
    screen.fill((250, 248, 239))
    
    score_surface = score_font.render(f"Score: {game.score}  |  Max Tile: {game.get_max_tile()}", True, (119, 110, 101))
    screen.blit(score_surface, (20, 20))

    grid_size = 400
    grid_x, grid_y = 50, 80
    cell_size = 85
    padding = 10

    # 1. Draw Board Background
    pygame.draw.rect(screen, (187, 173, 160), (grid_x, grid_y, grid_size, grid_size), border_radius=8)
    for r in range(4):
        for c in range(4):
            x = grid_x + padding + c * (cell_size + padding)
            y = grid_y + padding + r * (cell_size + padding)
            pygame.draw.rect(screen, (205, 193, 180), (x, y, cell_size, cell_size), border_radius=5)

    # 2. Draw Moving Tiles on top
    for t in visual_tiles:
        x = grid_x + padding + t.c * (cell_size + padding)
        y = grid_y + padding + t.r * (cell_size + padding)

        color = TILE_COLORS.get(t.val, (60, 58, 50))
        pygame.draw.rect(screen, color, (x, y, cell_size, cell_size), border_radius=5)

        text_color = TEXT_COLORS.get(t.val, (249, 246, 242))
        text_surface = font.render(str(t.val), True, text_color)
        text_rect = text_surface.get_rect(center=(x + cell_size // 2, y + cell_size // 2))
        screen.blit(text_surface, text_rect)

def main():
    with open("config.json", "r") as file:
        config = json.load(file)

    backup_file = f"{config['environment']}_training_backup.json"

    try:
        pop = Population.load_json(backup_file, config)
        print(f"Loaded Generation {pop.current_generation} successfully!")
    except FileNotFoundError:
        print(f"File {backup_file} not found! Run main.py first.")
        return

    best_genome = max(pop.genomes, key=lambda g: g.fitness)
    network = NeuralNetwork(best_genome, config)

    pygame.init()
    screen = pygame.display.set_mode((500, 520))
    pygame.display.set_caption("NEAT 2048 Champion Live Viewer")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 28, bold=True)
    score_font = pygame.font.SysFont("Arial", 20, bold=True)

    game = Game2048()
    network.reset()

    visual_tiles = sync_grid_to_visual(game.grid)
    is_animating = False
    cooldown = 0
    running = True
    auto_play = True

    while running:
        manual_action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset
                    game.reset()
                    network.reset()
                    visual_tiles = sync_grid_to_visual(game.grid)
                    is_animating = False
                elif event.key == pygame.K_SPACE: # Toggle AI
                    auto_play = not auto_play
                elif not auto_play: # Manual Overrides
                    if event.key == pygame.K_UP: manual_action = 0
                    elif event.key == pygame.K_RIGHT: manual_action = 1
                    elif event.key == pygame.K_DOWN: manual_action = 2
                    elif event.key == pygame.K_LEFT: manual_action = 3

        # Logically take a move if not currently animating
        if not is_animating and not game.is_game_over():
            if cooldown > 0:
                cooldown -= 1
            else:
                chosen_action = None
                valid_moves = game.get_valid_moves()

                if auto_play:
                    state = game.get_normalized_state()
                    outputs = network.feed_forward(state)
                    action_preferences = sorted(range(4), key=lambda i: outputs[i], reverse=True)
                    chosen_action = next((act for act in action_preferences if act in valid_moves), None)
                elif manual_action in valid_moves:
                    chosen_action = manual_action

                if chosen_action is not None:
                    # 1. Figure out where the graphics should slide
                    calculate_visual_targets(visual_tiles, game.grid, chosen_action)
                    is_animating = True
                    # 2. Update the hidden game engine instantly
                    game.move(chosen_action)
        
        # Smoothly move visuals towards their targets
        if is_animating:
            all_done = True
            for t in visual_tiles:
                t.update(speed=0.45) 
                if not t.is_at_target():
                    all_done = False
            
            if all_done:
                is_animating = False
                cooldown = 1 # Small pause before the AI swipes again
                visual_tiles = sync_grid_to_visual(game.grid) # Snap to exact game grid, spawning new tiles

        draw_board(screen, game, visual_tiles, font, score_font)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()