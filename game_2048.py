import random
import math

class Game2048:
    def __init__(self):
        self.size = 4
        self.reset()

    def reset(self):
        self.grid = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.spawn_tile()
        self.spawn_tile()
        return self.get_normalized_state()

    def spawn_tile(self):
        empty_cells = [(r, c) for r in range(self.size) for c in range(self.size) if self.grid[r][c] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.grid[r][c] = 4 if random.random() < 0.1 else 2

    def get_normalized_state(self) -> list[float]:
        """Converts grid values into normalized log2 values (0.0 to 1.0)."""
        flattened = []
        for r in range(self.size):
            for c in range(self.size):
                val = self.grid[r][c]
                # Log2 normalization scales tile powers (2->1, 4->2, 2048->11, etc.)
                normalized = (math.log2(val) / 16.0) if val > 0 else 0.0
                flattened.append(normalized)
        return flattened

    def _compress_and_merge_line(self, line):
        # 1. Slide non-zero values left
        non_zeros = [x for x in line if x != 0]
        merged = []
        score_gained = 0
        skip = False

        # 2. Merge adjacent matching pairs
        for i in range(len(non_zeros)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zeros) and non_zeros[i] == non_zeros[i + 1]:
                merged_val = non_zeros[i] * 2
                merged.append(merged_val)
                score_gained += merged_val
                skip = True
            else:
                merged.append(non_zeros[i])

        # 3. Pad remaining line with zeros
        merged.extend([0] * (self.size - len(merged)))
        return merged, score_gained

    def move(self, direction: int) -> tuple[bool, int]:
        """
        Directions: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        Returns: (has_grid_changed, score_gained)
        """
        old_grid = [row[:] for row in self.grid]
        total_score_gained = 0

        for idx in range(self.size):
            if direction == 0:   # UP
                line = [self.grid[r][idx] for r in range(self.size)]
                new_line, score = self._compress_and_merge_line(line)
                for r in range(self.size):
                    self.grid[r][idx] = new_line[r]
            elif direction == 1: # RIGHT
                line = [self.grid[idx][c] for c in reversed(range(self.size))]
                new_line, score = self._compress_and_merge_line(line)
                for i, c in enumerate(reversed(range(self.size))):
                    self.grid[idx][c] = new_line[i]
            elif direction == 2: # DOWN
                line = [self.grid[r][idx] for r in reversed(range(self.size))]
                new_line, score = self._compress_and_merge_line(line)
                for i, r in enumerate(reversed(range(self.size))):
                    self.grid[r][idx] = new_line[i]
            elif direction == 3: # LEFT
                line = [self.grid[idx][c] for c in range(self.size)]
                new_line, score = self._compress_and_merge_line(line)
                for c in range(self.size):
                    self.grid[idx][c] = new_line[c]

            total_score_gained += score

        moved = old_grid != self.grid
        if moved:
            self.score += total_score_gained
            self.spawn_tile()

        return moved, total_score_gained

    def get_valid_moves(self) -> list[int]:
        valid = []
        for direction in range(4):
            # Test move on a temporary copy
            temp_game = Game2048()
            temp_game.grid = [row[:] for row in self.grid]
            moved, _ = temp_game.move(direction)
            if moved:
                valid.append(direction)
        return valid

    def is_game_over(self) -> bool:
        return len(self.get_valid_moves()) == 0

    def get_max_tile(self) -> int:
        return max(max(row) for row in self.grid)