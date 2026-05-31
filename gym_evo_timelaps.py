import gymnasium as gym
import json
import time
import os
import glob
import threading
from population import Population
from phenotype import NeuralNetwork

# Global flag for the background thread to communicate with the main Gym loop
skip_requested = False

def listen_for_enter():
    """Background thread that listens for the Enter key."""
    global skip_requested
    while True:
        try:
            # input() blocks until Enter is pressed
            input() 
            skip_requested = True
        except EOFError:
            break

def main():
    global skip_requested
    
    # 1. Read the Config
    with open("config.json", "r") as file:
        config = json.load(file)
        
    gym_task = config["gym_task"]
    save_folder = "auto_saves"
    
    # 2. Find and Sort the Backup Files
    if not os.path.exists(save_folder):
        print(f"Directory '{save_folder}' not found!")
        return
        
    search_pattern = os.path.join(save_folder, "gen_*_backup.json")
    backup_files = glob.glob(search_pattern)
    
    if not backup_files:
        print(f"No backup files found in {save_folder}!")
        return

    # Sort files numerically by extracting the number from "gen_XX_backup.json"
    backup_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

    # 3. Create the Gym Environment ONCE (Keeps the window open!)
    env = gym.make(gym_task, render_mode="human")
    num_trials = config["num_trials"]

    print(f"Found {len(backup_files)} generations to play. Starting Timelapse...")
    print("💡 TIP: Press [ENTER] in the terminal to skip a boring generation!")
    time.sleep(2) 

    # Start the background listener thread
    threading.Thread(target=listen_for_enter, daemon=True).start()

    # 4. The Master Timelapse Loop
    for file_path in backup_files:
        print("\n" + "="*40)
        
        # Load the specific generation
        pop = Population.load_json(file_path, config)
        gen = pop.current_generation
        
        # Extract the Champion
        best_genome = max(pop.genomes, key=lambda g: g.fitness)
        network = NeuralNetwork(best_genome, config)
        
        print(f"🎬 PLAYING GENERATION {gen} | Fitness: {best_genome.fitness:.2f}")
        
        skip_file = False
        
        # Play the trials for this champion
        for trial in range(num_trials):
            if skip_file:
                break
                
            current_seed = (gen * 10) + trial
            print(f"--- Trial {trial + 1}/{num_trials} (Seed: {current_seed}) ---")
            
            state, info = env.reset(seed=current_seed)
            network.reset()
            total_reward = 0.0
            
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                # Check if the user pressed Enter in the background thread
                if skip_requested:
                    print("⏭️  Skip requested! Jumping to next generation...")
                    skip_requested = False
                    skip_file = True
                    break

                time.sleep(0.02) 
                
                outputs = network.feed_forward(state)
                
                # The Universal Adapter
                if config.get("use_arg_max", False):
                    action = outputs.index(max(outputs))
                else:
                    if len(outputs) > 1: 
                        action = outputs
                    else:
                        action = int(outputs[0])
                
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                
            if not skip_file:
                print(f"Score: {total_reward:.2f}")
                time.sleep(0.5) # Tiny pause before the next trial
            
        if not skip_file:
            print("="*40)
            
            # Make the pause between generations skippable too!
            pause_time = 1.5
            start_pause = time.time()
            while time.time() - start_pause < pause_time:
                if skip_requested:
                    skip_requested = False
                    break
                time.sleep(0.1)
        
    print("\n✅ Timelapse Complete!")
    env.close()

if __name__ == "__main__":
    main()