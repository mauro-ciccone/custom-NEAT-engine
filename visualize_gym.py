import gymnasium as gym
import json
import time
from population import Population
from phenotype import NeuralNetwork

def main():
    with open("config.json", "r") as file:
        config = json.load(file)
        
    gym_task = config["gym_task"]
    backup_file = f"{config['environment']}_training_backup.json"
        
    try:
        pop = Population.load_json(backup_file, config)
        print(f"Loaded Generation {pop.current_generation} for {gym_task}")
    except FileNotFoundError:
        print(f"No backup found at {backup_file}!")
        return

    best_genome = max(pop.genomes, key=lambda g: g.fitness)
    network = NeuralNetwork(best_genome, config)

    env = gym.make(gym_task, render_mode="human")
    
    print(f"Playing Champion | Average Fitness: {best_genome.fitness:.2f}")
    
    gen = pop.current_generation
    num_trials = config["num_trials"]

    for trial in range(num_trials):
            current_seed = (gen * 10) + trial
            print(f"\n--- Starting Trial {trial + 1}/{config['num_trials']} (Seed: {current_seed}) ---")
            
            # Force Gym to use the exact same weather/spawn conditions!
            state, info = env.reset(seed=current_seed)
            network.reset()
            total_reward = 0.0
            
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
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
                
            print(f"Trial {trial + 1} Finished! Score: {total_reward:.2f}")
            
            # Tiny pause between trials so you can read the score
            time.sleep(0.7)

if __name__ == "__main__":
    main()