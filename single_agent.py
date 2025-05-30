import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tabulate import tabulate  
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
from collections import defaultdict, deque
import logging
import os
import random




output_dir = "ShipSim_Plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

total_visits = np.zeros((10, 10), dtype=int)

# Grid settings
grid_size = 10
start_pos = (0, grid_size-1)
victory_zone = (grid_size - 1, 0)
episodes = 3000      
max_steps_per_episode = 65  

# Generate critical zone in a central area of the grid
def generate_centered_critical_zone():
    center_positions = [
        (grid_size // 2, grid_size // 2),
        (grid_size // 2, grid_size // 2 - 1),
        (grid_size // 2, grid_size // 2 + 1),
        (grid_size // 2 - 1, grid_size // 2),
        (grid_size // 2 + 1, grid_size // 2),
    ]
    x, y = random.choice(center_positions)
    return [(x, y), (x, y + 1), (x + 1, y), (x + 1, y + 1)]

critical_zone = generate_centered_critical_zone()

num_actions = 4  
Q_table = np.zeros((grid_size, grid_size, num_actions))
learning_rate = 0.1
discount_factor = 0.95
exploration_rate = 1.0
exploration_decay = 0.99    
min_exploration_rate = 0.05
batch_size = 32  
memory = deque(maxlen=2000)


move_penalty = -10          
victory_reward = 500       
critical_penalty = -200     
min_cumulative_reward = -250   


best_reward = float('-inf')
worst_reward = float('inf')
best_path = None
worst_path = None
successful_runs = []         
rewards_over_episodes = []    
success_count = 0           
successful_rewards = []      


actions = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


def train_agent():
    global exploration_rate, best_reward, worst_reward, best_path, worst_path, successful_runs, success_count, successful_rewards

    for episode in range(episodes):
        state = np.array(start_pos)
        total_reward = 0
        done = False
        episode_path = [tuple(state)]
        visited_states = set()

        for step in range(max_steps_per_episode):

            total_visits[state[0], state[1]] += 1
            visited_states.add(tuple(state))

            # Determine valid actions (avoid revisiting states)
            valid_actions = []
            for i in range(num_actions):
                dx, dy = actions[i]
                new_state = (state[0] + dx, state[1] + dy)
                if (tuple(new_state) not in visited_states and 
                    0 <= new_state[0] < grid_size and 
                    0 <= new_state[1] < grid_size):
                    valid_actions.append(i)
            if not valid_actions:
                valid_actions = list(range(num_actions))

            # Epsilon-greedy action selection
            if np.random.rand() <= exploration_rate:
                action = random.choice(valid_actions)
            else:
                q_values = Q_table[state[0], state[1]]
                valid_q_values = [(i, q_values[i]) for i in valid_actions]
                action = max(valid_q_values, key=lambda x: x[1])[0]

            dx, dy = actions[action]
            new_state = (state[0] + dx, state[1] + dy)
            if not (0 <= new_state[0] < grid_size and 0 <= new_state[1] < grid_size):
                new_state = tuple(state)

            # Check for terminal conditions and assign reward accordingly
            if tuple(new_state) in critical_zone:
                reward = critical_penalty
                done = True
            elif tuple(new_state) == victory_zone:
                reward = victory_reward
                done = True
            else:
                reward = move_penalty

            memory.append((state, action, reward, new_state, done))
            total_reward += reward

           
            old_value = Q_table[state[0], state[1], action]
            if done:
                future_rewards = 0
            else:
                future_rewards = np.max(Q_table[new_state[0], new_state[1]])
            Q_table[state[0], state[1], action] = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * future_rewards)

            state = np.array(new_state)
            episode_path.append(tuple(state))

            # Terminate if cumulative reward falls below threshold
            if total_reward < min_cumulative_reward:
                done = True

            
            if done:
                break

        rewards_over_episodes.append(total_reward)

        # If the agent reached the victory zone, we record the episode as successful
        if tuple(state) == victory_zone:
            successful_runs.append((episode_path, total_reward))
            successful_rewards.append(total_reward)
            success_count += 1

            Q_table[victory_zone[0], victory_zone[1], :] = victory_reward

        if total_reward > best_reward:
            best_reward = total_reward
            best_path = episode_path
        if total_reward < worst_reward and tuple(state) == victory_zone:
            worst_reward = total_reward
            worst_path = episode_path

        print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}, Best: {best_reward}, Worst: {worst_reward}")
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

    if successful_runs:
        worst_run = min(successful_runs, key=lambda x: x[1])
        worst_reward = worst_run[1]
        worst_path = worst_run[0]
    else:
        worst_path = []


def animate_best_run(path):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.invert_yaxis()
    ax.set_title("Best Run with Rewards")
    ax.scatter(*zip(*critical_zone), color="red", label="Critical Zone", s=100, marker="s")
    ax.scatter(*victory_zone, color="green", label="Victory Zone", s=100, marker="o")
    ax.legend()
    reward_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center')
    line, = ax.plot([], [], 'b-', lw=2)
    def update(frame):
        if frame < len(path):
            x, y = path[frame]
            ax.plot(x, y, 'bo')
            ax.annotate(f'Step {frame}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
            line.set_data(*zip(*path[:frame+1]))
            reward_text.set_text(f'Current Reward: {best_reward}')
    anim = FuncAnimation(fig, update, frames=len(path), interval=300, repeat=False)
    plt.show()

def animate_worst_run(path):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.invert_yaxis()
    ax.set_title("Worst Run with Rewards")
    ax.scatter(*zip(*critical_zone), color="red", label="Critical Zone", s=100, marker="s")
    ax.scatter(*victory_zone, color="green", label="Victory Zone", s=100, marker="o")
    ax.legend()
    reward_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center')
    line, = ax.plot([], [], 'b-', lw=2)
    def update(frame):
        if frame < len(path):
            x, y = path[frame]
            ax.plot(x, y, 'bo')
            ax.annotate(f'Step {frame}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
            line.set_data(*zip(*path[:frame+1]))
            reward_text.set_text(f'Total Reward: {worst_reward}')
    anim = FuncAnimation(fig, update, frames=len(path), interval=300, repeat=False)
    plt.show()


def perform_risk_analysis():
    overall_avg_reward = np.mean(rewards_over_episodes) if rewards_over_episodes else 0
    success_rate = (success_count / episodes) * 100 if episodes > 0 else 0
    avg_success_reward = np.mean(successful_rewards) if successful_rewards else "N/A"
    avg_steps_victory = np.mean([len(path) for (path, rew) in successful_runs]) if successful_runs else "N/A"
    
    data = [
        ["Success Count", success_count],
        ["Success Rate (%)", f"{success_rate:.2f}"],
        ["Average Reward (all episodes)", round(overall_avg_reward, 2)],
        ["Average Reward (Successful Runs)", round(avg_success_reward, 2) if isinstance(avg_success_reward, float) else avg_success_reward],
        ["Best Reward", best_reward],
        ["Worst Reward", worst_reward],
        ["Avg. Steps to Victory", round(avg_steps_victory, 2) if isinstance(avg_steps_victory, float) else avg_steps_victory],
    ]
    table_pretty = tabulate(data, headers=["Metric", "Value"], tablefmt="pretty")
    table_latex = tabulate(data, headers=["Metric", "Value"], tablefmt="latex")
    print("Risk Analysis Metrics (Pretty):")
    print(table_pretty)
    print("\nRisk Analysis Metrics (LaTeX):")
    print(table_latex)
    
    if successful_runs:
        runs = range(1, len(successful_runs) + 1)
        plt.figure(figsize=(10, 6))
        final_rewards = [rew for (path, rew) in successful_runs]
        plt.plot(runs, final_rewards, label="Final Reward", marker='o', linestyle='-')
        plt.xlabel("Successful Run Number", fontsize=14)
        plt.ylabel("Final Cumulative Reward", fontsize=14)
        plt.title("Final Cumulative Rewards of Successful Runs", fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(output_dir, "ShipSim_Final_Cumulative_Rewards.pdf")
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.show()
    else:
        print("No successful runs to plot cumulative rewards.")

def enhanced_risk_analysis():
    import numpy as np
    from tabulate import tabulate
    rewards_array = np.array(rewards_over_episodes)
    if rewards_array.size > 0:
        mean_val = np.mean(rewards_array)
        std_val = np.std(rewards_array)
        median_val = np.median(rewards_array)
        q25 = np.percentile(rewards_array, 25)
        q75 = np.percentile(rewards_array, 75)
        n = rewards_array.size
        ci_lower = mean_val - 1.96*(std_val/np.sqrt(n))
        ci_upper = mean_val + 1.96*(std_val/np.sqrt(n))
        analysis_data = [["Reward", round(mean_val,2), round(std_val,2), round(median_val,2),
                          round(q25,2), round(q75,2), f"[{round(ci_lower,2)}, {round(ci_upper,2)}]"]]
    else:
        analysis_data = [["Reward", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]]
    headers = ["Metric", "Mean", "Std Dev", "Median", "25th %", "75th %", "95% CI"]
    table_pretty = tabulate(analysis_data, headers=headers, tablefmt="pretty")
    table_latex = tabulate(analysis_data, headers=headers, tablefmt="latex")
    print("Reward Distribution Metrics (Pretty):")
    print(table_pretty)
    print("\nReward Distribution Metrics (LaTeX):")
    print(table_latex)

def plot_learning_curves():
    rewards_array = np.array(rewards_over_episodes)
    if rewards_array.size > 0:
        window = 50
        cumsum_vec = np.cumsum(np.insert(rewards_array, 0, 0))
        moving_avg = (cumsum_vec[window:] - cumsum_vec[:-window]) / window
        plt.figure(figsize=(12,8))
        plt.plot(moving_avg, label="Moving Avg Reward")
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Cumulative Reward (Moving Avg)", fontsize=14)
        plt.title("Learning Curves", fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(output_dir, "ShipSim_Learning_Curves.pdf")
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.show()

def plot_heatmap(
    data,
    grid_size,
    title,
    cmap='viridis',
    annotations=False,
    fs_title=72,
    fs_label=68,
    fs_tick=64,
    fs_ann=68,
    marker_scale=7500
):
    fig, ax = plt.subplots(figsize=(74, 58))

    im = ax.imshow(data, cmap=cmap, origin='lower')

    if 'Visitation Frequencies' in title:
        for cz in critical_zone:
            ax.scatter(
                cz[0], cz[1],
                marker='s',
                color='red',
                s=marker_scale,
                edgecolors='black',
                linewidths=2
            )

    
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    if 'Visitation Frequencies' in title:
        cbar_lbl = 'Visitation Count'
    elif 'Maximum Q-values' in title:
        cbar_lbl = 'Max Q-value'
    else:
        cbar_lbl = ''
    cbar.set_label(cbar_lbl, rotation=270, labelpad=20, fontsize=fs_label)
    cbar.ax.tick_params(labelsize=fs_tick)


    ax.set_title(title, fontsize=fs_title, fontweight='bold')
    ax.set_xlabel('X-axis', fontsize=fs_label)
    ax.set_ylabel('Y-axis', fontsize=fs_label)

    
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.tick_params(axis='both', which='major', labelsize=fs_tick, length=6)

    
    if annotations:
        vmax = data.max()
        for i in range(grid_size):
            for j in range(grid_size):
                val = data[i, j]
                txt = f"{val:.2f}" if 'Maximum Q-values' in title else str(val)
                color = 'white' if val > vmax/2 else 'black'
                ax.text(
                    j, i, txt,
                    ha='center', va='center',
                    color=color,
                    fontsize=fs_ann
                )

    fig.tight_layout()
    fn = os.path.join(output_dir, title.replace(" ", "_") + ".pdf")
    fig.savefig(fn, format="pdf", bbox_inches="tight")
    plt.close(fig)

def plot_heatmaps():
    total_visits[victory_zone] = success_count
    plot_heatmap(total_visits.T, grid_size, "Ship Visitation Frequencies", cmap='Blues', annotations=True)
    max_q = np.max(Q_table, axis=2)
    plot_heatmap(max_q.T, grid_size, "Ship Maximum Q-values", cmap='Oranges', annotations=True)

def print_all_statistics_latex():
    overall_avg_reward = np.mean(rewards_over_episodes) if rewards_over_episodes else 0
    success_rate = (success_count / episodes) * 100 if episodes > 0 else 0
    avg_success_reward = np.mean(successful_rewards) if successful_rewards else "N/A"
    avg_steps_victory = np.mean([len(path) for (path, rew) in successful_runs]) if successful_runs else "N/A"
    risk_data = [
        ["Success Count", success_count],
        ["Success Rate (%)", f"{success_rate:.2f}"],
        ["Average Reward (all episodes)", round(overall_avg_reward, 2)],
        ["Average Reward (Successful Runs)", round(avg_success_reward, 2) if isinstance(avg_success_reward, float) else avg_success_reward],
        ["Best Reward", best_reward],
        ["Worst Reward", worst_reward],
        ["Avg. Steps to Victory", round(avg_steps_victory, 2) if isinstance(avg_steps_victory, float) else avg_steps_victory],
    ]
    risk_table_latex = tabulate(risk_data, headers=["Metric", "Value"], tablefmt="latex")
    print("Risk Analysis Metrics (LaTeX):")
    print(risk_table_latex)

    if successful_runs:
        steps = [len(path) for (path, rew) in successful_runs]
        avg_steps = np.mean(steps)
        std_steps = np.std(steps)
        steps_data = [["Ship", round(avg_steps, 2), round(std_steps, 2)]]
    else:
        steps_data = [["Ship", "N/A", "N/A"]]
    steps_table_latex = tabulate(steps_data, headers=["Entity", "Average Steps", "Std Dev"], tablefmt="latex")
    print("\nAverage Steps to Victory (LaTeX):")
    print(steps_table_latex)

    max_q = np.max(Q_table, axis=2)
    avg_max_q = np.mean(max_q)
    qvalue_data = [["Ship", round(avg_max_q, 2)]]
    qvalue_table_latex = tabulate(qvalue_data, headers=["Entity", "Avg Max Q‑value"], tablefmt="latex")
    print("\nAverage Maximum Q‑Values per State (LaTeX):")
    print(qvalue_table_latex)

    total = np.sum(total_visits)
    if total == 0:
        entropy = 0
    else:
        probs = total_visits.flatten() / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
    visitation_data = [["Ship", round(entropy, 2)]]
    visitation_table_latex = tabulate(visitation_data, headers=["Entity", "Entropy"], tablefmt="latex")
    print("\nVisitation Entropy (LaTeX):")
    print(visitation_table_latex)


def main():
    global successful_runs, collision_paths
    train_agent()

    if best_path:
        animate_best_run(best_path)
    if worst_path:
        animate_worst_run(worst_path)

  # perform_risk_analysis()
  # enhanced_risk_analysis()
  # plot_learning_curves() 
    #plot_heatmaps()
    #print_all_statistics_latex()

if __name__ == "__main__":
    main()
