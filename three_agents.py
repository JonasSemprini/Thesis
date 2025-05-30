import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tabulate import tabulate  
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
from collections import defaultdict
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, auc



@dataclass(frozen=True)
class GridSettings:
    size: int = 10
    
    agent1_start: Tuple[int, int] = (0, 9)         # Top-left corner (row 0, col 9)
    agent2_start: Tuple[int, int] = (9, 0)         # Bottom-right corner (row 9, col 0)
    agent3_start: Tuple[int, int] = (0, 0)         # Top-left corner (row 0, col 0)
   
    agent1_victory_zone: Tuple[int, int] = (9, 0)  # Agent 1 aims for bottom-right (row 9, col 0)
    agent2_victory_zone: Tuple[int, int] = (0, 9)  # Agent 2 aims for top-left (row 0, col 9)
    agent3_victory_zone: Tuple[int, int] = (9, 9)  # Agent 3 aims for bottom-right (row 9, col 9)
    critical_zone_size: int = 2

@dataclass(frozen=True)
class SimulationParameters:
    episodes: int = 3000
    max_steps_per_episode: int = 65


@dataclass(frozen=True)
class Rewards:
    victory: int = 500
    directional: int = 20
    wandering: int = -10

@dataclass(frozen=True)
class Penalties:
    critical: int = -200
    collision: int = -500
    revisit: int = -50
    min_cumulative: int = -250



class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

ACTIONS = [
    (-1, 0),  # UP
    (1, 0),   # DOWN
    (0, -1),  # LEFT
    (0, 1)    # RIGHT
]

@dataclass
class QLearningParameters:
    num_actions: int = 4
    actions: List[Tuple[int, int]] = field(default_factory=lambda: ACTIONS)
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 1.0
    exploration_decay: float = 0.99
    min_exploration_rate: float = 0.05

# Initialize instances
grid_settings = GridSettings()
simulation_params = SimulationParameters()
rewards = Rewards()
penalties = Penalties()
ql_params = QLearningParameters()



Q_table_agent1 = np.zeros((grid_settings.size, grid_settings.size, ql_params.num_actions))
Q_table_agent2 = np.zeros((grid_settings.size, grid_settings.size, ql_params.num_actions))
Q_table_agent3 = np.zeros((grid_settings.size, grid_settings.size, ql_params.num_actions))


Position = Tuple[int, int]
Path = List[Position]
RewardsList = List[int]
AgentRun = Tuple[Path, RewardsList]


@dataclass
class SuccessfulPath:
    agent1_run: AgentRun
    agent2_run: AgentRun
    agent3_run: AgentRun

@dataclass
class CollisionPath:
    agent1_run: AgentRun
    agent2_run: AgentRun
    agent3_run: AgentRun


success_count: int = 0
collision_count: int = 0

successful_paths: List[SuccessfulPath] = []
collision_paths: List[CollisionPath] = []

reward_stats: Dict[str, Dict[str, List]] = {
    "agent1": defaultdict(list),
    "agent2": defaultdict(list),
    "agent3": defaultdict(list),
}


avg_max_q_history: List[float] = []
success_rate_history: List[float] = []


total_visits_agent1 = np.zeros((grid_settings.size, grid_settings.size), dtype=int)
total_visits_agent2 = np.zeros((grid_settings.size, grid_settings.size), dtype=int)
total_visits_agent3 = np.zeros((grid_settings.size, grid_settings.size), dtype=int)



logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)



def generate_dynamic_critical_zone(grid_size: int, critical_zone_size: int) -> List[Tuple[int, int]]:
    center = grid_size // 2
    return [(center + i, center + j) for i in range(-1, critical_zone_size - 1) 
            for j in range(-1, critical_zone_size - 1)]

critical_zone = generate_dynamic_critical_zone(grid_settings.size, grid_settings.critical_zone_size)

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def calculate_reward(agent_pos: Tuple[int, int],
                     other_agent_pos: Tuple[int, int],
                     victory_zone: Tuple[int, int],
                     critical_zone: List[Tuple[int, int]],
                     last_state: Tuple[int, int],
                     visited_states: Dict[Tuple[int, int], int],
                     has_reached_goal: bool,
                     rewards: Rewards,
                     penalties: Penalties) -> int:
    if has_reached_goal:
        return 0
    reward = 0
    if agent_pos == victory_zone:
        return rewards.victory
    if any(manhattan_distance(agent_pos, cz) < 2 for cz in critical_zone):
        reward += penalties.critical
    if agent_pos in visited_states:
        reward += penalties.revisit * visited_states[agent_pos]
    if last_state is not None and manhattan_distance(agent_pos, victory_zone) < manhattan_distance(last_state, victory_zone):
        reward += rewards.directional
    reward += rewards.wandering
    return reward

def epsilon_greedy_action_selection(state: Tuple[int, int],
                                    Q_table: np.ndarray,
                                    epsilon: float,
                                    last_state: Tuple[int, int],
                                    actions: List[Tuple[int, int]],
                                    grid_size: int) -> int:
    valid_actions = []
    for action_idx, (dx, dy) in enumerate(actions):
        new_state = (state[0] + dx, state[1] + dy)
        if (0 <= new_state[0] < grid_size and 0 <= new_state[1] < grid_size and
            new_state != state and new_state != last_state):
            valid_actions.append(action_idx)
            
    if not valid_actions:
        return -1

    if np.random.rand() <= epsilon:
        return np.random.choice(valid_actions)

    x, y = state
    q_values = Q_table[x, y, valid_actions]

    return valid_actions[np.argmax(q_values)]

def update_q_table(Q_table: np.ndarray,
                   state: Tuple[int, int],
                   action: int,
                   reward: int,
                   new_state: Tuple[int, int],
                   learning_rate: float,
                   discount_factor: float) -> None:
    x, y = state
    nx, ny = new_state
    old_value = Q_table[x, y, action]
    future_rewards = np.max(Q_table[nx, ny])
    Q_table[x, y, action] = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * future_rewards)
    logging.debug(f"Updated Q[{x},{y},{action}] from {old_value:.2f} to {Q_table[x, y, action]:.2f}")



def train_agents():
    global success_count, collision_count, successful_paths, collision_paths
    global total_visits_agent1, total_visits_agent2, total_visits_agent3

    for episode in range(1, simulation_params.episodes + 1):
        agent1_pos = grid_settings.agent1_start
        agent2_pos = grid_settings.agent2_start
        agent3_pos = grid_settings.agent3_start
        total_reward_agent1 = 0
        total_reward_agent2 = 0
        total_reward_agent3 = 0
        visited_states_agent1 = defaultdict(int)
        visited_states_agent2 = defaultdict(int)
        visited_states_agent3 = defaultdict(int)
        episode_path_agent1 = [agent1_pos]
        episode_path_agent2 = [agent2_pos]
        episode_path_agent3 = [agent3_pos]
        last_state_agent1 = None
        last_state_agent2 = None
        last_state_agent3 = None
        agent1_reached_goal = False
        agent2_reached_goal = False
        agent3_reached_goal = False
        cumulative_rewards_agent1 = [0]
        cumulative_rewards_agent2 = [0]
        cumulative_rewards_agent3 = [0]
        logging.debug(f"Episode {episode}: Agent 1 starts at {agent1_pos}, Agent 2 starts at {agent2_pos}, Agent 3 starts at {agent3_pos}")

        for step in range(simulation_params.max_steps_per_episode):
            visited_states_agent1[agent1_pos] += 1
            visited_states_agent2[agent2_pos] += 1
            visited_states_agent3[agent3_pos] += 1

            
            if not agent1_reached_goal:
                current_state_agent1 = agent1_pos
                action_agent1 = epsilon_greedy_action_selection(
                    state=agent1_pos,
                    Q_table=Q_table_agent1,
                    epsilon=ql_params.exploration_rate,
                    last_state=last_state_agent1,
                    actions=ql_params.actions,
                    grid_size=grid_settings.size
                )
                if action_agent1 == -1:
                    logging.debug(f"Episode {episode}: Agent 1 has no valid actions.")
                    break
                dx1, dy1 = ql_params.actions[action_agent1]
                new_agent1_pos = (agent1_pos[0] + dx1, agent1_pos[1] + dy1)
                if (new_agent1_pos != agent1_pos and new_agent1_pos != last_state_agent1 and
                    0 <= new_agent1_pos[0] < grid_settings.size and 0 <= new_agent1_pos[1] < grid_settings.size):
                    last_state_agent1 = agent1_pos
                    agent1_pos = new_agent1_pos
                    reward_agent1 = calculate_reward(
                        agent_pos=agent1_pos,
                        other_agent_pos=agent2_pos,  
                        victory_zone=grid_settings.agent1_victory_zone,
                        critical_zone=critical_zone,
                        last_state=last_state_agent1,
                        visited_states=visited_states_agent1,
                        has_reached_goal=agent1_reached_goal,
                        rewards=rewards,
                        penalties=penalties
                    )
                    total_reward_agent1 += reward_agent1
                    episode_path_agent1.append(agent1_pos)
                    cumulative_rewards_agent1.append(total_reward_agent1)
                else:
                    cumulative_rewards_agent1.append(total_reward_agent1)
                if agent1_pos == grid_settings.agent1_victory_zone:
                    agent1_reached_goal = True
                    logging.debug(f"Episode {episode}: Agent 1 reached its victory zone.")

            
            if not agent2_reached_goal:
                current_state_agent2 = agent2_pos
                action_agent2 = epsilon_greedy_action_selection(
                    state=agent2_pos,
                    Q_table=Q_table_agent2,
                    epsilon=ql_params.exploration_rate,
                    last_state=last_state_agent2,
                    actions=ql_params.actions,
                    grid_size=grid_settings.size
                )
                if action_agent2 == -1:
                    logging.debug(f"Episode {episode}: Agent 2 has no valid actions.")
                    break
                dx2, dy2 = ql_params.actions[action_agent2]
                new_agent2_pos = (agent2_pos[0] + dx2, agent2_pos[1] + dy2)
                if (new_agent2_pos != agent2_pos and new_agent2_pos != last_state_agent2 and
                    0 <= new_agent2_pos[0] < grid_settings.size and 0 <= new_agent2_pos[1] < grid_settings.size):
                    last_state_agent2 = agent2_pos
                    agent2_pos = new_agent2_pos
                    reward_agent2 = calculate_reward(
                        agent_pos=agent2_pos,
                        other_agent_pos=agent1_pos, 
                        victory_zone=grid_settings.agent2_victory_zone,
                        critical_zone=critical_zone,
                        last_state=last_state_agent2,
                        visited_states=visited_states_agent2,
                        has_reached_goal=agent2_reached_goal,
                        rewards=rewards,
                        penalties=penalties
                    )
                    total_reward_agent2 += reward_agent2
                    episode_path_agent2.append(agent2_pos)
                    cumulative_rewards_agent2.append(total_reward_agent2)
                else:
                    cumulative_rewards_agent2.append(total_reward_agent2)
                if agent2_pos == grid_settings.agent2_victory_zone:
                    agent2_reached_goal = True
                    logging.debug(f"Episode {episode}: Agent 2 reached its victory zone.")

            # --- Agent 3 action ---
            if not agent3_reached_goal:
                current_state_agent3 = agent3_pos
                action_agent3 = epsilon_greedy_action_selection(
                    state=agent3_pos,
                    Q_table=Q_table_agent3,
                    epsilon=ql_params.exploration_rate,
                    last_state=last_state_agent3,
                    actions=ql_params.actions,
                    grid_size=grid_settings.size
                )
                if action_agent3 == -1:
                    logging.debug(f"Episode {episode}: Agent 3 has no valid actions.")
                    break
                dx3, dy3 = ql_params.actions[action_agent3]
                new_agent3_pos = (agent3_pos[0] + dx3, agent3_pos[1] + dy3)
                if (new_agent3_pos != agent3_pos and new_agent3_pos != last_state_agent3 and
                    0 <= new_agent3_pos[0] < grid_settings.size and 0 <= new_agent3_pos[1] < grid_settings.size):
                    last_state_agent3 = agent3_pos
                    agent3_pos = new_agent3_pos
                    reward_agent3 = calculate_reward(
                        agent_pos=agent3_pos,
                        other_agent_pos=agent1_pos, 
                        victory_zone=grid_settings.agent3_victory_zone,
                        critical_zone=critical_zone,
                        last_state=last_state_agent3,
                        visited_states=visited_states_agent3,
                        has_reached_goal=agent3_reached_goal,
                        rewards=rewards,
                        penalties=penalties
                    )
                    total_reward_agent3 += reward_agent3
                    episode_path_agent3.append(agent3_pos)
                    cumulative_rewards_agent3.append(total_reward_agent3)
                else:
                    cumulative_rewards_agent3.append(total_reward_agent3)
                if agent3_pos == grid_settings.agent3_victory_zone:
                    agent3_reached_goal = True
                    logging.debug(f"Episode {episode}: Agent 3 reached its victory zone.")

            
            if (manhattan_distance(agent1_pos, agent2_pos) < 1.5 or
                manhattan_distance(agent1_pos, agent3_pos) < 1.5 or
                manhattan_distance(agent2_pos, agent3_pos) < 1.5):
                collision_count += 1
                collision_penalty = penalties.collision
                total_reward_agent1 += collision_penalty
                total_reward_agent2 += collision_penalty
                total_reward_agent3 += collision_penalty
                cumulative_rewards_agent1[-1] += collision_penalty
                cumulative_rewards_agent2[-1] += collision_penalty
                cumulative_rewards_agent3[-1] += collision_penalty
                collision_paths.append(CollisionPath(
                    agent1_run=(episode_path_agent1, cumulative_rewards_agent1.copy()),
                    agent2_run=(episode_path_agent2, cumulative_rewards_agent2.copy()),
                    agent3_run=(episode_path_agent3, cumulative_rewards_agent3.copy())
                ))
                logging.info(f"Episode {episode}: Collision occurred among agents at positions {agent1_pos}, {agent2_pos}, {agent3_pos}.")
                break
            else:
                if not agent1_reached_goal and action_agent1 != -1:
                    update_q_table(Q_table=Q_table_agent1, state=current_state_agent1, action=action_agent1,
                                   reward=reward_agent1, new_state=agent1_pos,
                                   learning_rate=ql_params.learning_rate, discount_factor=ql_params.discount_factor)
                if not agent2_reached_goal and action_agent2 != -1:
                    update_q_table(Q_table=Q_table_agent2, state=current_state_agent2, action=action_agent2,
                                   reward=reward_agent2, new_state=agent2_pos,
                                   learning_rate=ql_params.learning_rate, discount_factor=ql_params.discount_factor)
                if not agent3_reached_goal and action_agent3 != -1:
                    update_q_table(Q_table=Q_table_agent3, state=current_state_agent3, action=action_agent3,
                                   reward=reward_agent3, new_state=agent3_pos,
                                   learning_rate=ql_params.learning_rate, discount_factor=ql_params.discount_factor)

            if agent1_reached_goal and agent2_reached_goal and agent3_reached_goal:
                success_count += 1
                successful_paths.append(SuccessfulPath(
                    agent1_run=(episode_path_agent1, cumulative_rewards_agent1.copy()),
                    agent2_run=(episode_path_agent2, cumulative_rewards_agent2.copy()),
                    agent3_run=(episode_path_agent3, cumulative_rewards_agent3.copy())
                ))
                logging.info(f"Episode {episode}: All agents reached their victory zones.")
                break

            if total_reward_agent1 < penalties.min_cumulative or total_reward_agent2 < penalties.min_cumulative or total_reward_agent3 < penalties.min_cumulative:
                logging.info(f"Episode {episode}: Terminated due to low cumulative rewards.")
                break

        

        reward_stats["agent1"]["rewards"].append(total_reward_agent1)
        reward_stats["agent2"]["rewards"].append(total_reward_agent2)
        reward_stats["agent3"]["rewards"].append(total_reward_agent3)
         
        max_q1 = np.max(Q_table_agent1, axis=2); mask1 = total_visits_agent1 > 0
        max_q2 = np.max(Q_table_agent2, axis=2); mask2 = total_visits_agent2 > 0
        max_q3 = np.max(Q_table_agent3, axis=2); mask3 = total_visits_agent3 > 0

        vals = []
        if mask1.any(): vals.append(np.mean(max_q1[mask1]))
        if mask2.any(): vals.append(np.mean(max_q2[mask2]))
        if mask3.any(): vals.append(np.mean(max_q3[mask3]))
        avg_max_q_history.append(np.mean(vals) if vals else 0)

    
        success_rate = success_count / episode
        success_rate_history.append(success_rate)

        
        ql_params.exploration_rate = max(
            ql_params.min_exploration_rate,
            ql_params.exploration_rate * ql_params.exploration_decay
        )
        for (x, y), count in visited_states_agent1.items():
            total_visits_agent1[x, y] += count
        for (x, y), count in visited_states_agent2.items():
            total_visits_agent2[x, y] += count
        for (x, y), count in visited_states_agent3.items():
            total_visits_agent3[x, y] += count
        if episode % 100 == 0:
            logging.info(f"Episode {episode}/{simulation_params.episodes} completed. "
                         f"Successes: {success_count}, Collisions: {collision_count}, "
                         f"Exploration Rate: {ql_params.exploration_rate:.4f}")



def animate_simulation(paths, title):
    """
    Animate the movement of three agents in a run.
    """
    agent1_path, agent1_cumulative_rewards = paths.agent1_run
    agent2_path, agent2_cumulative_rewards = paths.agent2_run
    agent3_path, agent3_cumulative_rewards = paths.agent3_run

    max_frames = max(len(agent1_path), len(agent2_path), len(agent3_path))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, grid_settings.size)
    ax.set_ylim(-1, grid_settings.size)
    ax.set_xticks(range(grid_settings.size))
    ax.set_yticks(range(grid_settings.size))
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if len(critical_zone) > 0:
        critical_x, critical_y = zip(*critical_zone)
        ax.scatter(critical_y, critical_x, color="red", label="Critical Zone", s=100, marker="s", alpha=0.6)
    ax.scatter(grid_settings.agent1_victory_zone[1], grid_settings.agent1_victory_zone[0], color="green",
               label="Agent 1 Victory Zone", s=200, marker="*", edgecolors='black', linewidths=1.5)
    ax.scatter(grid_settings.agent2_victory_zone[1], grid_settings.agent2_victory_zone[0], color="blue",
               label="Agent 2 Victory Zone", s=200, marker="*", edgecolors='black', linewidths=1.5)
    ax.scatter(grid_settings.agent3_victory_zone[1], grid_settings.agent3_victory_zone[0], color="purple",
               label="Agent 3 Victory Zone", s=200, marker="*", edgecolors='black', linewidths=1.5)

    agent1_line, = ax.plot([], [], 'b-', lw=2, alpha=0.7)
    agent2_line, = ax.plot([], [], 'g-', lw=2, alpha=0.7)
    agent3_line, = ax.plot([], [], 'm-', lw=2, alpha=0.7)

    agent1_marker, = ax.plot([], [], 'bo', markersize=10, label="Agent 1 Position")
    agent2_marker, = ax.plot([], [], 'go', markersize=10, label="Agent 2 Position")
    agent3_marker, = ax.plot([], [], 'mo', markersize=10, label="Agent 3 Position")

    step_text = ax.text(0.5, -0.1, '', transform=ax.transAxes, fontsize=12, ha='center')
    reward_text_agent1 = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=10, ha='left', color='blue')
    reward_text_agent2 = ax.text(0.98, 0.92, '', transform=ax.transAxes, fontsize=10, ha='right', color='green')
    reward_text_agent3 = ax.text(0.5, 0.92, '', transform=ax.transAxes, fontsize=10, ha='center', color='magenta')
    steps_text_agent1 = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=10, ha='left', color='blue')
    steps_text_agent2 = ax.text(0.98, 0.85, '', transform=ax.transAxes, fontsize=10, ha='right', color='green')
    steps_text_agent3 = ax.text(0.5, 0.85, '', transform=ax.transAxes, fontsize=10, ha='center', color='magenta')
    collision_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, fontsize=12, ha='center', color='red', fontweight='bold')

    def update(frame):
        if frame < len(agent1_path):
            agent1_line.set_data([pos[1] for pos in agent1_path[:frame + 1]],
                                 [pos[0] for pos in agent1_path[:frame + 1]])
            agent1_marker.set_data([agent1_path[frame][1]], [agent1_path[frame][0]])
        if frame < len(agent2_path):
            agent2_line.set_data([pos[1] for pos in agent2_path[:frame + 1]],
                                 [pos[0] for pos in agent2_path[:frame + 1]])
            agent2_marker.set_data([agent2_path[frame][1]], [agent2_path[frame][0]])
        if frame < len(agent3_path):
            agent3_line.set_data([pos[1] for pos in agent3_path[:frame + 1]],
                                 [pos[0] for pos in agent3_path[:frame + 1]])
            agent3_marker.set_data([agent3_path[frame][1]], [agent3_path[frame][0]])

        step_text.set_text(f"Global Step: {frame + 1}")
        if frame < len(agent1_cumulative_rewards):
            reward_text_agent1.set_text(f"Agent 1 Reward: {round(agent1_cumulative_rewards[frame], 2)}")
            steps_text_agent1.set_text(f"Agent 1 Steps: {frame + 1}")
        if frame < len(agent2_cumulative_rewards):
            reward_text_agent2.set_text(f"Agent 2 Reward: {round(agent2_cumulative_rewards[frame], 2)}")
            steps_text_agent2.set_text(f"Agent 2 Steps: {frame + 1}")
        if frame < len(agent3_cumulative_rewards):
            reward_text_agent3.set_text(f"Agent 3 Reward: {round(agent3_cumulative_rewards[frame], 2)}")
            steps_text_agent3.set_text(f"Agent 3 Steps: {frame + 1}")

        if frame < len(agent1_path) and frame < len(agent2_path) and frame < len(agent3_path):
            d12 = manhattan_distance(agent1_path[frame], agent2_path[frame])
            d13 = manhattan_distance(agent1_path[frame], agent3_path[frame])
            d23 = manhattan_distance(agent2_path[frame], agent3_path[frame])
            collision_occurred = (d12 < 2 or d13 < 2 or d23 < 2)
        else:
            collision_occurred = False
        if collision_occurred:
            collision_text.set_text("Collision Occurred!")
        else:
            collision_text.set_text("")
        return (agent1_line, agent2_line, agent3_line, agent1_marker,
                agent2_marker, agent3_marker, step_text, reward_text_agent1,
                reward_text_agent2, reward_text_agent3, steps_text_agent1,
                steps_text_agent2, steps_text_agent3, collision_text)

    ax.legend(loc="upper right", fontsize=10, bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=max_frames, interval=400, repeat=False)
    gif_filename = "Three_Agents_Plots_Eps/" + title.replace(" ", "_") + ".gif"
    anim.save(gif_filename, writer='pillow', fps=2)
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
                cz[1], cz[0],
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
    fn = "Three_Agents_Plots_Eps/" + title.replace(" ", "_") + ".pdf"
    fig.savefig(fn, format="pdf", bbox_inches="tight")
    plt.close(fig)

def plot_heatmaps():
    total_visits_agent1[grid_settings.agent1_start] = success_count
    total_visits_agent2[grid_settings.agent2_victory_zone] = success_count
    total_visits_agent3[grid_settings.agent3_victory_zone] = success_count

    plot_heatmap(total_visits_agent1, grid_settings.size, "Agent 1 Visitation Frequencies", cmap='Blues', annotations=True)
    plot_heatmap(total_visits_agent2.T, grid_settings.size, "Agent 2 Visitation Frequencies", cmap='Greens', annotations=True)
    plot_heatmap(total_visits_agent3.T, grid_settings.size, "Agent 3 Visitation Frequencies", cmap='Oranges', annotations=True)

    max_q_agent1 = np.max(Q_table_agent1, axis=2)
    max_q_agent2 = np.max(Q_table_agent2, axis=2)
    max_q_agent3 = np.flip(np.max(Q_table_agent3, axis=2))

    plot_heatmap(max_q_agent1, grid_settings.size, "Agent 1 Maximum Q-values", cmap='Purples', annotations=True)
    plot_heatmap(max_q_agent2, grid_settings.size, "Agent 2 Maximum Q-values", cmap='Oranges', annotations=True)
    plot_heatmap(max_q_agent3, grid_settings.size, "Agent 3 Maximum Q-values", cmap='Greys', annotations=True)

def perform_risk_analysis():

    
    agent1_avg_reward = np.mean(reward_stats["agent1"]["rewards"]) if reward_stats["agent1"]["rewards"] else 0
    agent2_avg_reward = np.mean(reward_stats["agent2"]["rewards"]) if reward_stats["agent2"]["rewards"] else 0
    agent3_avg_reward = np.mean(reward_stats["agent3"]["rewards"]) if reward_stats["agent3"]["rewards"] else 0

   
    avg_success_steps = (np.mean([len(path.agent1_run[0]) for path in successful_paths])
                         if successful_paths else "No successful runs")
    
    
    if successful_paths:
        agent1_avg_success_reward = np.mean([path.agent1_run[1][-1] for path in successful_paths])
        agent2_avg_success_reward = np.mean([path.agent2_run[1][-1] for path in successful_paths])
        agent3_avg_success_reward = np.mean([path.agent3_run[1][-1] for path in successful_paths])
    else:
        agent1_avg_success_reward = "N/A"
        agent2_avg_success_reward = "N/A"
        agent3_avg_success_reward = "N/A"
    
    data = [
        ["Success Count", success_count],
        ["Collision Count", collision_count],
        ["Average Reward (Agent 1)", round(agent1_avg_reward, 2) if reward_stats["agent1"]["rewards"] else "N/A"],
        ["Average Reward (Agent 2)", round(agent2_avg_reward, 2) if reward_stats["agent2"]["rewards"] else "N/A"],
        ["Average Reward (Agent 3)", round(agent3_avg_reward, 2) if reward_stats["agent3"]["rewards"] else "N/A"],
        ["Average Accumulated Reward (Agent 1) Over Successful Runs", 
         round(agent1_avg_success_reward, 2) if isinstance(agent1_avg_success_reward, float) else agent1_avg_success_reward],
        ["Average Accumulated Reward (Agent 2) Over Successful Runs", 
         round(agent2_avg_success_reward, 2) if isinstance(agent2_avg_success_reward, float) else agent2_avg_success_reward],
        ["Average Accumulated Reward (Agent 3) Over Successful Runs", 
         round(agent3_avg_success_reward, 2) if isinstance(agent3_avg_success_reward, float) else agent3_avg_success_reward],
        ["Average Steps to Victory", avg_success_steps],
    ]
    
    print(tabulate(data, headers=["Metric", "Value"], tablefmt="pretty"))
    
   
    if successful_paths:
        agent1_final_rewards = [path.agent1_run[1][-1] for path in successful_paths]
        agent2_final_rewards = [path.agent2_run[1][-1] for path in successful_paths]
        agent3_final_rewards = [path.agent3_run[1][-1] for path in successful_paths]
        runs = range(1, len(successful_paths) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(runs, agent1_final_rewards, label='Agent 1', color='blue')
        plt.plot(runs, agent2_final_rewards, label='Agent 2', color='green')
        plt.plot(runs, agent3_final_rewards, label='Agent 3', color='magenta')
        plt.xlabel('Successful Run Number', fontsize=14)
        plt.ylabel('Final Cumulative Reward', fontsize=14)
        plt.title('Final Cumulative Rewards of Successful Runs', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No successful runs to plot cumulative rewards.")

def enhanced_risk_analysis(burn_in: int = 0):
    agents = ['agent1', 'agent2', 'agent3']

    analysis_data = []
    for agent in agents:
        rewards_full = np.array(reward_stats[agent]["rewards"])
        if rewards_full.size > burn_in:
            rewards_array = rewards_full[burn_in:]
            mean_val = np.mean(rewards_array)
            std_val = np.std(rewards_array)
            median_val = np.median(rewards_array)
            q25 = np.percentile(rewards_array, 25)
            q75 = np.percentile(rewards_array, 75)
            n = rewards_array.size
            ci_lower = mean_val - 1.96 * (std_val / np.sqrt(n))
            ci_upper = mean_val + 1.96 * (std_val / np.sqrt(n))
            analysis_data.append([
                agent,
                round(mean_val, 2),
                round(std_val, 2),
                round(median_val, 2),
                round(q25, 2),
                round(q75, 2),
                f"[{round(ci_lower, 2)}, {round(ci_upper, 2)}]"
            ])
        else:
            analysis_data.append([agent, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

    headers_rewards = ["Agent", "Mean", "Std Dev", "Median", "25th %", "75th %", "95% CI"]
    pretty_rewards = tabulate(analysis_data, headers=headers_rewards, tablefmt="pretty")
    latex_rewards  = tabulate(analysis_data, headers=headers_rewards, tablefmt="latex")

    print(f"Reward Distribution Metrics (burn-in={burn_in}) (Pretty):")
    print(pretty_rewards)
    print(f"\nReward Distribution Metrics (burn-in={burn_in}) (LaTeX):")
    print(latex_rewards)
    
    
    steps_data = []
    if successful_paths:
        for agent in agents:
            agent_steps = [len(getattr(path, f"{agent}_run")[0]) for path in successful_paths]
            steps_data.append([agent, round(np.mean(agent_steps),2), round(np.std(agent_steps),2)])
    else:
        steps_data = [["N/A", "N/A", "N/A"]]
    headers_steps = ["Agent", "Average Steps", "Std Dev"]
    pretty_steps = tabulate(steps_data, headers=headers_steps, tablefmt="pretty")
    latex_steps = tabulate(steps_data, headers=headers_steps, tablefmt="latex")
    print("\nAverage Steps to Victory (Pretty):")
    print(pretty_steps)
    print("\nAverage Steps to Victory (LaTeX):")
    print(latex_steps)
    
    
    qvalue_data = []
    max_q_agents = [
        np.max(Q_table_agent1, axis=2),
        np.max(Q_table_agent2, axis=2),
        np.max(Q_table_agent3, axis=2)
    ]
    for i, agent in enumerate(agents):
        avg_max_q = np.mean(max_q_agents[i])
        qvalue_data.append([agent, round(avg_max_q,2)])
    headers_qvalues = ["Agent", "Avg Max Q‑value"]
    pretty_qvalues = tabulate(qvalue_data, headers=headers_qvalues, tablefmt="pretty")
    latex_qvalues = tabulate(qvalue_data, headers=headers_qvalues, tablefmt="latex")
    print("\nAverage Maximum Q‑Values per State (Pretty):")
    print(pretty_qvalues)
    print("\nAverage Maximum Q‑Values per State (LaTeX):")
    print(latex_qvalues)
    
    # Visitation Entropy
    def compute_entropy(visits):
        total = np.sum(visits)
        if total == 0:
            return 0
        probs = visits.flatten() / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))
    visitation_data = []
    visit_matrices = [total_visits_agent1, total_visits_agent2, total_visits_agent3]
    for i, agent in enumerate(agents):
        entropy = compute_entropy(visit_matrices[i])
        visitation_data.append([agent, round(entropy,2)])
    headers_visitation = ["Agent", "Entropy"]
    pretty_visitation = tabulate(visitation_data, headers=headers_visitation, tablefmt="pretty")
    latex_visitation = tabulate(visitation_data, headers=headers_visitation, tablefmt="latex")
    print("\nVisitation Entropy (Pretty):")
    print(pretty_visitation)
    print("\nVisitation Entropy (LaTeX):")
    print(latex_visitation)
    
    # Success Rate
    success_rate = len(successful_paths) / simulation_params.episodes
    print("\nSuccess Rate:")
    print(f"Success Rate: {success_rate:.4f} ({len(successful_paths)} successes over {simulation_params.episodes} episodes)")

def plot_learning_curves():

    plt.figure(figsize=(12,8))
    agents = ['agent1', 'agent2', 'agent3']
    for agent in agents:
        rewards_array = np.array(reward_stats[agent]["rewards"])
        if rewards_array.size > 0:
            window = 65
            cumsum_vec = np.cumsum(np.insert(rewards_array, 0, 0))
            moving_avg = (cumsum_vec[window:] - cumsum_vec[:-window]) / window
            plt.plot(moving_avg, label=f"{agent} Moving Avg")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward (Moving Avg)")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Three_Agents_Plots_Eps/Learning_Curves.pdf", format="pdf")
    plt.show()

def plot_convergence():
    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(avg_max_q_history) + 1)

    fig, ax1 = plt.subplots(figsize=(10,6))

    
    ax1.plot(episodes, avg_max_q_history, label="Avg Max Q (visited)",
             linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Avg Max Q", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

   
    ax2 = ax1.twinx()
    ax2.plot(episodes, success_rate_history, label="Success Rate",
             color="tab:orange", linestyle="--", linewidth=2)
    ax2.set_ylabel("Success Rate", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1.0)

    
    lines, labels = ax1.get_lines()+ax2.get_lines(), [l.get_label() for l in ax1.get_lines()+ax2.get_lines()]
    ax1.legend(lines, labels, loc="lower right")

    plt.title("Convergence of Q‑values vs. Success Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_episodes_to_threshold(threshold: float=0.8) -> [int]:
    for i, val in enumerate(success_rate_history, start=1):
        if val>=threshold: return i
    return None

def compute_correlations():
   
    if len(avg_max_q_history)<2:
        print("Not enough data for correlation.")
        return
    pearson_r, _ = pearsonr(avg_max_q_history, success_rate_history)
    spearman_rho, _ = spearmanr(avg_max_q_history, success_rate_history)
    print(f"Pearson correlation (Q vs Success): {pearson_r:.3f}")
    print(f"Spearman correlation (Q vs Success): {spearman_rho:.3f}")

def compute_r2():
    
    X = np.array(avg_max_q_history).reshape(-1, 1)
    y = np.array(success_rate_history)
    
    if len(X) < 2:
        print("Not enough data for R².")
        return
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"Coefficient of Determination (R²): {r2:.3f}")

def compute_success_auc_sklearn():
    """Compute AUC of success_rate_history via scikit-learn’s auc."""
    if len(success_rate_history) < 2:
        print("Not enough data for AUC.")
        return
    
    episodes = np.arange(1, len(success_rate_history) + 1)
    auc_val = auc(episodes, success_rate_history)
    print(f"Area under success‑curve (sklearn): {auc_val:.3f}")

def main():
    global successful_paths, collision_paths
    train_agents()

    if successful_paths:
        most_successful = max(successful_paths, key=lambda x: (x.agent1_run[1][-1] + x.agent2_run[1][-1] + x.agent3_run[1][-1]))
        animate_simulation(most_successful, "Most Successful Run")
    if successful_paths:
        least_successful = min(successful_paths, key=lambda x: (x.agent1_run[1][-1] + x.agent2_run[1][-1] + x.agent3_run[1][-1]))
        animate_simulation(least_successful, "Least Successful Run")
    if collision_paths:
        animate_simulation(collision_paths[0], "Collision Scenario")
    
    #perform_risk_analysis()
    #enhanced_risk_analysis(burn_in=700)
    #plot_convergence()
    #plot_heatmaps()
    #plot_learning_curves()
    #compute_success_auc_sklearn()
    #compute_correlations()
    #compute_r2() 
if __name__ == "__main__":
    main()
