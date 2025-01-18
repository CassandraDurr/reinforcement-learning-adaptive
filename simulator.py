from agents import QLearningAgent, DynaQAgent, DynaQPlusAgent
from environment import AdaptiveGridWorld
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Simulator:
    def __init__(
        self,
        env: AdaptiveGridWorld,
        agent: QLearningAgent | DynaQAgent | DynaQPlusAgent,
        episodes=500,
        steps_per_episode=100,
        store_heatmaps: bool = True,
    ):
        """Initialises the simulator."""
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.total_rewards = []
        self.state_action_coverage = []
        self.completion_count = []
        self.steps_to_completion = []
        self.death_count = []
        self.td_errors = []
        # Track policy metrics
        self.policy_entropies = []
        self.policy_changes = []

        #  Store Q-tables for each episode
        self.store_heatmaps = store_heatmaps
        self.episode_action_values = (
            []
        )  # list of [ state_size x action_size ] snapshots

    def get_q_table(self):
        """Fetches the Q-table for the current agent."""
        if hasattr(self.agent, "q_table"):
            return self.agent.q_table
        else:
            raise AttributeError("The agent does not have a Q-table.")

    def compute_policy_entropy(self):
        """Computes the average policy entropy for Q-learning across all states."""
        entropies = []
        for state in range(self.agent.state_size):
            q_values = self.agent.q_table[state]
            exp_q_values = np.exp(q_values)  # Exponentiate Q-values
            probabilities = exp_q_values / np.sum(
                exp_q_values
            )  # Normalise to probabilities
            entropy = -np.sum(
                probabilities * np.log(probabilities + 1e-10)
            )  # Compute entropy
            entropies.append(entropy)
        return np.mean(entropies)

    def track_policy_changes(self):
        """Tracks policy changes and entropy if applicable."""
        new_policy = np.argmax(self.agent.q_table, axis=1)
        self.policy_change_count = np.sum(self.agent.previous_policy != new_policy)
        self.policy_changes.append(self.policy_change_count)
        self.agent.previous_policy = new_policy
        self.policy_entropies.append(self.compute_policy_entropy())

    def run(self, **kwargs):
        """Runs the simulation for a given strategy."""
        visited_state_actions = set()  # For state-action coverage

        for episode_idx in range(self.episodes):
            self.env.apply_changes(episode_idx)  # Adapt environment
            self.initialise_episode_statistics(episode_idx)

            completed, death, steps = self.run_episode(kwargs, visited_state_actions)

            self.update_metrics(completed, death, steps, visited_state_actions)

        return self.total_rewards

    def initialise_episode_statistics(self, episode_idx):
        """Initialises statistics for a new episode."""
        self.env.reset()
        self.current_state_index = 0
        self.current_total_reward = 0
        self.episode_td_errors = []
        self.policy_change_count = 0

    def run_episode(self, kwargs, visited_state_actions):
        """Runs a single episode."""
        completed, death, steps = False, False, 0
        state_index = self.current_state_index

        for _ in range(self.steps_per_episode):
            action = self.choose_action(state_index, kwargs)
            visited_state_actions.add((state_index, action))
            self.agent.state_action_visits[state_index, action] += 1

            next_state, reward, done, status = self.env.step(action)
            next_state_index = self.get_state_index(next_state)

            td_error = self.agent.update(
                state_index, action, reward, next_state_index, done
            )
            self.track_td_errors(td_error)

            state_index = next_state_index
            self.current_total_reward += reward
            steps += 1

            if done:
                completed, death = self.handle_done_status(status)
                break

        self.current_state_index = state_index
        return completed, death, steps

    def choose_action(self, state_index, kwargs):
        """Chooses an action based on the agent's strategy."""
        return self.agent.choose_action(state_index, **kwargs)

    def get_state_index(self, state):
        """Calculates the state index from the environment state."""
        return state[0] * self.env.size[1] + state[1]

    def track_td_errors(self, td_error):
        """Tracks TD errors if applicable."""
        self.episode_td_errors.append(abs(td_error))

    def handle_done_status(self, status):
        """Handles the 'done' status of the episode."""
        if status == "completion":
            return True, False
        elif status == "death":
            return False, True
        return False, False

    def update_metrics(self, completed, death, steps, visited_state_actions):
        """Updates metrics after an episode."""
        self.total_rewards.append(self.current_total_reward)
        self.state_action_coverage.append(
            len(visited_state_actions)
            / (self.agent.state_size * self.agent.action_size)
        )
        self.completion_count.append(1 if completed else 0)
        self.death_count.append(1 if death else 0)
        self.steps_to_completion.append(steps if completed else self.steps_per_episode)
        self.td_errors.append(np.mean(self.episode_td_errors))
        self.track_policy_changes()

        if self.store_heatmaps:
            self.episode_action_values.append(np.copy(self.agent.q_table))

    def plot_policy_at_episode(self, episode):
        """
        Plots the agent's policy (best actions) at a specific episode.

        Args:
            episode (int): The episode index for which to plot the policy.
        """
        if not self.store_heatmaps or episode >= len(self.episode_action_values):
            print("Policy data not available for this episode.")
            return

        # Initialise goal and lava pools based on initial setup
        goal = self.env.initial_goal
        lava_pools = set(self.env.initial_lava_pools)
        small_rewards = set(self.env.initial_small_reward_blocks)

        # Apply environment changes up to the given episode
        for ep, changes in self.env.changes.items():
            if ep > episode:
                break
            if "goal" in changes:
                goal = changes["goal"]
            if "add_lava_pools" in changes:
                lava_pools.update(changes["add_lava_pools"])
            if "remove_lava_pools" in changes:
                lava_pools.difference_update(changes["remove_lava_pools"])
            if "add_small_reward_blocks" in changes:
                small_rewards.update(changes["add_small_reward_blocks"])
            if "remove_small_reward_blocks" in changes:
                small_rewards.difference_update(changes["remove_small_reward_blocks"])

        # Extract the action values (Q) for the given episode
        action_values = self.episode_action_values[episode]
        policy = np.argmax(action_values, axis=1)  # Greedy action per state

        grid_size = self.env.size
        actions = ["↑", "↓", "←", "→"]
        policy_grid = np.full(grid_size, fill_value=" ", dtype="<U2")

        # Map the best action for each state onto the grid
        for state in range(len(policy)):
            row = state // grid_size[1]
            col = state % grid_size[1]
            if np.all(
                action_values[state] == action_values[state][0]
            ):  # All values equal
                policy_grid[row, col] = "•"
            else:
                policy_grid[row, col] = actions[policy[state]]

        # Plot the policy grid
        # Calculate average action value for coloring
        avg_action_values = np.mean(action_values, axis=1).reshape(grid_size)
        fig, ax = plt.subplots(figsize=(5, 5))
        cax = ax.imshow(avg_action_values, cmap="Greys", alpha=0.5)
        fig.colorbar(cax, ax=ax, label="Average Action Value")

        # Add action symbols
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                if ((row, col) not in lava_pools) and ((row, col) != goal):
                    ax.text(
                        col,
                        row,
                        policy_grid[row, col],
                        ha="center",
                        va="center",
                        fontsize=14,
                        color="black",
                    )

        # Highlight lava pools and goal dynamically for the episode
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                if (row, col) in lava_pools:
                    rect = plt.Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        fill=True,
                        color="red",
                        linewidth=2,
                    )
                    ax.add_patch(rect)
                elif (row, col) == goal:
                    rect = plt.Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        fill=True,
                        color="green",
                        linewidth=2,
                    )
                    ax.add_patch(rect)
                elif (row, col) in small_rewards:
                    rect = plt.Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="green",
                        linewidth=2,
                    )
                    ax.add_patch(rect)

        plt.title(f"Policy at Episode {episode}")
        plt.xticks(range(grid_size[1]))
        plt.yticks(range(grid_size[0]))
        plt.tight_layout()
        plt.show()

    def plot_time_series_per_change(self):
        """
        Plots time series of death rates and completion rates normalized per change block.
        Marks environment changes on the plot.
        """
        if not hasattr(self.env, "changes") or not self.env.changes:
            print("No environment changes to segment data.")
            return

        # Identify change points (start, change episodes, end)
        change_episodes = sorted(self.env.changes.keys())
        change_points = [0] + change_episodes + [len(self.death_count)]

        plt.figure(figsize=(10, 6))

        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]

            # Extract deaths and completions for the block
            block_deaths_raw = self.death_count[start:end]
            block_completions_raw = self.completion_count[start:end]

            # Calculate cumulative values for the block
            block_deaths = np.cumsum(block_deaths_raw)
            block_completions = np.cumsum(block_completions_raw)
            block_length = np.arange(1, end - start + 1)

            # Calculate rates for the block
            death_rate = block_deaths / block_length
            completion_rate = block_completions / block_length

            # Plot the time series for this block
            plt.plot(np.arange(start, end), death_rate, color="dimgrey", linewidth=2)
            plt.plot(np.arange(start, end), completion_rate, color="blue", linewidth=2)

            # Mark the start of the block
            if i > 0:  # Skip the very first episode
                plt.axvline(
                    x=start,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                )

        plt.xlabel("Episode")
        plt.ylabel("Rate")
        plt.title("Per-Change Block Death and Completion Rates Over Time")
        plt.tight_layout()
        plt.show()

    def plot_death_and_completion_rates_with_changes(self):
        """
        Plots death rate and completion rate with markers for environment changes
        on different y-axes with color-coded axes.
        """
        episodes = range(len(self.death_count))
        cumulative_deaths = np.cumsum(self.death_count)
        cumulative_completions = np.cumsum(self.completion_count)

        # Calculate death and completion rates
        death_rate = cumulative_deaths / (np.arange(1, len(self.death_count) + 1))
        completion_rate = cumulative_completions / (
            np.arange(1, len(self.completion_count) + 1)
        )

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot death rate on the first y-axis
        ax1.plot(episodes, death_rate, label="Death Rate", color="dimgrey", linewidth=2)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Death Rate", color="dimgrey")
        ax1.tick_params(axis="y", labelcolor="dimgrey")

        # Create a second y-axis for the completion rate
        ax2 = ax1.twinx()
        ax2.plot(
            episodes,
            completion_rate,
            label="Completion Rate",
            color="blue",
            linewidth=2,
        )
        ax2.set_ylabel("Completion Rate", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        # Mark environment changes
        if hasattr(self.env, "changes"):
            change_episodes = sorted(self.env.changes.keys())
            for ch_ep in change_episodes:
                ax1.axvline(x=ch_ep, color="red", linestyle="--", alpha=0.7)

        plt.title("Death Rate and Completion Rate with Environment Changes")
        fig.tight_layout()
        plt.show()

    def plot_metrics(self):
        """Plots all metrics in a single figure with multiple subplots."""

        if hasattr(self.env, "changes") or self.env.changes:

            # Identify change points (start, change episodes, end)
            change_episodes = sorted(self.env.changes.keys())

            _, axes = plt.subplots(3, 2, figsize=(12, 9))

            # Plot rewards
            axes[0, 0].plot(
                range(1, self.episodes + 1),
                self.total_rewards,
                label="Total Rewards",
                color="blue",
            )
            for chng in change_episodes:
                axes[0, 0].axvline(
                    x=chng,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                )
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Total Reward")
            axes[0, 0].set_title("Total Rewards per Episode")
            axes[0, 0].grid(axis="y", color="#dfdfdf")

            # Plot TD errors
            axes[0, 1].plot(
                range(1, self.episodes + 1),
                self.td_errors,
                label="Average TD Error",
                color="green",
            )
            for chng in change_episodes:
                axes[0, 1].axvline(
                    x=chng,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                )
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("TD Error")
            axes[0, 1].set_title("Average TD Error per Episode")
            axes[0, 1].grid(axis="y", color="#dfdfdf")

            # Plot cumulative completions
            cumulative_completions = np.cumsum(self.completion_count)
            completion_rates = cumulative_completions / np.arange(1, self.episodes + 1)
            axes[1, 0].plot(
                range(1, self.episodes + 1),
                completion_rates,
                label="Completion Rate",
                color="orange",
            )
            for chng in change_episodes:
                axes[1, 0].axvline(
                    x=chng,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                )
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Completion Rate")
            axes[1, 0].set_title("Completion Rate per Episode")
            axes[1, 0].set_ylim(0, 1)  # Ensure the completion axis goes from 0 to 1
            axes[1, 0].grid(axis="y", color="#dfdfdf")

            # Plot state-action coverage
            axes[1, 1].plot(
                range(1, self.episodes + 1),
                self.state_action_coverage,
                label="State-Action Coverage",
                color="purple",
            )
            for chng in change_episodes:
                axes[1, 1].axvline(
                    x=chng,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                )
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Coverage")
            axes[1, 1].set_title("State-Action Coverage per Episode")
            axes[1, 1].set_ylim(0, 1)  # Ensure the coverage axis goes from 0 to 1
            axes[1, 1].grid(axis="y", color="#dfdfdf")

            # Plot policy changes
            axes[2, 0].plot(
                range(1, self.episodes + 1),
                self.policy_changes,
                label="Policy Changes",
                color="indigo",
            )
            for chng in change_episodes:
                axes[2, 0].axvline(
                    x=chng,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                )
            axes[2, 0].set_xlabel("Episode")
            axes[2, 0].set_ylabel("Policy Changes")
            axes[2, 0].set_title("Policy Changes per Episode")
            axes[2, 0].grid(axis="y", color="#dfdfdf")

            # Plot policy entropy
            axes[2, 1].plot(
                range(1, self.episodes + 1),
                self.policy_entropies,
                label="Policy Entropy",
                color="teal",
            )
            for chng in change_episodes:
                axes[2, 1].axvline(
                    x=chng,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                )
            axes[2, 1].set_xlabel("Episode")
            axes[2, 1].set_ylabel("Entropy")
            axes[2, 1].set_title("Policy Entropy per Episode")
            axes[2, 1].grid(axis="y", color="#dfdfdf")

            # Adjust layout and show the figure
            plt.tight_layout()
            plt.show()

        else:
            print("No environment changes to segment data.")

    def plot_agent_value_heatmaps(self, filename="q_learning"):
        """
        Plots the agent's action-value estimates (could be Q-values or mu-values).
        If the agent also has 'std' (Thompson), optionally plot a separate figure of std.
        """
        # -- First, plot the main values (Q or mu) --
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        actions = ["↑", "↓", "←", "→"]
        grid_size = self.env.size
        action_values = self.agent.q_table  # Q for Q-learning, mu for Thompson

        for i in range(self.agent.action_size):
            heatmap_data = np.zeros(grid_size)
            for state in range(self.agent.state_size):
                row = state // grid_size[1]
                col = state % grid_size[1]
                heatmap_data[row, col] = action_values[state, i]

            ax = axes[i // 2, i % 2]
            cax = ax.imshow(heatmap_data, cmap="YlOrRd_r", interpolation="nearest")
            fig.colorbar(cax, ax=ax)
            ax.set_title(f"Action Value Heatmap: {actions[i]}")

            # Add annotations
            for row in range(grid_size[0]):
                for col in range(grid_size[1]):
                    ax.text(
                        col,
                        row,
                        f"{heatmap_data[row,col]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

            # Add outlines for lava_pools and goal
            for row in range(grid_size[0]):
                for col in range(grid_size[1]):
                    # If it's a lava
                    if (row, col) in self.env.lava_pools:
                        rect = plt.Rectangle(
                            (col - 0.5, row - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="black",
                            linewidth=2,
                        )
                        ax.add_patch(rect)

                    # If it's the goal cell
                    if (row, col) == self.env.goal:
                        rect = plt.Rectangle(
                            (col - 0.5, row - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="green",
                            linewidth=2,
                        )
                        ax.add_patch(rect)

        plt.suptitle("Estimated Action Values (Q or mu)")
        plt.tight_layout()
        plt.show()

    def plot_action_frequency_heatmaps(self):
        """Plots 2x2 heatmaps for action frequencies of each action."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))

        actions = ["↑", "↓", "←", "→"]
        grid_size = self.env.size
        action_visits = self.agent.state_action_visits

        for i, action in enumerate(range(4)):
            heatmap_data = np.zeros(grid_size)
            for state in range(self.agent.state_size):
                row = state // grid_size[1]
                col = state % grid_size[1]
                heatmap_data[row, col] = action_visits[state, action]

            ax = axes[i // 2, i % 2]
            cax = ax.imshow(heatmap_data, cmap="Blues", interpolation="nearest")
            fig.colorbar(cax, ax=ax)
            ax.set_title(f"Action Frequency Heatmap for {actions[i]}")

            # Add text annotations for action frequencies
            for row in range(grid_size[0]):
                for col in range(grid_size[1]):
                    ax.text(
                        col,
                        row,
                        f"{heatmap_data[row, col]:.0f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

            # Add outlines for lava_pools and goal
            for row in range(grid_size[0]):
                for col in range(grid_size[1]):
                    # If it's a lava
                    if (row, col) in self.env.lava_pools:
                        rect = plt.Rectangle(
                            (col - 0.5, row - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="black",
                            linewidth=2,
                        )
                        ax.add_patch(rect)

                    # If it's the goal cell
                    if (row, col) == self.env.goal:
                        rect = plt.Rectangle(
                            (col - 0.5, row - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="green",
                            linewidth=2,
                        )
                        ax.add_patch(rect)

        plt.tight_layout()
        plt.show()

    def animate_agent_value_heatmaps(self, interval=5, filename="q_learning"):
        """
        Creates an animation of the agent's action-value heatmaps over episodes.
        'interval' is the delay between frames in milliseconds.
        """
        actions = ["↑", "↓", "←", "→"]
        grid_size = self.env.size
        num_episodes = len(self.episode_action_values)  # how many snapshots

        # -- 4 subplots for each action if action_size=4 --
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        fig.suptitle("Action-Value Heatmaps Over Time")

        ims = []
        for i in range(self.agent.action_size):
            # Initialise empty heatmap
            heatmap_data = np.zeros(grid_size)
            ax = axes[i // 2, i % 2]
            im = ax.imshow(
                heatmap_data, cmap="YlOrRd_r", interpolation="nearest", vmin=-5, vmax=10
            )
            ax.set_title(f"Action {actions[i]}")
            fig.colorbar(im, ax=ax)
            ims.append(im)

        # -- Define init function --
        def init():
            """Set initial frame's data."""
            for i in range(self.agent.action_size):
                # Just fill with zeros or something
                ims[i].set_data(np.zeros(grid_size))
            return ims

        # -- Define the update function for each frame i --
        def update(frame):
            """Update heatmap data for episode=frame."""
            action_values = self.episode_action_values[
                frame
            ]  # shape [state_size, action_size]
            # For each action, fill the heatmap with the correct values
            for a in range(self.agent.action_size):
                heatmap_data = np.zeros(grid_size)
                for state in range(self.agent.state_size):
                    row = state // grid_size[1]
                    col = state % grid_size[1]
                    heatmap_data[row, col] = action_values[state, a]
                ims[a].set_data(heatmap_data)

            fig.suptitle(f"Action-Value Heatmaps: Episode {frame+1}/{num_episodes}")
            return ims

        # Create the animation
        anim = FuncAnimation(
            fig,
            update,
            frames=num_episodes,
            init_func=init,
            interval=interval,
            blit=False,
        )

        anim.save(f"heatmaps_animation_{filename}.gif", writer="imagemagick", fps=1)
