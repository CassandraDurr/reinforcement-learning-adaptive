"""File containing the re-inforcement learning agents code."""

import random
from collections import defaultdict
from queue import PriorityQueue

import numpy as np


class QLearningAgent:
    """Q-Learning agent with epsilon-greedy action selection."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.99,
        epsilon_min=0.01,
        epsilon_decay="linear",  # or 'exponential'
        episodes=500,
    ):
        """Initialise Q-Learning agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((state_size, action_size))

        # Epsilon decay parameters
        self.epsilon_max = epsilon
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_type = epsilon_decay
        self.episodes = episodes
        self.current_episode = 0  # Track current episode

        # Get decay rate for exponential decay
        if epsilon_decay == "exponential":
            self.decay_rate = (epsilon_min / epsilon) ** (1 / episodes)

        # Visit counts
        self.state_visits = np.zeros(state_size)  # N(s)
        self.state_action_visits = np.zeros((state_size, action_size))  # N(s, a)

        # Track previous policy for comparison
        self.previous_policy = np.argmax(self.q_table, axis=1)  # Greedy policy

    def choose_action(self, state: int):
        """Select an action based on the exploration strategy and updates epsilon."""
        # Update epsilon dynamically
        self.update_epsilon()

        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def update_epsilon(self):
        """Update epsilon based on the decay type."""
        if self.epsilon_decay_type == "linear":
            decay_step = (self.epsilon_max - self.epsilon_min) / self.episodes
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_step)
        elif self.epsilon_decay_type == "exponential":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

    def update(self, state: int, action: int, reward: int, next_state: int, done: bool):
        """Update the Q-value table."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (
            0
            if done
            else self.discount_factor * self.q_table[next_state, best_next_action]
        )
        # Calculate TD error
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

        return td_error

    def display_q_table(self, grid_size: tuple[int, int], actions: list[str] = None):
        """Display the Q-table in a grid format for easier visualization.

        Args:
            grid_size (tuple[int, int]): Size of the grid-world (rows, cols).
            actions (list[str]): Symbols representing the actions.
        """
        if actions is None:
            actions = ["↑", "↓", "←", "→"]

        print("\nQ-Table Visualization:\n")
        for row in range(grid_size[0]):
            for col in range(grid_size[1]):
                state = row * grid_size[1] + col
                q_values = self.q_table[state]
                best_action = np.argmax(q_values)
                print(f"{actions[best_action]} {q_values}", end="\t")
            print("\n")

    def display_state_q_values(
        self,
        state: tuple[int, int],
        grid_size: tuple[int, int],
        actions: list[str] = None,
    ):
        """Display the Q-values and the best action for a single state.

        Args:
            state (tuple[int, int]): State to inspect (row, col).
            grid_size (tuple[int, int]): Size of the grid-world.
            actions (list[str]): Symbols representing the actions.
        """
        if actions is None:
            actions = ["↑", "↓", "←", "→"]

        state_index = (
            state[0] * grid_size[1] + state[1]
        )  # Map (row, col) to state index
        q_values = self.q_table[state_index]  # Retrieve Q-values
        best_action = np.argmax(q_values)  # Find the action with the highest Q-value

        print(f"\nState: {state} (Index: {state_index})")
        print("Q-Values for Actions:")
        for i, action in enumerate(actions):
            print(f"{action}: {q_values[i]:.4f}")
        print(
            f"\nBest Action: {actions[best_action]} (Q-Value: {q_values[best_action]:.4f})"
        )


class DynaQAgent(QLearningAgent):
    """Tabular Dyna-Q agent capable of forgetfulness and prioritised sweeping."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate=0.1,
        discount_factor=0.99,
        planning_steps=10,
        epsilon=0.99,
        epsilon_min=0.01,
        epsilon_decay="linear",  # or 'exponential'
        episodes=500,
        forgetful_bool=True,
        prioritised_sweeping=False,
        priority_threshold=0.1,
        forgetting_threshold=1000,  # Forget transitions older than this many updates
    ):
        """Initialise agent."""
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            episodes=episodes,
        )

        self.planning_steps = planning_steps
        self.prioritised_sweeping = prioritised_sweeping
        self.priority_threshold = priority_threshold
        self.forgetting_threshold = forgetting_threshold
        self.forget = forgetful_bool

        # Q-table
        self.q_table = np.zeros((state_size, action_size))

        # Model as a dictionary: (state, action) -> (reward, next_state, timestamp)
        self.model = defaultdict(lambda: (0, None, 0))  # Default timestamp is 0

        # Priority queue for prioritised sweeping
        self.priority_queue = PriorityQueue() if prioritised_sweeping else None

        # Track the current update step
        self.current_step = 0

    def update(self, state, action, reward, next_state, done):
        """Update Q-table, model, and perform planning."""
        self.current_step += 1  # Increment the update step

        # Q-learning update
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (
            0
            if done
            else self.discount_factor * self.q_table[next_state, best_next_action]
        )
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

        # Update the model with the current timestamp
        self.model[(state, action)] = (reward, next_state, self.current_step)

        # Add to priority queue if prioritised sweeping is enabled
        if self.prioritised_sweeping and abs(td_error) > self.priority_threshold:
            self.priority_queue.put((-abs(td_error), state, action))

        # Forget outdated transitions
        if self.forget:
            self.forget_old_transitions()

        # Planning phase
        for _ in range(self.planning_steps):
            if self.prioritised_sweeping:
                # Prioritised sweeping: Get the highest priority transition
                if self.priority_queue.empty():
                    break
                _, s, a = self.priority_queue.get()
                r, s_prime, _ = self.model[(s, a)]
            else:
                # Random sampling for planning
                if not self.model:
                    break
                s, a = random.choice(list(self.model.keys()))
                r, s_prime, _ = self.model[(s, a)]

            # Perform Q-learning update for the simulated transition
            if s_prime is not None:
                best_next_a = np.argmax(self.q_table[s_prime])
                td_target_sim = (
                    r + self.discount_factor * self.q_table[s_prime, best_next_a]
                )
            else:
                td_target_sim = r

            td_error_sim = td_target_sim - self.q_table[s, a]
            self.q_table[s, a] += self.learning_rate * td_error_sim

            # If prioritised sweeping is enabled, update priorities for predecessors
            if self.prioritised_sweeping:
                for (s_pre, a_pre), (r_pre, s_next, _) in self.model.items():
                    if s_next == s:
                        best_next_a = np.argmax(self.q_table[s])
                        td_target_pre = (
                            r_pre + self.discount_factor * self.q_table[s, best_next_a]
                        )
                        td_error_pre = td_target_pre - self.q_table[s_pre, a_pre]
                        if abs(td_error_pre) > self.priority_threshold:
                            self.priority_queue.put((-abs(td_error_pre), s_pre, a_pre))

        return td_error

    def forget_old_transitions(self):
        """Remove transitions older than the forgetting threshold."""
        outdated_keys = [
            (s, a)
            for (s, a), (_, _, timestamp) in self.model.items()
            if self.current_step - timestamp > self.forgetting_threshold
        ]
        for key in outdated_keys:
            del self.model[key]


class DynaQPlusAgent(QLearningAgent):
    """Dyna-Q+ agent."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate=0.1,
        discount_factor=0.99,
        planning_steps=10,
        epsilon=0.99,
        epsilon_min=0.01,
        epsilon_decay="linear",  # or 'exponential'
        episodes=500,
        novelty_bonus_weight=0.01,  # Novelty bonus scaling factor
    ):
        """Initialise tabular Dyna-Q+ agent with novelty bonus for exploration."""
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            episodes=episodes,
        )

        self.planning_steps = planning_steps
        self.novelty_bonus_weight = novelty_bonus_weight

        # Q-table
        self.q_table = np.zeros((state_size, action_size))

        # Model as a dictionary: (state, action) -> (reward, next_state, last_visit_time)
        self.model = defaultdict(lambda: (0, None, 0))  # Default last visit time is 0

        # Track the current time step
        self.current_step = 0

    def update(self, state, action, reward, next_state, done):
        """Update Q-table, model, and perform planning."""
        self.current_step += 1  # Increment the current step

        # Q-learning update
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (
            0
            if done
            else self.discount_factor * self.q_table[next_state, best_next_action]
        )
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

        # Update the model with the current timestamp
        self.model[(state, action)] = (reward, next_state, self.current_step)

        # Planning phase
        for _ in range(self.planning_steps):
            # Random sampling for planning
            if not self.model:
                break
            sampled_state, sampled_action = random.choice(list(self.model.keys()))
            sampled_reward, sampled_next_state, last_visit = self.model[
                (sampled_state, sampled_action)
            ]

            # Add novelty bonus based on time since last visit
            time_since_last_visit = self.current_step - last_visit
            novelty_bonus = self.novelty_bonus_weight * np.sqrt(time_since_last_visit)
            adjusted_reward = sampled_reward + novelty_bonus

            # Perform Q-learning update for the simulated transition
            if sampled_next_state is not None:
                best_next_a = np.argmax(self.q_table[sampled_next_state])
                td_target_sim = (
                    adjusted_reward
                    + self.discount_factor
                    * self.q_table[sampled_next_state, best_next_a]
                )
            else:
                td_target_sim = adjusted_reward

            td_error_sim = td_target_sim - self.q_table[sampled_state, sampled_action]
            self.q_table[sampled_state, sampled_action] += (
                self.learning_rate * td_error_sim
            )

        return td_error
