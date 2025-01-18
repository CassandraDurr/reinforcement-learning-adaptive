class AdaptiveGridWorld:
    def __init__(
        self,
        size: tuple[int, int],
        initial_goal: tuple[int, int],
        reward_means: dict[tuple[int, int], float],
        lava_penalty: float = -10,
        boundary_penalty: float = -4,
        lava_pools: list[tuple[int, int]] | None = None,
        changes: dict[int, dict] | None = None,
        small_reward_blocks: list[tuple[int, int]] | None = None,
        small_reward_value: float = 5.0,  # Default reward for small blocks
    ):
        """
        An adaptive GridWorld.

        Args:
            size: (rows, cols)
            initial_goal: Position of the first goal
            reward_means: dict of {(row,col): mean_reward}, for each special cell
                          Cells not in this dict have a default mean of -1.
            lava_penalty: penalty for lava_pools
            boundary_penalty: penalty for ramming into lava_pools
            lava_pools: initial list of lava positions
            changes: a dict of {episode_index: {"goal":(x,y),
                                                "add_lava_pools":[(r,c), ...],
                                                "remove_lava_pools":[(r,c), ...]}}
                     describing environment changes to apply AFTER finishing that episode.
            small_reward_blocks: list of (row, col)
                                 indicating cells with small, consumable rewards.
            small_reward_value: fixed reward value for small reward blocks.
        """
        # Store parameters
        self.initial_goal = initial_goal
        self.initial_lava_pools = lava_pools.copy() if lava_pools else []
        self.initial_small_reward_blocks = (
            small_reward_blocks.copy() if small_reward_blocks else []
        )
        self.size = size
        self.lava_penalty = lava_penalty
        self.boundary_penalty = boundary_penalty
        self.lava_pools = lava_pools if lava_pools else []
        self.reward_means = reward_means
        self.default_mean = -1.0
        self.changes = changes if changes else {}
        self.small_reward_value = small_reward_value
        self.small_reward_blocks = small_reward_blocks
        self.small_reward_blocks_consumed = []
        self.goal = initial_goal
        self.start_pos = (0, 0)
        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.agent_pos = self.start_pos
        # Combine small_reward_blocks and small_reward_blocks_consumed
        self.small_reward_blocks = (
            self.small_reward_blocks + self.small_reward_blocks_consumed
        )
        self.small_reward_blocks_consumed = []
        return self.agent_pos

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, str]:
        """Executes an action, returns (next_state, reward, done). Rewards are stochastic."""
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = moves[action]

        proposed_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])
        next_pos = (
            max(0, min(proposed_pos[0], self.size[0] - 1)),
            max(0, min(proposed_pos[1], self.size[1] - 1)),
        )

        boundary_penalty = self.boundary_penalty if proposed_pos != next_pos else 0

        if next_pos in self.lava_pools:
            reward = self.lava_penalty
            return next_pos, reward, True, "death"

        small_reward = 0
        if next_pos in self.small_reward_blocks:
            small_reward = self.small_reward_value
            # Remove tuple from small_reward_blocks and add it to small_reward_blocks_consumed
            self.small_reward_blocks.remove(next_pos)
            self.small_reward_blocks_consumed.append(next_pos)

        self.agent_pos = next_pos

        cell_mean = self.reward_means.get(next_pos, self.default_mean)
        base_reward = cell_mean
        reward = base_reward + small_reward + boundary_penalty

        if next_pos == self.goal:
            return next_pos, reward, True, "completion"

        return next_pos, reward, False, "ongoing"

    def apply_changes(self, episode_idx: int):
        """Applies environment changes based on the episode index."""
        if episode_idx in self.changes:
            change_dict = self.changes[episode_idx]

            if "goal" in change_dict:
                self.goal = change_dict["goal"]

            if "add_lava_pools" in change_dict:
                self.lava_pools.extend(
                    pool
                    for pool in change_dict["add_lava_pools"]
                    if pool not in self.lava_pools
                )

            if "remove_lava_pools" in change_dict:
                self.lava_pools = [
                    pool
                    for pool in self.lava_pools
                    if pool not in change_dict["remove_lava_pools"]
                ]

            if "add_small_reward_blocks" in change_dict:
                self.small_reward_blocks.extend(
                    small_reward
                    for small_reward in change_dict["add_small_reward_blocks"]
                    if small_reward not in self.small_reward_blocks
                )
                self.small_reward_blocks_consumed = []

            if "remove_small_reward_blocks" in change_dict:
                self.small_reward_blocks = [
                    small_reward
                    for small_reward in self.small_reward_blocks
                    if small_reward not in change_dict["remove_small_reward_blocks"]
                ]
