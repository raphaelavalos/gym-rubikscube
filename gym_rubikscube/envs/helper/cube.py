import gym
import numpy as np
import random
from gym import spaces
from gym_rubikscube.envs.helper.common import plot


class Cube(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size, nbr_actions, nbr_meta_actions, max_move_random, encoded_shape, initial_state, transform,
                 encode, render, adjacency_matrix=None):
        self.size = size
        self.nbr_actions = nbr_actions
        self.nbr_meta_actions = nbr_meta_actions
        self.max_move_random = max_move_random
        self.encoded_shape = encoded_shape
        self.action_space = spaces.Discrete(nbr_actions + nbr_meta_actions)
        self.observation_space = spaces.Box(0, 1, encoded_shape, dtype=np.bool)
        self.goal_state = initial_state
        self.initial_state = initial_state
        self.transform = transform
        self.encode = encode
        self.state, _ = self.randomize(self.goal_state)
        self._render = render
        self._ax = None
        self._fig = None
        self.adjacency_matrix = adjacency_matrix
        self.distance = None

    def _init_side(self, color):
        return [color if idx == self.size ** 2 // 2 else None for idx in range(self.size ** 2)]

    def _init_sides(self):
        return [
            self._init_side('W'),  # top
            self._init_side('G'),  # left
            self._init_side('O'),  # back
            self._init_side('R'),  # front
            self._init_side('B'),  # right
            self._init_side('Y')  # bottom
        ]

    def set_distance(self, distance):
        """
        Set a distance (useful when the env is wrapped)
        :param distance: None | int the new distance
        """
        self.distance = distance

    def get_observation(self):
        """
        Returns the observation
        :return: np.array
        """
        return self.encode(self.state)

    def reset(self):
        """
        Gym reset method
        Randomize from the goal state to create the state with a maximum of self.max_move_random actions
        :return: np.array
        """
        self.state, _ = self.randomize(self.goal_state, self.distance)
        return self.get_observation()

    def step(self, action):
        """
        Gym step method
        :param action: int encoding the (meta_)action
        :return: [np.array, int, bool, dict]
        """
        new_state = self.transform(self.state, action)
        done = new_state == self.goal_state
        reward = 0 if done else -1
        self.state = new_state
        return self.get_observation(), reward, done, {}

    def randomize(self, state, n=None):
        """
        From a given state applies n actions (and no meta_action)
        If n is none n is sampled between 1 and max_move_random
        :param state: State
        :param n: None or int
        :return: [State, list(State)]
        """
        assert (n is None) or ((type(n) is int) and (n > 0))
        if n is None:
            n = random.randint(1, self.max_move_random)
        state_path = [state]
        action = random.choice(range(self.nbr_actions))
        actions = set(range(self.nbr_actions))
        for _ in range(n):
            state_path.append(self.transform(state_path[-1], action))
            action = random.choice(list(actions.difference([action])))
        return state_path[-1], state_path

    def render(self, mode='human'):
        self._fig, self._ax = plot(self._render(self.state), self._fig, self._ax)

    def seed(self, seed=None):
        random.seed(seed)
