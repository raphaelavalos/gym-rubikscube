from gym_rubikscube.envs.helper.cube import Cube
from gym_rubikscube.envs.helper.rubiks3x3_helper import initial_state, encode, render, transform, adjacency_matrix, \
    generate_cube
import numpy as np
from gym import spaces
import random

max_move_random = 26
size = 3
encoded_shape = (20, 24)
nbr_actions = 12
nbr_meta_actions = 0


class Rubiks3x3GoalEnv(Cube):

    def __init__(self):
        super(Rubiks3x3GoalEnv, self).__init__(
            size=size,
            max_move_random=max_move_random,
            encoded_shape=encoded_shape,
            nbr_actions=nbr_actions,
            nbr_meta_actions=nbr_meta_actions,
            initial_state=initial_state,
            transform=transform,
            encode=encode,
            render=render,
            adjacency_matrix=adjacency_matrix
        )
        self.observation_space = spaces.Dict({
            'state': spaces.Box(0, 1, encoded_shape, dtype=np.bool),
            'goal': spaces.Box(0, 1, encoded_shape, dtype=np.bool)
        })
        self.randomize_goal = True
        self.goal_curriculum = False

    def set_goal_curriculum(self, goal_curriculum):
        self.goal_curriculum = goal_curriculum

    def reset(self):
        assert self.distance is not None or not self.goal_curriculum
        if self.randomize_goal:
            if self.goal_curriculum:
                if self.distance == 1 or random.random() < 0.3:
                    self.goal_state = self.initial_state
                    self.state, _ = self.randomize(self.goal_state, self.distance)
                else:
                    self.state, state_path = self.randomize(self.initial_state, self.distance)
                    self.goal_state = random.choice(state_path[1:-1])
            else:
                self.goal_state, _ = self.randomize(self.initial_state)
                self.state, _ = self.randomize(self.goal_state, self.distance)
        else:
            self.goal_state = self.initial_state
            self.state, _ = self.randomize(self.goal_state, self.distance)
        return self.get_observation()

    def get_observation(self):
        observation = {
            'state': self.encode(self.state),
            'goal': self.encode(self.goal_state)
        }
        return observation

    def randomize(self, state, n=None):
        if n is None:
            return generate_cube(), None
        else:
            return super().randomize(state, n)
