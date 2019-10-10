from gym_rubikscube.envs.helper.cube import Cube
from gym_rubikscube.envs.helper.rubiks3x3_helper import initial_state, transform, encode, encoded_shape, render, \
    adjacency_matrix, generate_cube

max_move_random = 26
size = 3
action_size = 12


class Rubiks3x3Env(Cube):

    def __init__(self):
        super(Rubiks3x3Env, self).__init__(
            size=size,
            max_move_random=max_move_random,
            encoded_shape=encoded_shape,
            nbr_actions=action_size,
            nbr_meta_actions=0,
            initial_state=initial_state,
            transform=transform,
            encode=encode,
            render=render,
            adjacency_matrix=adjacency_matrix
        )

    def randomize(self, state, n=None):
        if n is None:
            return generate_cube(), None
        else:
            return super().randomize(state, n)
