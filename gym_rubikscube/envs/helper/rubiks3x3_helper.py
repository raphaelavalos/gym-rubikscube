import enum
import numpy as np
import random
import collections
from gym_rubikscube.envs.helper.common import RenderedState, permute, rotate, map_orient, _flip

State = collections.namedtuple("State", field_names=['corner_pos',
                                                     'side_pos',
                                                     'corner_ort',
                                                     'side_ort'])

initial_state = State(corner_pos=tuple(range(8)),
                      side_pos=tuple(range(12)),
                      corner_ort=tuple([0] * 8),
                      side_ort=tuple([0] * 12))


class Action(enum.Enum):
    R1 = 0
    r1 = 1
    L1 = 2
    l1 = 3
    U1 = 4
    u1 = 5
    D1 = 6
    d1 = 7
    F1 = 8
    f1 = 9
    B1 = 10
    b1 = 11

    @property
    def inverse(self):
        return _inverse_action[self]

    @classmethod
    def random(cls):
        return Action(random.randint(0, len(cls) - 1))

    def random_except(self):
        action = Action.random()
        while self.inverse == action:
            action = Action.random()
        return action


_inverse_action = {
    Action.R1: Action.r1,
    Action.r1: Action.R1,
    Action.L1: Action.l1,
    Action.l1: Action.L1,
    Action.U1: Action.u1,
    Action.u1: Action.U1,
    Action.D1: Action.d1,
    Action.d1: Action.D1,
    Action.F1: Action.f1,
    Action.f1: Action.F1,
    Action.B1: Action.b1,
    Action.b1: Action.B1
}

Transformation = collections.namedtuple('Transformation',
                                        ['corner_map',
                                         'side_map',
                                         'corner_rotate',
                                         'side_flip'])

empty_transform = Transformation(corner_map=(),
                                 side_map=(),
                                 corner_rotate=(),
                                 side_flip=())

_transform_map = {
    Action.R1: empty_transform._replace(
        corner_map=((1, 2), (2, 6), (6, 5), (5, 1)),
        side_map=((1, 6), (6, 9), (9, 5), (5, 1)),
        corner_rotate=((1, 2), (2, 1), (5, 1), (6, 2))),

    Action.L1: empty_transform._replace(
        corner_map=((3, 0), (7, 3), (0, 4), (4, 7)),
        side_map=((7, 3), (3, 4), (11, 7), (4, 11)),
        corner_rotate=((0, 1), (3, 2), (4, 2), (7, 1))),
    Action.U1: empty_transform._replace(
        corner_map=((0, 3), (1, 0), (2, 1), (3, 2)),
        side_map=((0, 3), (1, 0), (2, 1), (3, 2))),
    Action.D1: empty_transform._replace(
        corner_map=((4, 5), (5, 6), (6, 7), (7, 4)),
        side_map=((8, 9), (9, 10), (10, 11), (11, 8))),
    Action.F1: empty_transform._replace(
        corner_map=((0, 1), (1, 5), (5, 4), (4, 0)),
        side_map=((0, 5), (4, 0), (5, 8), (8, 4)),
        corner_rotate=((0, 2), (1, 1), (4, 1), (5, 2)),
        side_flip=(0, 4, 5, 8)),
    Action.B1: empty_transform._replace(
        corner_map=((2, 3), (3, 7), (7, 6), (6, 2)),
        side_map=((2, 7), (6, 2), (7, 10), (10, 6)),
        corner_rotate=((2, 2), (3, 1), (6, 1), (7, 2)),
        side_flip=(2, 6, 7, 10)),
}


def transform(state, action):
    """
    Transform the state with the action
    :param state: State
    :param action: int
    :return: State
    """
    assert isinstance(state, State)
    action = Action(action)

    is_inv = action not in _transform_map
    if is_inv:
        action = action.inverse
    trans = _transform_map[action]
    corner_pos = permute(state.corner_pos, trans.corner_map, is_inv)
    corner_ort = permute(state.corner_ort, trans.corner_map, is_inv)
    corner_ort = rotate(corner_ort, trans.corner_rotate)
    side_pos = permute(state.side_pos, trans.side_map, is_inv)
    side_ort = state.side_ort
    side_ort = permute(side_ort, trans.side_map, is_inv)
    if trans.side_flip:
        side_ort = _flip(side_ort, trans.side_flip)
    return State(corner_pos=tuple(corner_pos), corner_ort=tuple(corner_ort),
                 side_pos=tuple(side_pos), side_ort=tuple(side_ort))


# make initial state of rendered side
def _init_side(color):
    return [color if idx == 4 else None for idx in range(9)]


# create initial sides in the right order
def _init_sides():
    return [
        _init_side('W'),  # top
        _init_side('G'),  # left
        _init_side('O'),  # back
        _init_side('R'),  # front
        _init_side('B'),  # right
        _init_side('Y')  # bottom
    ]


# corner cubelets colors (clockwise from main label). Order of cubelets are first top,
# in counter-clockwise, started from front left
corner_colors = (
    ('W', 'R', 'G'), ('W', 'B', 'R'), ('W', 'O', 'B'), ('W', 'G', 'O'),
    ('Y', 'G', 'R'), ('Y', 'R', 'B'), ('Y', 'B', 'O'), ('Y', 'O', 'G')
)

side_colors = (
    ('W', 'R'), ('W', 'B'), ('W', 'O'), ('W', 'G'),
    ('R', 'G'), ('R', 'B'), ('O', 'B'), ('O', 'G'),
    ('Y', 'R'), ('Y', 'B'), ('Y', 'O'), ('Y', 'G')
)

# map every 3-side cubelet to their projection on sides
# sides are indexed in the order of _init_sides() function result
corner_maps = (
    # top layer
    ((0, 6), (3, 0), (1, 2)),
    ((0, 8), (4, 0), (3, 2)),
    ((0, 2), (2, 0), (4, 2)),
    ((0, 0), (1, 0), (2, 2)),
    # bottom layer
    ((5, 0), (1, 8), (3, 6)),
    ((5, 2), (3, 8), (4, 6)),
    ((5, 8), (4, 8), (2, 6)),
    ((5, 6), (2, 8), (1, 6))
)

# map every 2-side cubelet to their projection on sides
side_maps = (
    # top layer
    ((0, 7), (3, 1)),
    ((0, 5), (4, 1)),
    ((0, 1), (2, 1)),
    ((0, 3), (1, 1)),
    # middle layer
    ((3, 3), (1, 5)),
    ((3, 5), (4, 3)),
    ((2, 3), (4, 5)),
    ((2, 5), (1, 3)),
    # bottom layer
    ((5, 1), (3, 7)),
    ((5, 5), (4, 7)),
    ((5, 7), (2, 7)),
    ((5, 3), (1, 7))
)


# render state into human readable form
def render(state):
    """
    Render the state
    :param state: State
    :return: RenderedState
    """
    assert isinstance(state, State)
    global corner_colors, corner_maps, side_colors, side_maps

    sides = _init_sides()

    for corner, orient, maps in zip(state.corner_pos, state.corner_ort, corner_maps):
        cols = corner_colors[corner]
        cols = map_orient(cols, orient)
        for (arr_idx, index), col in zip(maps, cols):
            sides[arr_idx][index] = col

    for side, orient, maps in zip(state.side_pos, state.side_ort, side_maps):
        cols = side_colors[side]
        cols = cols if orient == 0 else (cols[1], cols[0])
        for (arr_idx, index), col in zip(maps, cols):
            sides[arr_idx][index] = col

    return RenderedState(top=sides[0], left=sides[1], back=sides[2], front=sides[3],
                         right=sides[4], bottom=sides[5])


# shape of encoded cube state
encoded_shape = (20, 24)


def encode(state):
    """
    Encode cube into bool numpy array
    Follows encoding described in paper https://arxiv.org/abs/1805.07470
    :param state: State
    :return np.array
    """
    assert isinstance(state, State)

    target = np.zeros(encoded_shape, np.bool)

    # handle corner cubelets: find their permuted position
    for corner_idx in range(8):
        perm_pos = state.corner_pos.index(corner_idx)
        corn_ort = state.corner_ort[perm_pos]
        target[corner_idx, perm_pos * 3 + corn_ort] = 1

    # handle side cubelets
    for side_idx in range(12):
        perm_pos = state.side_pos.index(side_idx)
        side_ort = state.side_ort[perm_pos]
        target[8 + side_idx, perm_pos * 2 + side_ort] = 1

    return target


def make_adjacency_matrix():
    adjacency_matrix = np.eye(20, dtype=np.float32)
    # corners
    adjacency_matrix[0][[1, 3, 4]] = 1.
    adjacency_matrix[1][[0, 2, 5]] = 1.
    adjacency_matrix[2][[1, 3, 6]] = 1.
    adjacency_matrix[3][[0, 2, 7]] = 1.
    adjacency_matrix[4][[0, 5, 7]] = 1.
    adjacency_matrix[5][[1, 4, 6]] = 1.
    adjacency_matrix[6][[2, 5, 7]] = 1.
    adjacency_matrix[7][[3, 4, 6]] = 1.
    # edges
    adjacency_matrix[8][[9, 11, 12, 15]] = 1.
    adjacency_matrix[9][[8, 10, 12, 13]] = 1.
    adjacency_matrix[10][[9, 11, 13, 14]] = 1.
    adjacency_matrix[11][[8, 10, 14, 15]] = 1.
    adjacency_matrix[12][[8, 9, 16, 17]] = 1.
    adjacency_matrix[13][[9, 10, 17, 18]] = 1.
    adjacency_matrix[14][[10, 11, 18, 19]] = 1.
    adjacency_matrix[15][[8, 11, 16, 19]] = 1.
    adjacency_matrix[16][[12, 15, 17, 19]] = 1.
    adjacency_matrix[17][[12, 13, 16, 18]] = 1.
    adjacency_matrix[18][[13, 14, 17, 19]] = 1.
    adjacency_matrix[19][[14, 15, 16, 18]] = 1.
    return adjacency_matrix


def generate_pair_permutation(n):
    while True:
        perm = np.random.permutation(n)
        if permutation_sign(perm) == 1:
            return perm


def generate_corner_ort():
    while True:
        ort = np.random.randint(0, 3, 8)
        if sum(ort) % 3 == 0:
            return ort


def generate_side_ort():
    while True:
        ort = np.random.randint(0, 2, 12)
        if sum(ort) % 2 == 0:
            return ort


def permutation_sign(perm):
    return np.sign([perm[j] - perm[i] for i in range(len(perm)) for j in range(i + 1, len(perm))]).prod()


def generate_cube():
    return State(
        corner_pos=tuple(generate_pair_permutation(8)),
        side_pos=tuple(generate_pair_permutation(12)),
        corner_ort=tuple(generate_corner_ort()),
        side_ort=tuple(generate_side_ort())
    )


adjacency_matrix = make_adjacency_matrix()
