import collections
import enum
import numpy as np
import random
from gym_rubikscube.envs.helper.common import RenderedState, permute, rotate, map_orient, _flip

State = collections.namedtuple("State", field_names=['corner_pos',
                                                     'side_centered_pos',
                                                     'side_norm_pos',
                                                     'one_centered_pos',
                                                     'one_corner_pos',
                                                     'corner_ort',
                                                     'side_centered_ort',
                                                     'side_norm_ort'])

initial_state = State(corner_pos=tuple(range(8)),
                      side_centered_pos=tuple(range(12)),
                      side_norm_pos=tuple(range(24)),
                      one_centered_pos=tuple(range(24)),
                      one_corner_pos=tuple(range(24)),
                      corner_ort=tuple([0] * 8),
                      side_centered_ort=tuple([0] * 12),
                      side_norm_ort=tuple([0] * 24))


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
    R2 = 12
    r2 = 13
    L2 = 14
    l2 = 15
    U2 = 16
    u2 = 17
    D2 = 18
    d2 = 19
    F2 = 20
    f2 = 21
    B2 = 22
    b2 = 23

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
    Action.b1: Action.B1,
    Action.R2: Action.r2,
    Action.r2: Action.R2,
    Action.L2: Action.l2,
    Action.l2: Action.L2,
    Action.U2: Action.u2,
    Action.u2: Action.U2,
    Action.D2: Action.d2,
    Action.d2: Action.D2,
    Action.F2: Action.f2,
    Action.f2: Action.F2,
    Action.B2: Action.b2,
    Action.b2: Action.B2
}

Transformation = collections.namedtuple('Transformation',
                                        ['corner_map',
                                         'side_centered_map',
                                         'side_norm_map',
                                         'one_centered_map',
                                         'one_corner_map',
                                         'corner_rotate',
                                         'side_centered_flip',
                                         'side_norm_flip'])

empty_transform = Transformation(corner_map=(),
                                 side_centered_map=(),
                                 side_norm_map=(),
                                 one_centered_map=(),
                                 one_corner_map=(),
                                 corner_rotate=(),
                                 side_centered_flip=(),
                                 side_norm_flip=())

_transform_map = {
    Action.R1: empty_transform._replace(corner_map=((1, 2), (2, 6), (6, 5), (5, 1)),
                                        side_centered_map=((1, 6), (6, 9), (9, 5), (5, 1)),
                                        side_norm_map=((13, 2), (2, 10), (10, 19), (19, 13),
                                                       (9, 3), (3, 14), (14, 18), (18, 9)),
                                        one_centered_map=((5, 11), (11, 17), (17, 10), (10, 5)),
                                        one_corner_map=((6, 7), (7, 15), (15, 14), (14, 6)),
                                        corner_rotate=((1, 2), (2, 1), (5, 1), (6, 2))),

    Action.R2: empty_transform._replace(side_norm_map=((1, 4), (4, 20), (20, 17), (17, 1)),
                                        one_centered_map=((1, 12), (12, 21), (21, 9), (9, 1)),
                                        one_corner_map=((1, 8), (8, 22), (22, 13), (13, 1),
                                                        (2, 16), (16, 21), (21, 5), (5, 2)),
                                        side_norm_flip=(1, 4, 20, 17)),

    Action.L1: empty_transform._replace(corner_map=((3, 0), (7, 3), (0, 4), (4, 7)),
                                        side_centered_map=((3, 4), (4, 11), (11, 7), (7, 3)),
                                        side_norm_map=((6, 8), (8, 23), (23, 15), (15, 6),
                                                       (12, 22), (22, 11), (11, 7), (7, 12)),
                                        one_centered_map=((7, 15), (15, 19), (19, 14), (14, 7)),
                                        one_corner_map=((10, 11), (11, 19), (19, 18), (18, 10)),
                                        corner_rotate=((0, 1), (3, 2), (4, 2), (7, 1))),

    Action.L2: empty_transform._replace(side_norm_map=((0, 16), (16, 21), (21, 5), (5, 0)),
                                        one_centered_map=((3, 8), (8, 23), (23, 13), (13, 3)),
                                        one_corner_map=((3, 4), (4, 20), (20, 17), (17, 3),
                                                        (0, 12), (12, 23), (23, 9), (9, 0)),
                                        side_norm_flip=(0, 16, 21, 5)),

    Action.U1: empty_transform._replace(corner_map=((0, 3), (1, 0), (2, 1), (3, 2)),
                                        one_centered_map=((0, 3), (3, 2), (2, 1), (1, 0)),
                                        one_corner_map=((0, 3), (3, 2), (2, 1), (1, 0)),
                                        side_centered_map=((0, 3), (3, 2), (2, 1), (1, 0)),
                                        side_norm_map=((0, 6), (6, 4), (4, 2), (2, 0),
                                                       (1, 7), (7, 5), (5, 3), (3, 1))),

    Action.U2: empty_transform._replace(side_norm_map=((8, 11), (11, 10), (10, 9), (9, 8)),
                                        one_centered_map=((4, 7), (7, 6), (6, 5), (5, 4)),
                                        one_corner_map=((5, 11), (11, 9), (9, 7), (7, 5),
                                                        (4, 10), (10, 8), (8, 6), (6, 4)),
                                        side_norm_flip=(8, 11, 10, 9)),

    Action.D1: empty_transform._replace(corner_map=((4, 5), (5, 6), (6, 7), (7, 4)),
                                        side_centered_map=((8, 9), (9, 10), (10, 11), (11, 8)),
                                        side_norm_map=((16, 18), (18, 20), (20, 22), (22, 16),
                                                       (17, 19), (19, 21), (21, 23), (23, 17)),
                                        one_centered_map=((20, 21), (21, 22), (22, 23), (23, 20)),
                                        one_corner_map=((20, 21), (21, 22), (22, 23), (23, 20))),

    Action.D2: empty_transform._replace(side_norm_map=((12, 13), (13, 14), (14, 15), (15, 12)),
                                        one_centered_map=((16, 17), (17, 18), (18, 19), (19, 16)),
                                        one_corner_map=((12, 14), (14, 16), (16, 18), (18, 12),
                                                        (13, 15), (15, 17), (17, 19), (19, 13)),
                                        side_norm_flip=(12, 13, 14, 15)),

    Action.F1: empty_transform._replace(corner_map=((0, 1), (1, 5), (5, 4), (4, 0)),
                                        side_centered_map=((0, 5), (5, 8), (8, 4), (4, 0)),
                                        side_norm_map=((0, 9), (9, 17), (17, 12), (12, 0),
                                                       (1, 13), (13, 16), (16, 8), (8, 1)),
                                        one_centered_map=((4, 9), (9, 16), (16, 8), (8, 4)),
                                        one_corner_map=((4, 5), (5, 13), (13, 12), (12, 4)),
                                        corner_rotate=((0, 2), (1, 1), (4, 1), (5, 2)),
                                        side_centered_flip=(0, 5, 8, 4),
                                        side_norm_flip=(0, 9, 17, 12, 1, 13, 16, 8)),

    Action.F2: empty_transform._replace(side_norm_map=((7, 2), (2, 18), (18, 23), (23, 7)),
                                        one_centered_map=((0, 10), (10, 20), (20, 15), (15, 0)),
                                        one_corner_map=((0, 6), (6, 21), (21, 19), (19, 0),
                                                        (1, 14), (14, 20), (20, 11), (11, 1)),
                                        side_norm_flip=(7, 2, 18, 23)),

    Action.B1: empty_transform._replace(corner_map=((2, 3), (3, 7), (7, 6), (6, 2)),
                                        side_centered_map=((2, 7), (7, 10), (10, 6), (6, 2)),
                                        side_norm_map=((4, 11), (11, 21), (21, 14), (14, 4),
                                                       (5, 15), (15, 20), (20, 10), (10, 5)),
                                        one_centered_map=((12, 6), (6, 13), (13, 18), (18, 12)),
                                        one_corner_map=((8, 9), (9, 17), (17, 16), (16, 8)),
                                        corner_rotate=((2, 2), (3, 1), (6, 1), (7, 2)),
                                        side_centered_flip=(2, 7, 10, 6),
                                        side_norm_flip=(4, 11, 21, 14, 5, 15, 20, 10)),

    Action.B2: empty_transform._replace(side_norm_map=((3, 6), (6, 22), (22, 19), (19, 3)),
                                        one_centered_map=((2, 14), (14, 22), (22, 11), (11, 2)),
                                        one_corner_map=((2, 10), (10, 23), (23, 15), (15, 2),
                                                        (3, 18), (18, 22), (22, 7), (7, 3)),
                                        side_norm_flip=(3, 6, 22, 19))
}


def transform(state, action):
    assert isinstance(state, State)
    action = Action(action)
    # global _transform_map

    is_inv = action not in _transform_map
    if is_inv:
        action = action.inverse
    trans = _transform_map[action]
    corner_pos = permute(state.corner_pos, trans.corner_map, is_inv)
    corner_ort = permute(state.corner_ort, trans.corner_map, is_inv)
    corner_ort = rotate(corner_ort, trans.corner_rotate)

    side_centered_pos = permute(state.side_centered_pos, trans.side_centered_map, is_inv)
    side_centered_ort = permute(state.side_centered_ort, trans.side_centered_map, is_inv)
    if trans.side_centered_flip:
        side_centered_ort = _flip(side_centered_ort, trans.side_centered_flip)

    side_norm_pos = permute(state.side_norm_pos, trans.side_norm_map, is_inv)
    side_norm_ort = permute(state.side_norm_ort, trans.side_norm_map, is_inv)
    if trans.side_norm_flip:
        side_norm_ort = _flip(side_norm_ort, trans.side_norm_flip)

    one_centered_pos = permute(state.one_centered_pos, trans.one_centered_map, is_inv)
    one_corner_pos = permute(state.one_corner_pos, trans.one_corner_map, is_inv)

    return State(corner_pos=corner_pos,
                 side_centered_pos=side_centered_pos,
                 side_norm_pos=side_norm_pos,
                 one_centered_pos=one_centered_pos,
                 one_corner_pos=one_corner_pos,
                 corner_ort=corner_ort,
                 side_centered_ort=side_centered_ort,
                 side_norm_ort=side_norm_ort)


# make initial state of rendered side
def _init_side(color):
    return [color if idx == 12 else None for idx in range(25)]


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

side_centered_colors = (
    ('W', 'R'), ('W', 'B'), ('W', 'O'), ('W', 'G'),
    ('R', 'G'), ('R', 'B'), ('O', 'B'), ('O', 'G'),
    ('Y', 'R'), ('Y', 'B'), ('Y', 'O'), ('Y', 'G')
)

side_norm_colors = (
    ('W', 'R'), ('W', 'R'), ('W', 'B'), ('W', 'B'),
    ('W', 'O'), ('W', 'O'), ('W', 'G'), ('W', 'G'),
    ('R', 'G'), ('R', 'B'), ('O', 'B'), ('O', 'G'),
    ('R', 'G'), ('R', 'B'), ('O', 'B'), ('O', 'G'),
    ('Y', 'R'), ('Y', 'R'), ('Y', 'B'), ('Y', 'B'),
    ('Y', 'O'), ('Y', 'O'), ('Y', 'G'), ('Y', 'G')
)

one_center_colors = (
    ('W',), ('W',), ('W',), ('W',),
    ('R',), ('B',), ('O',), ('G',),
    ('R',), ('R',), ('B',), ('B',),
    ('O',), ('O',), ('G',), ('G',),
    ('R',), ('B',), ('O',), ('G',),
    ('Y',), ('Y',), ('Y',), ('Y',),
)

one_corner_colors = (
    ('W',), ('W',), ('W',), ('W',),
    ('R',), ('R',), ('B',), ('B',),
    ('O',), ('O',), ('G',), ('G',),
    ('R',), ('R',), ('B',), ('B',),
    ('O',), ('O',), ('G',), ('G',),
    ('Y',), ('Y',), ('Y',), ('Y',),
)

# map every 3-side cubelet to their projection on sides
# sides are indexed in the order of _init_sides() function result
corner_maps = (
    # top layer
    ((0, 20), (3, 0), (1, 4)),
    ((0, 24), (4, 0), (3, 4)),
    ((0, 4), (2, 0), (4, 4)),
    ((0, 0), (1, 0), (2, 4)),
    # bottom layer
    ((5, 0), (1, 24), (3, 20)),
    ((5, 4), (3, 24), (4, 20)),
    ((5, 24), (4, 24), (2, 20)),
    ((5, 20), (2, 24), (1, 20))
)

# map every 2-side cubelet to their projection on sides
side_centered_maps = (
    # top layer
    ((0, 22), (3, 2)),
    ((0, 14), (4, 2)),
    ((0, 2), (2, 2)),
    ((0, 10), (1, 2)),
    # middle layer
    ((3, 10), (1, 14)),
    ((3, 14), (4, 10)),
    ((2, 10), (4, 14)),
    ((2, 14), (1, 10)),
    # bottom layer
    ((5, 2), (3, 22)),
    ((5, 14), (4, 22)),
    ((5, 22), (2, 22)),
    ((5, 10), (1, 22))
)

side_norm_maps = (
    # top layer
    ((0, 21), (3, 1)),
    ((0, 23), (3, 3)),
    ((0, 19), (4, 1)),
    ((0, 9), (4, 3)),
    ((0, 3), (2, 1)),
    ((0, 1), (2, 3)),
    ((0, 5), (1, 1)),
    ((0, 15), (1, 3)),
    # middle layer
    ((3, 5), (1, 9)),
    ((3, 9), (4, 5)),
    ((2, 5), (4, 9)),
    ((2, 9), (1, 5)),
    ((3, 15), (1, 19)),
    ((3, 19), (4, 15)),
    ((2, 15), (4, 19)),
    ((2, 19), (1, 15)),
    # bottom layer
    ((5, 1), (3, 21)),
    ((5, 3), (3, 23)),
    ((5, 9), (4, 21)),
    ((5, 19), (4, 23)),
    ((5, 23), (2, 21)),
    ((5, 21), (2, 23)),
    ((5, 15), (1, 21)),
    ((5, 5), (1, 23))
)

one_centered_maps = (
    ((0, 17),),
    ((0, 13),),
    ((0, 7),),
    ((0, 11),),
    ((3, 7),),
    ((4, 7),),
    ((2, 7),),
    ((1, 7),),
    ((3, 11),),
    ((3, 13),),
    ((4, 11),),
    ((4, 13),),
    ((2, 11),),
    ((2, 13),),
    ((1, 11),),
    ((1, 13),),
    ((3, 17),),
    ((4, 17),),
    ((2, 17),),
    ((1, 17),),
    ((5, 7),),
    ((5, 13),),
    ((5, 17),),
    ((5, 11),),
)

one_corner_maps = (
    ((0, 16),),
    ((0, 18),),
    ((0, 8),),
    ((0, 6),),
    ((3, 6),),
    ((3, 8),),
    ((4, 6),),
    ((4, 8),),
    ((2, 6),),
    ((2, 8),),
    ((1, 6),),
    ((1, 8),),
    ((3, 16),),
    ((3, 18),),
    ((4, 16),),
    ((4, 18),),
    ((2, 16),),
    ((2, 18),),
    ((1, 16),),
    ((1, 18),),
    ((5, 6),),
    ((5, 8),),
    ((5, 18),),
    ((5, 16),),
)


# render state into human readable form
def render(state):
    assert isinstance(state, State)
    global corner_colors, corner_maps, side_centered_colors, side_centered_maps, side_norm_colors, side_norm_maps
    global one_center_colors, one_centered_maps, one_corner_colors, one_corner_maps

    sides = _init_sides()

    for corner, orient, maps in zip(state.corner_pos, state.corner_ort, corner_maps):
        cols = corner_colors[corner]
        cols = map_orient(cols, orient)
        for (arr_idx, index), col in zip(maps, cols):
            sides[arr_idx][index] = col

    for side, orient, maps in zip(state.side_centered_pos, state.side_centered_ort, side_centered_maps):
        cols = side_centered_colors[side]
        cols = cols if orient == 0 else (cols[1], cols[0])
        for (arr_idx, index), col in zip(maps, cols):
            sides[arr_idx][index] = col

    for side, orient, maps in zip(state.side_norm_pos, state.side_norm_ort, side_norm_maps):
        cols = side_norm_colors[side]
        cols = cols if orient == 0 else (cols[1], cols[0])
        for (arr_idx, index), col in zip(maps, cols):
            sides[arr_idx][index] = col

    for one, maps in zip(state.one_centered_pos, one_centered_maps):
        col = one_center_colors[one][0]
        arr_idx, index = maps[0]
        sides[arr_idx][index] = col

    for one, maps in zip(state.one_corner_pos, one_corner_maps):
        col = one_corner_colors[one][0]
        arr_idx, index = maps[0]
        sides[arr_idx][index] = col

    return RenderedState(top=sides[0], left=sides[1], back=sides[2], front=sides[3],
                         right=sides[4], bottom=sides[5])


# shape of encoded cube state
encoded_shape = (116, 24)


def encode(state):
    """
    Encode cube into existig zeroed numpy array
    Follows encoding described in paper https://arxiv.org/abs/1805.07470
    :param target: numpy array
    :param state: state to be encoded
    """
    assert isinstance(state, State)

    target = np.zeros(encoded_shape, np.bool)

    # handle corner cubelets: find their permuted position
    for corner_idx in range(8):
        perm_pos = state.corner_pos.index(corner_idx)
        corn_ort = state.corner_ort[perm_pos]
        target[corner_idx, perm_pos * 3 + corn_ort] = 1

    # handle side cubelets
    for side_norm_idx in range(24):
        perm_pos = state.side_norm_pos.index(side_norm_idx)
        side_ort = state.side_norm_ort[perm_pos]
        index = perm_pos * 2 + side_ort
        q, r = index // 24, index % 24
        target[8 + side_norm_idx * 2 + q, r] = 1

    for side_centered_idx in range(12):
        perm_pos = state.side_centered_pos.index(side_centered_idx)
        side_ort = state.side_centered_ort[perm_pos]
        target[56 + side_centered_idx, perm_pos * 2 + side_ort] = 1

    for one_centered_idx in range(24):
        perm_pos = state.one_centered_pos.index(one_centered_idx)
        target[68 + one_centered_idx, perm_pos] = 1

    for one_corner_idx in range(24):
        perm_pos = state.one_corner_pos.index(one_corner_idx)
        target[92 + one_corner_idx, perm_pos] = 1

    return target
