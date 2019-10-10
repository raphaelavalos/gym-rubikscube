import collections
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def permute(t, m, is_inv=False):
    """
    Perform permutation of tuple according to mapping m
    """
    r = list(t)
    for from_idx, to_idx in m:
        if is_inv:
            r[from_idx] = t[to_idx]
        else:
            r[to_idx] = t[from_idx]
    return tuple(r)


def rotate(corner_ort, corners):
    """
    Rotate given corners 120 degrees
    """
    r = list(corner_ort)
    for c, angle in corners:
        r[c] = (r[c] + angle) % 3
    return tuple(r)


# orient corner cubelet
def map_orient(cols, orient_id):
    if orient_id == 0:
        return cols
    elif orient_id == 1:
        return cols[2], cols[0], cols[1]
    else:
        return cols[1], cols[2], cols[0]


def _flip(side_ort, sides):
    return [
        o if idx not in sides else 1 - o
        for idx, o in enumerate(side_ort)
    ]


RenderedState = collections.namedtuple("RenderedState",
                                       field_names=['top', 'front', 'left', 'right', 'back', 'bottom'])

colors = {
    'B': 'blue',
    'W': 'white',
    'Y': 'yellow',
    'G': 'green',
    'O': 'orange',
    'R': 'red',
    'black': 'black'
}


def plot(render_state, fig=None, ax=None):
    size = int(math.sqrt(len(render_state.top)))
    if fig is None or ax is None or not plt.get_fignums():
        fig = plt.figure(facecolor='grey', figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d', xlim=(0, size), ylim=(0, size), zlim=(0, size), xticks=[], yticks=[],
                             zticks=[], frame_on=False)
    # top layer
    z = [size] * 4
    for i, c in enumerate(render_state.top):
        a, b = i % size, i // size
        x = [a, a, a + 1, a + 1]
        y = [size - 1 - b, size - b, size - b, size - 1 - b]
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=colors[c], edgecolors='black'))
    # front layer
    y = [0] * 4
    for i, c in enumerate(render_state.front):
        a, b = i % size, i // size
        x = [a, a, a + 1, a + 1]
        z = [size - 1 - b, size - b, size - b, size - 1 - b]
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=colors[c], edgecolors='black'))
    # right layer
    x = [size] * 4
    for i, c in enumerate(render_state.right):
        a, b = i % size, i // size
        y = [a, a, a + 1, a + 1]
        z = [size - 1 - b, size - b, size - b, size - 1 - b]
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=colors[c], edgecolors='black'))
    # left layer
    x = [0] * 4
    for i, c in enumerate(render_state.left):
        a, b = i % size, i // size
        y = [size - a, size - a, size - a - 1, size - a - 1]
        z = [size - 1 - b, size - b, size - b, size - 1 - b]
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=colors[c], edgecolors='black'))
    # back layer
    y = [size] * 4
    for i, c in enumerate(render_state.back):
        a, b = i % size, i // size
        x = [size - a, size - a, size - a - 1, size - a - 1]
        z = [size - 1 - b, size - b, size - b, size - 1 - b]
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=colors[c], edgecolors='black'))
    # bottom layer
    z = [0] * 4
    for i, c in enumerate(render_state.bottom):
        a, b = i % size, i // size
        x = [a, a, a + 1, a + 1]
        y = [b, b + 1, b + 1, b]
        verts = [list(zip(x, y, z))]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=colors[c], edgecolors='black'))
    plt.show()
    return fig, ax
