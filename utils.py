import numpy as np


def pyramid(inp, out, h):
    return np.linspace(inp, out, num=h + 2, dtype=np.int32).tolist()
