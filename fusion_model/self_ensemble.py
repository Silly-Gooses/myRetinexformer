from itertools import product

import torch


@torch.inference_mode()
def ensemble8(x, network):
    total = None
    for flip_height, flip_width, rotate in product((False, True), repeat=3):
        view = x
        if flip_height:
            view = view.flip((-2,))
        if flip_width:
            view = view.flip((-1,))
        if rotate:
            view = view.rot90(1, (-2, -1))
        pred = network(view)
        if rotate:
            pred = pred.rot90(3, (-2, -1))
        if flip_width:
            pred = pred.flip((-1,))
        if flip_height:
            pred = pred.flip((-2,))
        total = pred.float() if total is None else total + pred.float()
    return total / 8.0
