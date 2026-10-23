import numpy as np
import torch

from vlmforge.model.vision import CROP, GRID, MARGIN, overlap_crops, reconstruct, resize_lanczos3


def test_overlap_crops_matches_engine_tilings():
    # tilings the engine chose for the parity fixture images
    for (h, w), tiling in [((720, 1280), (2, 4)), ((480, 752), (2, 4)), ((512, 640), (3, 3))]:
        crops, t = overlap_crops(np.zeros((h, w, 3), np.uint8))
        assert t == tiling
        assert crops.shape == (1 + t[0] * t[1], CROP, CROP, 3) and crops.dtype == torch.uint8


def test_small_image_still_gets_one_local_crop():
    crops, t = overlap_crops(np.zeros((200, 300, 3), np.uint8))
    assert t == (1, 1) and crops.shape[0] == 2


def test_resize_shapes_and_identity():
    img = np.random.default_rng(0).integers(0, 256, (300, 1000, 3), dtype=np.uint8)
    assert (resize_lanczos3(img, 300, 1000) == img).all()
    assert resize_lanczos3(img, 50, 378).shape == (50, 378, 3)


def test_reconstruct_keeps_constant_features():
    local = torch.ones(6, GRID, GRID, 4)
    out = reconstruct(local, (2, 3))
    win = GRID - 2 * MARGIN
    assert out.shape == (win * 2 + 2 * MARGIN, win * 3 + 2 * MARGIN, 4)
    assert bool((out == 1).all())
