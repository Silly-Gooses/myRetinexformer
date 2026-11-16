def fixed_fusion(retinex_x8, hvi):
    if retinex_x8.shape != hvi.shape:
        raise ValueError("RetinexFormer and HVI predictions must have identical dimensions")
    return (0.25 * retinex_x8 + 0.75 * hvi).clamp(0, 1)
