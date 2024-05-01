import math

def adjust_moco_momentum(epoch, max_epoch, moco_m=0.99):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) * (1. - moco_m)
    return m
