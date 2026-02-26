#!/usr/bin/env python3
"""Generate exact scipy evaluation values at specific test points"""

import numpy as np
from scipy.interpolate import BPoly

def get_values(name, xi, yi, eval_points):
    bp = BPoly.from_derivatives(xi, yi)
    results = bp(eval_points)
    print(f"\n// {name}")
    print(f"// xi: {xi}")
    print(f"// yi: {yi}")
    print(f"// coeffs: {[bp.c[i].tolist() for i in range(bp.c.shape[0])]}")
    values_str = ", ".join([f"{v:.10f}" for v in results])
    print(f"{{{values_str}}}")
    return results

# Standard eval points
pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

print("=== Exact scipy values for test cases ===")

# Linear
get_values("linear_2pt", [0, 1], [[0], [1]], pts)

# Cubic Hermite
get_values("hermite_cubic", [0, 1], [[0, 1], [1, -1]], pts)

# Quintic Hermite
get_values("hermite_quintic", [0, 1], [[0, 1, 0], [1, -1, 0]], pts)

# Asymmetric 1_2
get_values("asymmetric_1_2", [0, 1], [[0], [1, 0]], pts)

# Asymmetric 2_1
get_values("asymmetric_2_1", [0, 1], [[0, 1], [1]], pts)

# Asymmetric 3_1
get_values("asymmetric_3_1", [0, 1], [[0, 1, 2], [1]], pts)

# Asymmetric 1_3
get_values("asymmetric_1_3", [0, 1], [[0], [1, -1, 0]], pts)

# Higher order 4
get_values("higher_order_4", [0, 1], [[0, 1, 0, 0], [1, -1, 0, 0]], pts)

# Non-unit interval
pts_2 = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
get_values("non_unit_interval", [0, 2], [[0, 0.5], [4, 2]], pts_2)

# Negative interval
pts_neg = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
get_values("negative_interval", [-1, 1], [[1, 0], [1, 0]], pts_neg)

# Three points mixed
pts_3 = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
get_values("three_points_mixed", [0, 1, 2], [[0, 1], [1], [0, -1]], pts_3)
