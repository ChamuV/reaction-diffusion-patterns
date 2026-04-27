# src/analysis_tools/turing.py

import numpy as np


# Basic stability checks
def trace_det(fu, fv, gu, gv):
    tr = fu + gv
    det = fu * gv - fv * gu
    return tr, det


def stable_without_diffusion(fu, fv, gu, gv):
    tr, det = trace_det(fu, fv, gu, gv)
    return (tr < 0) and (det > 0)


def turing_conditions(fu, fv, gu, gv, d):
    tr, det = trace_det(fu, fv, gu, gv)

    condition_1 = tr < 0
    condition_2 = det > 0
    condition_3 = d * fu + gv > 0
    condition_4 = (d * fu + gv) ** 2 > 4 * d * det

    return {
        "trace_negative": condition_1,
        "determinant_positive": condition_2,
        "diffusion_weighted_positive": condition_3,
        "discriminant_positive": condition_4,
        "turing_unstable": condition_1 and condition_2 and condition_3 and condition_4,
        "trace": tr,
        "determinant": det,
    }


# Dispersion relation
def h_k2(k2_vals, fu, fv, gu, gv, gamma, d):
    det = fu * gv - fv * gu

    return (
        d * k2_vals**2
        - gamma * (d * fu + gv) * k2_vals
        + gamma**2 * det
    )


def dominant_lambda_k2(k2_vals, fu, fv, gu, gv, gamma, d):
    B = k2_vals * (1 + d) - gamma * (fu + gv)
    C = h_k2(k2_vals, fu, fv, gu, gv, gamma, d)

    discriminant = B**2 - 4 * C
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0))

    lambda_1 = (-B + sqrt_disc) / 2
    lambda_2 = (-B - sqrt_disc) / 2

    return np.maximum(lambda_1, lambda_2)


def unstable_k2_band(k2_vals, fu, fv, gu, gv, gamma, d):
    lambdas = dominant_lambda_k2(k2_vals, fu, fv, gu, gv, gamma, d)
    mask = lambdas > 0

    if not np.any(mask):
        return None

    return k2_vals[mask][0], k2_vals[mask][-1]


# Critical diffusion ratio
def critical_diffusion_ratio(fu, fv, gu, gv):
    coeffs = [
        fu**2,
        -2 * fu * gv + 4 * fv * gu,
        gv**2
    ]

    roots = np.roots(coeffs)

    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]

    if len(roots) == 0:
        raise ValueError("No valid critical diffusion ratio found.")

    return np.max(roots)


# Parameter-space scan
def is_turing_unstable_for_mode(fu, fv, gu, gv, d, k2_vals):
    if not stable_without_diffusion(fu, fv, gu, gv):
        return False

    for k2 in k2_vals:
        a11 = fu - k2
        a22 = gv - d * k2

        tr = a11 + a22
        det = a11 * a22 - fv * gu

        disc = tr**2 - 4 * det

        if disc >= 0:
            eig1 = 0.5 * (tr + np.sqrt(disc))
            eig2 = 0.5 * (tr - np.sqrt(disc))

            if eig1 > 0 or eig2 > 0:
                return True

    return False


def compute_turing_mask(A, B, d, k2_vals, derivative_func):
    mask = np.zeros_like(A, dtype=bool)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            fu, fv, gu, gv = derivative_func(A[i, j], B[i, j])

            if not np.all(np.isfinite([fu, fv, gu, gv])):
                mask[i, j] = False
                continue

            mask[i, j] = is_turing_unstable_for_mode(
                fu, fv, gu, gv, d, k2_vals
            )

    return mask