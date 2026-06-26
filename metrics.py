import numba
import numpy as np


@numba.njit(parallel=True)
def argsort_rows(X):
    n, m = X.shape
    out = np.empty((n, m), dtype=np.int64)

    for i in numba.prange(n):
        out[i] = np.argsort(X[i])

    return out


@numba.njit(parallel=True, fastmath=False)
def _compute_ranks_from_argsort(nn):
    """
    nn[i] = argsort indices for row i
    returns ranks[i, j] = rank of j in row i
    """
    n = nn.shape[0]
    ranks = np.empty((n, n), dtype=np.int64)

    for i in numba.prange(n):
        for r in range(n):
            ranks[i, nn[i, r]] = r

    return ranks


@numba.njit(parallel=True, fastmath=False)
def metric_trustworthiness_numba(D_high, D_low, k):
    """
    Drop-in Numba replacement for metric_trustworthiness
    Assumes D_high and D_low are square distance matrices
    """

    n = D_high.shape[0]

    nn_orig = argsort_rows(D_high)
    nn_proj = argsort_rows(D_low)

    ranks_orig = _compute_ranks_from_argsort(nn_orig)

    partial = np.zeros(n, dtype=np.float64)

    for i in numba.prange(n):
        s = 0.0

        # mark original kNN (excluding self)
        in_orig = np.zeros(n, dtype=np.uint8)
        for t in range(1, k + 1):
            in_orig[nn_orig[i, t]] = 1

        # projected neighbors not in original
        for t in range(1, k + 1):
            j = nn_proj[i, t]
            if in_orig[j] == 0:
                s += ranks_orig[i, j] - k

        partial[i] = s

    total = partial.sum()
    norm = n * k * (2 * n - 3 * k - 1)

    return 1.0 - (2.0 * total) / norm


@numba.njit(parallel=True, fastmath=False)
def metric_continuity_numba(D_high, D_low, k):
    """
    Drop-in Numba replacement for metric_continuity
    Assumes D_high and D_low are square distance matrices
    """

    n = D_high.shape[0]

    nn_orig = argsort_rows(D_high)
    nn_proj = argsort_rows(D_low)

    ranks_proj = _compute_ranks_from_argsort(nn_proj)

    partial = np.zeros(n, dtype=np.float64)

    for i in numba.prange(n):
        s = 0.0

        # mark projected kNN (excluding self)
        in_proj = np.zeros(n, dtype=np.uint8)
        for t in range(1, k + 1):
            in_proj[nn_proj[i, t]] = 1

        # original neighbors not in projected
        for t in range(1, k + 1):
            j = nn_orig[i, t]
            if in_proj[j] == 0:
                s += ranks_proj[i, j] - k

        partial[i] = s

    total = partial.sum()
    norm = n * k * (2 * n - 3 * k - 1)

    return 1.0 - (2.0 * total) / norm


@numba.njit(parallel=True, fastmath=False)
def trustworthiness_continuity_powers_of_two(D_high, D_low):
    """
    Compute trustworthiness and continuity for
    k = 2, 4, 8, 16, ... floor(N/2)

    Returns
    -------
    ks    : int64 array of k values
    trust : float64 array
    cont  : float64 array
    """

    n = D_high.shape[0]
    kmax = n // 2

    nn_orig = argsort_rows(D_high)
    nn_proj = argsort_rows(D_low)

    ranks_orig = _compute_ranks_from_argsort(nn_orig)
    ranks_proj = _compute_ranks_from_argsort(nn_proj)

    trust_partial = np.zeros((n, kmax + 1), dtype=np.float64)
    cont_partial = np.zeros((n, kmax + 1), dtype=np.float64)

    for i in numba.prange(n):
        t_sum = 0.0
        c_sum = 0.0

        for k in range(1, kmax + 1):
            # Trustworthiness increment
            j_proj = nn_proj[i, k]
            r_orig = ranks_orig[i, j_proj]
            if r_orig > k:
                t_sum += r_orig - k

            # Continuity increment
            j_orig = nn_orig[i, k]
            r_proj = ranks_proj[i, j_orig]
            if r_proj > k:
                c_sum += r_proj - k

            trust_partial[i, k] = t_sum
            cont_partial[i, k] = c_sum

    # Count how many powers of two we have
    count = 0
    k = 2
    while k <= kmax:
        count += 1
        k *= 2

    ks = np.empty(count, dtype=np.int64)
    trust = np.empty(count, dtype=np.float64)
    cont = np.empty(count, dtype=np.float64)

    idx = 0
    k = 2
    while k <= kmax:
        total_t = trust_partial[:, k].sum()
        total_c = cont_partial[:, k].sum()
        norm = n * k * (2 * n - 3 * k - 1)

        ks[idx] = k
        trust[idx] = 1.0 - (2.0 * total_t) / norm
        cont[idx] = 1.0 - (2.0 * total_c) / norm

        idx += 1
        k *= 2

    return ks, trust, cont
