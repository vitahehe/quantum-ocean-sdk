import copy
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.sparse import dok_matrix, csc_matrix, lil_matrix
# ``joblib`` is an optional dependency required only for the
# parallelized version of the Hamiltonian generator

import trackhhl.event_model.q_event_model as em


def angular_and_bifurcation_checks(i, vectors, norms, segments, N, alpha, eps):
    """Helper used by the parallel generator to check a single index."""
    results_ang = []
    results_bif = []
    
    vect_i = vectors[i]
    norm_i = norms[i]

    for j in range(i + 1, N):
        vect_j = vectors[j]
        norm_j = norms[j]
        cosine = np.dot(vect_i, vect_j) / (norm_i * norm_j)

        # angular consistency
        if np.abs(cosine - 1) < eps:
            results_ang.append((i, j, 1))

        # bifurcation consistency
        seg_i, seg_j = segments[i], segments[j]
        if seg_i.from_hit == seg_j.from_hit and seg_i.to_hit != seg_j.to_hit:
            results_bif.append((i, j, -alpha))
        elif seg_i.from_hit != seg_j.from_hit and seg_i.to_hit == seg_j.to_hit:
            results_bif.append((i, j, -alpha))

    return results_ang, results_bif


def generate_hamiltonian(event, params):
    """Naive dense Hamiltonian builder (used for testing)."""
    lambda_val = params.get('lambda')
    alpha = params.get('alpha')
    beta = params.get('beta')

    modules = copy.deepcopy(event.modules)
    modules.sort(key=lambda a: a.z)

    segments = [
        em.segment(from_hit, to_hit)
        for idx in range(len(modules) - 1)
        for from_hit, to_hit in itertools.product(modules[idx].hits, modules[idx + 1].hits)
    ]
    N = len(segments)

    A_ang = np.zeros((N, N))
    A_bif = np.zeros((N, N))
    A_inh = np.zeros((N, N))
    b = np.zeros(N)

    # dense computation loop
    for i, seg_i in enumerate(segments):
        for j, seg_j in enumerate(segments):
            if i == j:
                continue
            vect_i = seg_i.to_vect()
            vect_j = seg_j.to_vect()
            cosine = np.dot(vect_i, vect_j) / (np.linalg.norm(vect_i) * np.linalg.norm(vect_j))
            eps = 1e-9
            if np.abs(cosine - 1) < eps:
                A_ang[i, j] = 1
            if (seg_i.from_hit == seg_j.from_hit and seg_i.to_hit != seg_j.to_hit) or \
               (seg_i.from_hit != seg_j.from_hit and seg_i.to_hit == seg_j.to_hit):
                A_bif[i, j] = -alpha
            if seg_i.from_hit.module_id == 1 and seg_j.to_hit.module_id == 1:
                A_inh[i, j] = beta

    A = -1 * (A_ang + A_bif + A_inh)
    return A, b, segments


def generate_hamiltonian_optimized(event, params):
    """Optimized dense algorithm with some cleanups."""
    lambda_val = params.get('lambda')
    alpha = params.get('alpha')
    beta = params.get('beta')

    modules = sorted(event.modules, key=lambda a: a.z)
    segments = [
        em.segment(from_hit, to_hit)
        for idx in range(len(modules) - 1)
        for from_hit, to_hit in itertools.product(modules[idx].hits, modules[idx + 1].hits)
    ]
    N = len(segments)
    b = np.zeros(N)

    # precompute s_ab to avoid recomputing inside loops
    s_ab = np.zeros((N, N), dtype=int)
    for i, seg_i in enumerate(segments):
        for j, seg_j in enumerate(segments):
            s_ab[i, j] = int(seg_i.from_hit.module_id == 1 and seg_j.to_hit.module_id == 1)

    A_ang = np.zeros((N, N))
    A_bif = np.zeros((N, N))
    A_inh = np.zeros((N, N))
    eps = 1e-9

    for i, seg_i in enumerate(segments):
        vect_i = seg_i.to_vect()
        norm_i = np.linalg.norm(vect_i)
        for j, seg_j in enumerate(segments):
            if i == j:
                continue
            vect_j = seg_j.to_vect()
            norm_j = np.linalg.norm(vect_j)
            cosine = np.dot(vect_i, vect_j) / (norm_i * norm_j)
            if np.abs(cosine - 1) < eps:
                A_ang[i, j] = 1
            if (seg_i.from_hit == seg_j.from_hit and seg_i.to_hit != seg_j.to_hit) or \
               (seg_i.from_hit != seg_j.from_hit and seg_i.to_hit == seg_j.to_hit):
                A_bif[i, j] = -alpha
            A_inh[i, j] = s_ab[i, j] * s_ab[j, i] * beta

    A = -1 * (A_ang + A_bif + A_inh)
    return A, b, segments


def generate_hamiltonian_optimizedG(event, params):
    """Version using sparse `lil_matrix` during construction."""
    lambda_val = params.get('lambda')
    alpha = params.get('alpha')
    beta = params.get('beta')

    modules = sorted(event.modules, key=lambda a: a.z)
    segments = [
        em.segment(from_hit, to_hit)
        for idx in range(len(modules) - 1)
        for from_hit, to_hit in itertools.product(modules[idx].hits, modules[idx + 1].hits)
    ]
    N = len(segments)
    b = np.zeros(N)

    s_ab = sp.lil_matrix((N, N), dtype=int)
    A_ang = sp.lil_matrix((N, N))
    A_bif = sp.lil_matrix((N, N))
    A_inh = sp.lil_matrix((N, N))

    for i, seg_i in enumerate(segments):
        for j, seg_j in enumerate(segments):
            if seg_i.from_hit.module_id == 1 and seg_j.to_hit.module_id == 1:
                s_ab[i, j] = 1

    eps = 1e-9
    for i, seg_i in enumerate(segments):
        vect_i = seg_i.to_vect()
        norm_i = np.linalg.norm(vect_i)
        for j, seg_j in enumerate(segments):
            if i == j:
                continue
            vect_j = seg_j.to_vect()
            norm_j = np.linalg.norm(vect_j)
            cosine = np.dot(vect_i, vect_j) / (norm_i * norm_j)
            if np.abs(cosine - 1) < eps:
                A_ang[i, j] = 1
            if (seg_i.from_hit == seg_j.from_hit and seg_i.to_hit != seg_j.to_hit) or \
               (seg_i.from_hit != seg_j.from_hit and seg_i.to_hit == seg_j.to_hit):
                A_bif[i, j] = -alpha
            A_inh[i, j] = s_ab[i, j] * s_ab[j, i] * beta

    A = -1 * (A_ang + A_bif + A_inh)
    return A.tocsc(), b, segments


def generate_hamiltonian_optimizedPAR(event, params):
    """Parallel version using joblib."""
    alpha = params.get('alpha')
    beta = params.get('beta')

    modules = sorted(event.modules, key=lambda a: a.z)
    segments = [
        em.segment(from_hit, to_hit)
        for idx in range(len(modules) - 1)
        for from_hit, to_hit in itertools.product(modules[idx].hits, modules[idx + 1].hits)
    ]
    N = len(segments)
    b = np.zeros(N)

    vectors = np.array([seg.to_vect() for seg in segments])
    norms = np.linalg.norm(vectors, axis=1)
    eps = 1e-9

    try:
        from joblib import Parallel, delayed
    except ImportError as exc:
        raise ImportError("joblib is required for the parallel Hamiltonian generator") from exc

    results = Parallel(n_jobs=-1, backend="loky")(  # type: ignore
        delayed(angular_and_bifurcation_checks)(i, vectors, norms, segments, N, alpha, eps)
        for i in range(N)
    )

    A_ang = dok_matrix((N, N), dtype=np.float64)
    A_bif = dok_matrix((N, N), dtype=np.float64)

    for ang_results, bif_results in results:
        for i, j, value in ang_results:
            A_ang[i, j] = value
            A_ang[j, i] = value
        for i, j, value in bif_results:
            A_bif[i, j] = value
            A_bif[j, i] = value

    A_ang = A_ang.tocsc()
    A_bif = A_bif.tocsc()

    module_ids_from = np.array([seg.from_hit.module_id for seg in segments])
    module_ids_to = np.array([seg.to_hit.module_id for seg in segments])
    s_ab = sp.csc_matrix((module_ids_from == 1) & (module_ids_to[:, None] == 1), dtype=int)
    A_inh = s_ab.multiply(s_ab.T) * beta

    A = -1 * (A_ang + A_bif + A_inh)
    true_solution = np.array([1 if segment.truth else 0 for segment in segments])
    return A, b, true_solution
