import os
import numpy as np
from scipy.sparse import csc_matrix
import dimod
from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler


def qubosolverSA(A, b, num_reads=100):
    """Simulated annealing sampler."""
    A = csc_matrix(A)
    bqm = BinaryQuadraticModel.empty(dimod.BINARY)
    bqm.add_variables_from({i: b[i] for i in range(len(b))})
    row, col = A.nonzero()
    for i, j in zip(row, col):
        if i != j:
            bqm.add_interaction(i, j, A[i, j])
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads)
    best_sample = response.first.sample
    sol = np.fromiter(best_sample.values(), dtype=int)
    print(f"SA solution: {sol}")
    return sol


def qubosolverQA(A, b, num_reads=100, token=None):
    """Quantum annealing via D-Wave sampler."""
    A = csc_matrix(A)
    bqm = BinaryQuadraticModel.empty(dimod.BINARY)
    bqm.add_variables_from({i: b[i] for i in range(len(b))})
    row, col = A.nonzero()
    for i, j in zip(row, col):
        if i != j:
            bqm.add_interaction(i, j, A[i, j])
    kwargs = {}
    if token:
        kwargs['token'] = token
    sampler = EmbeddingComposite(DWaveSampler(**kwargs))
    response = sampler.sample(bqm, num_reads=num_reads)
    best_sample = response.first.sample
    sol = np.fromiter(best_sample.values(), dtype=int)
    print(f"QA solution: {sol}")
    return sol


def qubosolverHr(A, b):
    """Hybrid solver using LeapHybridSampler."""
    A = csc_matrix(A)
    bqm = BinaryQuadraticModel.empty(dimod.BINARY)
    bqm.add_variables_from({i: b[i] for i in range(len(b))})
    row, col = A.nonzero()
    for i, j in zip(row, col):
        if i != j:
            bqm.add_interaction(i, j, A[i, j])
    sampler = LeapHybridSampler()
    response = sampler.sample(bqm)
    best_sample = response.first.sample
    sol = np.fromiter(best_sample.values(), dtype=int)
    print(f"Hybrid solution: {sol}")
    return sol


def brute_force(A, b):
    """Enumerate all configurations (small n only)."""
    A = csc_matrix(A)
    n = len(b)
    best_energy = np.inf
    best_sol = None
    from itertools import product
    for config in product([0, 1], repeat=n):
        config = np.array(config)
        energy = config @ A @ config + b @ config
        if energy < best_energy:
            best_energy = energy
            best_sol = config
    print(f"Brute force best solution: {best_sol}, energy: {best_energy}")
    return best_sol
