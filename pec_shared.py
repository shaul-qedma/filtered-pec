"""Shared utilities for PEC implementations."""

import numpy as np
from dataclasses import dataclass
from typing import Union, Dict, List, Tuple, Optional

ETA = np.array([[+1,+1,+1,+1], [+1,+1,-1,-1], [+1,-1,+1,-1], [+1,-1,-1,+1]], dtype=np.float64)

PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]

PAULI_BITS = {
    0: (0, 0),  # I
    1: (1, 0),  # X
    2: (1, 1),  # Y
    3: (0, 1),  # Z
}
BITS_TO_PAULI = {v: k for k, v in PAULI_BITS.items()}


STANDARD_GATES = {
    "CNOT": np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex),
    # Control on the second qubit (q1), target on the first (q0).
    "CNOT_R": np.array([[1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0]], dtype=complex),
    "CZ": np.diag([1, 1, 1, -1]).astype(complex),
    "SWAP": np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], dtype=complex),
}

# Finite bank of two-qubit Clifford gates for random circuit sampling.
CLIFFORD_2Q_GATES = ("CNOT", "CNOT_R", "CZ", "SWAP")

@dataclass
class Gate:
    qubits: tuple
    content: object

@dataclass  
class Circuit:
    n_qubits: int
    layers: list

def pauli_to_kraus(probs: np.ndarray) -> List[np.ndarray]:
    """
    Convert Pauli probabilities [p_I, p_X, p_Y, p_Z] to Kraus operators.
    
    Returns list of 4 Kraus operators K_i = sqrt(p_i) * P_i
    """
    if len(probs) != 4:
        raise ValueError("probs must have exactly 4 elements [p_I, p_X, p_Y, p_Z]")
    if not np.isclose(sum(probs), 1.0):
        raise ValueError(f"probabilities must sum to 1, got {sum(probs)}")
    if any(p < 0 for p in probs):
        raise ValueError("probabilities must be non-negative")
    
    return [np.sqrt(p) * P for p, P in zip(probs, PAULIS)]


def compose_paulis(a: int, b: int) -> int:
    """Compose Pauli indices a * b, ignoring global phase."""
    xa, za = PAULI_BITS[a]
    xb, zb = PAULI_BITS[b]
    return BITS_TO_PAULI[(xa ^ xb, za ^ zb)]

def random_pauli_probs(rng: np.random.Generator, p_I: float) -> np.ndarray:
    """
    Generate random Pauli probabilities with fixed identity probability.
    
    Args:
        rng: Random number generator
        p_I: Probability of identity (no error). Must be in [0, 1].
    
    Returns:
        Array [p_I, p_X, p_Y, p_Z] summing to 1
    """
    if not 0 <= p_I <= 1:
        raise ValueError(f"p_I must be in [0, 1], got {p_I}")
    
    rest = 1.0 - p_I
    if rest < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    # Random partition of remaining probability
    cuts = np.sort(rng.uniform(0, rest, size=2))
    p_X = cuts[0]
    p_Y = cuts[1] - cuts[0]
    p_Z = rest - cuts[1]
    
    return np.array([p_I, p_X, p_Y, p_Z])


def depolarizing_probs(p_I: float) -> np.ndarray:
    """
    Generate symmetric depolarizing channel probabilities.
    
    For depolarizing: p_X = p_Y = p_Z = (1 - p_I) / 3
    """
    if not 0 <= p_I <= 1:
        raise ValueError(f"p_I must be in [0, 1], got {p_I}")
    
    p_error = (1.0 - p_I) / 3.0
    return np.array([p_I, p_error, p_error, p_error])

def kraus_to_probs(kraus):
    return np.array([np.real(np.trace(K.conj().T @ K)) / 2 for K in kraus])

def quasi_probs(p):
    lam = ETA @ p
    return 0.25 * ETA @ (1.0 / lam)

def error_locations(circuit, noise_model):
    locs = []
    for l, layer in enumerate(circuit.layers):
        for gate in layer:
            if isinstance(gate.content, str) and gate.content in noise_model:
                kraus1, kraus2 = noise_model[gate.content]
                locs.append((l, gate.qubits[0], kraus_to_probs(kraus1)))
                locs.append((l, gate.qubits[1], kraus_to_probs(kraus2)))
    return locs

def haar_random_1q(rng):
    """Haar-random single-qubit unitary."""
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    phases = np.ones_like(d)
    nonzero = np.abs(d) > 0
    phases[nonzero] = d[nonzero] / np.abs(d[nonzero])
    return q * phases

def random_circuit(n_qubits, depth, rng, twoq_gate_names=None):
    """
    Brickwall circuit:
      - Even layers: Haar-random 1q gates on all qubits
      - Layers 1 mod 4: 2q gates on (0,1), (2,3), ...
      - Layers 3 mod 4: 2q gates on (1,2), (3,4), ...
    """
    if twoq_gate_names is None:
        twoq_gate_names = CLIFFORD_2Q_GATES

    layers = []
    for l in range(depth):
        layer = []
        if l % 2 == 0:
            for q in range(n_qubits):
                layer.append(Gate((q,), haar_random_1q(rng)))
        else:
            start = 0 if (l % 4 == 1) else 1
            for q in range(start, n_qubits - 1, 2):
                layer.append(Gate((q, q + 1), rng.choice(twoq_gate_names)))
        layers.append(layer)
    return Circuit(n_qubits, layers)


def random_noise_model(rng, p_I_range=(0.85, 0.95), gate_names=None) -> dict[str, tuple]:
    """
    Random per-gate Pauli noise model.

    Args:
        p_I_range: tuple (low, high) or dict mapping gate name to (low, high).
            If dict, a "default" key can provide a fallback range.
        gate_names: iterable of gate names to include.
    """
    if gate_names is None:
        gate_names = CLIFFORD_2Q_GATES

    if isinstance(p_I_range, dict):
        default_range = p_I_range.get("default", (0.85, 0.95))
        p_I_ranges = {gate: p_I_range.get(gate, default_range) for gate in gate_names}
    else:
        p_I_ranges = {gate: p_I_range for gate in gate_names}

    noise = {}
    for gate in gate_names:
        p_I_low, p_I_high = p_I_ranges[gate]
        kraus = []
        for _ in range(2):
            p_I = rng.uniform(p_I_low, p_I_high)
            rest = 1 - p_I
            p_X = rng.uniform(0, rest)
            p_Y = rng.uniform(0, rest - p_X)
            p_Z = rest - p_X - p_Y
            kraus.append(pauli_to_kraus(np.array([p_I, p_X, p_Y, p_Z])))
        noise[gate] = tuple(kraus)
    return noise

def random_product_state(n_qubits, rng):
    state = np.array([1], dtype=complex)
    for _ in range(n_qubits):
        t, p = rng.uniform(0, np.pi), rng.uniform(0, 2*np.pi)
        state = np.kron(state, np.array([np.cos(t/2), np.exp(1j*p)*np.sin(t/2)]))
    return state


def random_basis_state(n_qubits, rng):
    idx = int(rng.integers(0, 2 ** n_qubits))
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[idx] = 1.0
    return state

def random_observable(n_qubits, rng):
    return ''.join(rng.choice(list('IXYZ')) for _ in range(n_qubits))
