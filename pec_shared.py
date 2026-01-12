"""Shared utilities for PEC implementations."""

import numpy as np
from dataclasses import dataclass

ETA = np.array([[+1,+1,+1,+1], [+1,+1,-1,-1], [+1,-1,+1,-1], [+1,-1,-1,+1]], dtype=np.float64)

PAULI = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex)
]

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

def pauli_kraus(p):
    return tuple(np.sqrt(p[i]) * PAULI[i] for i in range(4))

def kraus_to_probs(kraus):
    return np.array([np.real(np.trace(K.conj().T @ K)) / 2 for K in kraus])

def quasi_probs(p):
    lam = ETA @ p
    return 0.25 * ETA @ (1.0 / lam)

class NoisySimulator:
    def __init__(self, gate_library, noise_model):
        self.gates = gate_library
        self.noise = noise_model
    
    def _apply_1q(self, state, U, q, n):
        state = state.reshape([2] * n)
        state = np.moveaxis(state, q, 0)
        shape = state.shape
        state = U @ state.reshape(2, -1)
        state = np.moveaxis(state.reshape(shape), 0, q)
        return state.reshape(-1)

    def _apply_2q(self, state, U, q1, q2, n):
        state = state.reshape([2] * n)
        state = np.moveaxis(state, [q1, q2], [0, 1])
        shape = state.shape
        state = U @ state.reshape(4, -1)
        state = np.moveaxis(state.reshape(shape), [0, 1], [q1, q2])
        return state.reshape(-1)

    def _apply_kraus(self, state, kraus_ops, q, n):
        probs = kraus_to_probs(kraus_ops)
        idx = np.random.choice(len(kraus_ops), p=probs)
        K_normalized = kraus_ops[idx] / np.sqrt(probs[idx])
        return self._apply_1q(state, K_normalized, q, n)

    def _build_obs(self, obs):
        P = np.array([[1]], dtype=complex)
        for c in obs:
            P = np.kron(P, PAULI["IXYZ".index(c)])
        return P

    def _expectation(self, state, obs):
        P = self._build_obs(obs) if isinstance(obs, str) else obs
        return np.real(np.conj(state) @ P @ state)

    def run(self, circuit, obs, init, insertions=None):
        insertions = insertions or {}
        state = init.copy()
        n = circuit.n_qubits
        for l, layer in enumerate(circuit.layers):
            for gate in layer:
                if isinstance(gate.content, np.ndarray):
                    state = self._apply_1q(state, gate.content, gate.qubits[0], n)
                else:
                    U = self.gates[gate.content]
                    state = self._apply_2q(state, U, gate.qubits[0], gate.qubits[1], n)
                    if gate.content in self.noise:
                        kraus1, kraus2 = self.noise[gate.content]
                        state = self._apply_kraus(state, kraus1, gate.qubits[0], n)
                        state = self._apply_kraus(state, kraus2, gate.qubits[1], n)
                    for q in gate.qubits:
                        if (l, q) in insertions:
                            state = self._apply_1q(state, PAULI[insertions[(l, q)]], q, n)
        return self._expectation(state, obs)

    def ideal(self, circuit, obs, init):
        state = init.copy()
        n = circuit.n_qubits
        for layer in circuit.layers:
            for gate in layer:
                if isinstance(gate.content, np.ndarray):
                    state = self._apply_1q(state, gate.content, gate.qubits[0], n)
                else:
                    state = self._apply_2q(state, self.gates[gate.content], gate.qubits[0], gate.qubits[1], n)
        return self._expectation(state, obs)

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
            kraus.append(pauli_kraus(np.array([p_I, p_X, p_Y, p_Z])))
        noise[gate] = tuple(kraus)
    return noise

def random_product_state(n_qubits, rng):
    state = np.array([1], dtype=complex)
    for _ in range(n_qubits):
        t, p = rng.uniform(0, np.pi), rng.uniform(0, 2*np.pi)
        state = np.kron(state, np.array([np.cos(t/2), np.exp(1j*p)*np.sin(t/2)]))
    return state

def random_observable(n_qubits, rng):
    return ''.join(rng.choice(list('IXYZ')) for _ in range(n_qubits))

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    circuit = random_circuit(3, 2, rng)
    noise = random_noise_model(rng)
    sim = NoisySimulator(STANDARD_GATES, noise)
    print(f"Shared module OK: {len(error_locations(circuit, noise))} error locations")
