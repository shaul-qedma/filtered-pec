import numpy as np 
from pec_shared import PAULIS, Circuit, Gate, STANDARD_GATES, kraus_to_probs, CLIFFORD_2Q_GATES, pauli_to_kraus
import numpy as np
from pec_shared import random_circuit, error_locations, random_noise_model

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
            P = np.kron(P, PAULIS["IXYZ".index(c)])
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
                            state = self._apply_1q(state, PAULIS[insertions[(l, q)]], q, n)
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

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    circuit = random_circuit(3, 2, rng)
    noise = random_noise_model(rng)
    sim = NoisySimulator(STANDARD_GATES, noise)
    print(f"Shared module OK: {len(error_locations(circuit, noise))} error locations")
