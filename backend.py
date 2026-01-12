# backend.py
"""
Simulation backend abstraction for PEC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

import numpy as np

from constants import DEFAULT_BATCH_SIZE
from pec_shared import Circuit, Gate

try:
    import cirq
except ImportError:  # pragma: no cover - optional dependency
    cirq = None  # type: ignore[assignment]

try:
    import stim  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    stim = None  # type: ignore[assignment]

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator, Pauli, Statevector
except ImportError:  # pragma: no cover - optional dependency
    QuantumCircuit = None  # type: ignore[assignment]
    UnitaryGate = None  # type: ignore[assignment]
    Operator = None  # type: ignore[assignment]
    Pauli = None  # type: ignore[assignment]
    Statevector = None  # type: ignore[assignment]

class Backend(ABC):
    """Abstract base class for circuit simulation backends."""
    
    @abstractmethod
    def expectation(
        self,
        circuit: Circuit,
        observable: str,
        initial_state: np.ndarray,
        noise: dict,
        insertions: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> float:
        """Compute expectation value for a single circuit configuration."""
        pass
    
    @abstractmethod
    def batch_expectations(
        self,
        circuit: Circuit,
        observable: str,
        initial_state: np.ndarray,
        noise: dict,
        insertions_list: Optional[List[Dict[Tuple[int, int], int]]] = None,
    ) -> List[float]:
        """Compute expectation values for a batch of circuit configurations."""
        pass
    
    @property
    def supports_batching(self) -> bool:
        """Whether this backend benefits from batched execution."""
        return True


class QiskitStatevector(Backend):
    """Qiskit statevector backend for unitary circuits with Pauli insertions."""

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        if QuantumCircuit is None or Statevector is None:
            raise ImportError("Qiskit required.")
        self.batch_size = batch_size

    def _to_statevector(self, state: np.ndarray, n_qubits: int) -> np.ndarray:
        """Convert state to Qiskit qubit ordering."""
        if n_qubits <= 1:
            return state.copy()
        reshaped = state.reshape([2] * n_qubits)
        return reshaped.transpose(list(reversed(range(n_qubits)))).reshape(-1)

    def _build_circuit(
        self,
        circuit: Circuit,
        insertions: Dict[Tuple[int, int], int],
    ):
        """Build Qiskit circuit with Pauli insertions."""
        n = circuit.n_qubits
        qc = QuantumCircuit(n)

        for l, layer in enumerate(circuit.layers):
            for gate in layer:
                if isinstance(gate.content, np.ndarray):
                    qc.append(UnitaryGate(gate.content), [gate.qubits[0]])
                    continue

                name = gate.content
                q0, q1 = gate.qubits

                if name == "CNOT":
                    qc.cx(q0, q1)
                elif name == "CNOT_R":
                    qc.cx(q1, q0)
                elif name == "CZ":
                    qc.cz(q0, q1)
                elif name == "SWAP":
                    qc.swap(q0, q1)
                else:
                    raise ValueError(f"Unsupported 2q gate: {name}")

                for q in (q0, q1):
                    s = insertions.get((l, q), 0)
                    if s == 1:
                        qc.x(q)
                    elif s == 2:
                        qc.y(q)
                    elif s == 3:
                        qc.z(q)

        return qc

    def expectation(
        self,
        circuit: Circuit,
        observable: str,
        initial_state: np.ndarray,
        noise: dict,
        insertions: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> float:
        """Single circuit expectation value."""
        return self.batch_expectations(
            circuit, observable, initial_state, noise, [insertions or {}]
        )[0]

    def batch_expectations(
        self,
        circuit: Circuit,
        observable: str,
        initial_state: np.ndarray,
        noise: dict,
        insertions_list: Optional[List[Dict[Tuple[int, int], int]]] = None,
    ) -> List[float]:
        """Batched circuit expectation values."""
        if insertions_list is None:
            insertions_list = [{}]

        init_sv = self._to_statevector(initial_state, circuit.n_qubits)
        pauli_obs = Operator(Pauli(observable[::-1]))

        values = []
        for insertions in insertions_list:
            qc = self._build_circuit(circuit, insertions)
            sv = Statevector(init_sv).evolve(qc)
            values.append(float(np.real(sv.expectation_value(pauli_obs))))

        return values


class CirqSimulator(Backend):
    """Cirq statevector simulation backend with Pauli insertions."""

    def __init__(self, n_threads: int = 0):
        if cirq is None:
            raise ImportError("cirq-core required. Install with: uv add 'cirq-core<1.0'")
        self.n_threads = n_threads  # Thread control not exposed by Cirq simulator.
        self._simulator = cirq.Simulator()

    def _to_cirq_state(self, state: np.ndarray, n_qubits: int) -> np.ndarray:
        """Convert state to Cirq's little-endian ordering."""
        if n_qubits <= 1:
            return state.copy()
        reshaped = state.reshape([2] * n_qubits)
        return reshaped.transpose(list(reversed(range(n_qubits)))).reshape(-1)

    def _pauli_gate(self, s: int):
        """Return Cirq Pauli gate for index s."""
        if s == 1:
            return cirq.X
        if s == 2:
            return cirq.Y
        if s == 3:
            return cirq.Z
        return None

    def _matrix_gate(self, matrix: np.ndarray):
        if hasattr(cirq, "MatrixGate"):
            return cirq.MatrixGate(matrix)
        if hasattr(cirq, "SingleQubitMatrixGate"):
            return cirq.SingleQubitMatrixGate(matrix)
        raise RuntimeError("Cirq matrix gate unavailable; upgrade Cirq.")

    def _build_circuit(
        self,
        circuit: Circuit,
        qubits: list,
        insertions: Dict[Tuple[int, int], int],
    ):
        """Build Cirq circuit with Pauli insertions."""
        moments = []

        for l, layer in enumerate(circuit.layers):
            layer_ops = []
            insertion_ops = []

            for gate in layer:
                if isinstance(gate.content, np.ndarray):
                    q = qubits[gate.qubits[0]]
                    layer_ops.append(self._matrix_gate(gate.content).on(q))
                    continue

                name = gate.content
                q0, q1 = qubits[gate.qubits[0]], qubits[gate.qubits[1]]

                if name == "CNOT":
                    layer_ops.append(cirq.CNOT(q0, q1))
                elif name == "CNOT_R":
                    layer_ops.append(cirq.CNOT(q1, q0))
                elif name == "CZ":
                    layer_ops.append(cirq.CZ(q0, q1))
                elif name == "SWAP":
                    layer_ops.append(cirq.SWAP(q0, q1))
                else:
                    raise ValueError(f"Unsupported 2q gate: {name}")

                for q_idx, q in [(gate.qubits[0], q0), (gate.qubits[1], q1)]:
                    s = insertions.get((l, q_idx), 0)
                    pauli = self._pauli_gate(s)
                    if pauli is not None:
                        insertion_ops.append(pauli.on(q))

            if layer_ops:
                moments.append(cirq.Moment(layer_ops))
            if insertion_ops:
                moments.append(cirq.Moment(insertion_ops))

        return cirq.Circuit(moments)

    def _pauli_operator(self, observable: str) -> np.ndarray:
        pauli_mats = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }
        op = np.array([[1]], dtype=complex)
        for char in observable[::-1]:
            op = np.kron(op, pauli_mats[char])
        return op

    def _compute_expectation(
        self,
        final_state: np.ndarray,
        pauli_op: np.ndarray,
    ) -> float:
        """Compute expectation value for a precomputed Pauli operator."""
        state = np.asarray(final_state)
        if state.ndim == 1:
            return float(np.real(np.conj(state) @ pauli_op @ state))
        return float(np.real(np.trace(state @ pauli_op)))

    def expectation(
        self,
        circuit: Circuit,
        observable: str,
        initial_state: np.ndarray,
        noise: dict,
        insertions: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> float:
        """Single circuit expectation value."""
        return self.batch_expectations(
            circuit, observable, initial_state, noise, [insertions or {}]
        )[0]

    def batch_expectations(
        self,
        circuit: Circuit,
        observable: str,
        initial_state: np.ndarray,
        noise: dict,
        insertions_list: Optional[List[Dict[Tuple[int, int], int]]] = None,
    ) -> List[float]:
        """Batched circuit expectation values."""
        if insertions_list is None:
            insertions_list = [{}]

        n = circuit.n_qubits
        qubits = cirq.LineQubit.range(n)

        init_state = self._to_cirq_state(initial_state, n)
        pauli_op = self._pauli_operator(observable)

        values = []
        for insertions in insertions_list:
            cirq_circuit = self._build_circuit(circuit, qubits, insertions)

            result = self._simulator.simulate(
                cirq_circuit,
                initial_state=init_state,
                qubit_order=qubits,
            )

            final_state = getattr(result, "final_state_vector", None)
            if callable(final_state):
                final_state = final_state()
            if final_state is None:
                raise RuntimeError("Cirq simulator did not return a final state.")
            final_state = np.asarray(final_state)

            values.append(self._compute_expectation(final_state, pauli_op))

        return values


class StimClifford(Backend):
    """Stim-backed stabilizer simulator for Clifford circuits with Pauli noise."""

    def __init__(self):
        if stim is None:
            raise ImportError("stim required. Install with: uv add stim")
        self._gate_cache: Dict[Tuple[str, Tuple[int, ...]], "stim.Circuit"] = {}

    def _gate_circuit(self, op: str, targets: Tuple[int, ...]):
        key = (op, targets)
        if key not in self._gate_cache:
            circ = stim.Circuit()
            circ.append(op, list(targets))  # type: ignore[arg-type]
            self._gate_cache[key] = circ
        return self._gate_cache[key]

    def _basis_bits(self, state: Optional[np.ndarray], n_qubits: int) -> List[int]:
        if state is None:
            return [0] * n_qubits
        if state.ndim != 1 or state.size != 2 ** n_qubits:
            raise ValueError("Stim backend requires a computational basis state.")
        mags = np.abs(state)
        idx = int(np.argmax(mags))
        if not np.isclose(np.sum(mags > 1e-8), 1):
            raise ValueError("Stim backend only supports computational basis states.")
        return [(idx >> i) & 1 for i in range(n_qubits)]

    def _basis_expectation(self, pauli, bits: List[int]) -> float:
        pauli_str = str(pauli)
        phase = 1.0 + 0.0j
        if pauli_str.startswith("-i"):
            phase = -1j
            pauli_str = pauli_str[2:]
        elif pauli_str.startswith("i"):
            phase = 1j
            pauli_str = pauli_str[1:]
        elif pauli_str.startswith("-"):
            phase = -1.0
            pauli_str = pauli_str[1:]
        elif pauli_str.startswith("+"):
            pauli_str = pauli_str[1:]

        for q, p in enumerate(pauli_str):
            if p in ("X", "Y"):
                return 0.0
            if p == "Z" and bits[q]:
                phase *= -1
        return float(np.real(phase))

    def _apply_ops(self, pauli, ops: List[Tuple[str, Tuple[int, ...]]]):
        for op, targets in ops:
            pauli = pauli.before(self._gate_circuit(op, targets))
        return pauli

    def _apply_gate(self, pauli, gate: Gate):
        if isinstance(gate.content, np.ndarray):
            seq = getattr(gate, "clifford_seq", None)
            if seq is None:
                cliff_idx = getattr(gate, "clifford_idx", None)
                if cliff_idx is None:
                    raise ValueError("Stim backend requires clifford_seq or clifford_idx on 1Q gates.")
                from clifford import CLIFFORD_1Q_DECOMP

                seq = CLIFFORD_1Q_DECOMP[int(cliff_idx)]
            q = int(gate.qubits[0])
            ops = [(op, (q,)) for op in seq]
            return self._apply_ops(pauli, ops)

        if isinstance(gate.content, str):
            q0, q1 = int(gate.qubits[0]), int(gate.qubits[1])
            if gate.content == "CNOT":
                return self._apply_ops(pauli, [("CX", (q0, q1))])
            if gate.content == "CNOT_R":
                return self._apply_ops(pauli, [("CX", (q1, q0))])
            if gate.content == "CZ":
                return self._apply_ops(pauli, [("CZ", (q0, q1))])
            if gate.content == "SWAP":
                return self._apply_ops(pauli, [("SWAP", (q0, q1))])
            raise ValueError(f"Unsupported Clifford 2Q gate: {gate.content}")

        raise ValueError("Unsupported Clifford gate content type.")

    def expectation(
        self,
        circuit: Circuit,
        observable: str,
        initial_state: np.ndarray,
        noise: dict,
        insertions: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> float:
        return self.batch_expectations(
            circuit, observable, initial_state, noise, [insertions or {}]
        )[0]

    def batch_expectations(
        self,
        circuit: Circuit,
        observable: str,
        initial_state: np.ndarray,
        noise: dict,
        insertions_list: Optional[List[Dict[Tuple[int, int], int]]] = None,
    ) -> List[float]:
        if insertions_list is None:
            insertions_list = [{}]

        n_qubits = circuit.n_qubits
        if len(observable) != n_qubits:
            raise ValueError("Observable length must match circuit.n_qubits.")

        bits = self._basis_bits(initial_state, n_qubits)

        values = []
        for insertions in insertions_list:
            pauli = stim.PauliString(observable)
            for layer_idx in range(len(circuit.layers) - 1, -1, -1):
                layer = circuit.layers[layer_idx]
                for gate in layer:
                    if isinstance(gate.content, np.ndarray):
                        pauli = self._apply_gate(pauli, gate)
                        continue

                    name = gate.content
                    q0, q1 = int(gate.qubits[0]), int(gate.qubits[1])
                    if isinstance(name, str):
                        for qubit in (q0, q1):
                            s = int(insertions.get((layer_idx, qubit), 0))
                            if s == 1:
                                pauli = self._apply_ops(pauli, [("X", (qubit,))])
                            elif s == 2:
                                pauli = self._apply_ops(pauli, [("Y", (qubit,))])
                            elif s == 3:
                                pauli = self._apply_ops(pauli, [("Z", (qubit,))])

                    pauli = self._apply_gate(pauli, gate)

            values.append(self._basis_expectation(pauli, bits))

        return values




def create_backend(
    name: str = "qiskit_statevector",
    batch_size: int = DEFAULT_BATCH_SIZE,
    shots: int = 1024,
    n_threads: int = 0,
) -> Backend:
    """Factory function to create backend instances."""
    if name == "qiskit_statevector":
        return QiskitStatevector(batch_size=batch_size)
    if name == "cirq":
        return CirqSimulator(n_threads=n_threads)
    if name == "stim":
        return StimClifford()
    else:
        raise ValueError(f"Unknown backend: {name}")
