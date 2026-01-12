# backend.py
"""
Simulation backend abstraction for PEC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

import numpy as np

from constants import DEFAULT_BATCH_SIZE
from pec_shared import Circuit, Gate, kraus_to_probs


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
    """Qiskit Aer statevector simulation backend."""
    
    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        self.batch_size = batch_size
        self._qiskit = None
        self._noise_cache: Dict[int, dict] = {}
    
    def _ensure_qiskit(self):
        """Lazy import of Qiskit."""
        if self._qiskit is not None:
            return
        
        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit.library import UnitaryGate
            from qiskit.quantum_info import Operator, Pauli, Statevector, Kraus
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise ImportError("Qiskit and qiskit-aer required.") from exc
        
        self._qiskit = {
            "QuantumCircuit": QuantumCircuit,
            "UnitaryGate": UnitaryGate,
            "Operator": Operator,
            "Pauli": Pauli,
            "Statevector": Statevector,
            "Kraus": Kraus,
            "AerSimulator": AerSimulator(method="statevector"),
        }
    
    def _get_noise_instructions(self, noise: dict) -> dict:
        """Convert noise model to Qiskit Kraus instructions (cached)."""
        if not noise:
            return {}
        
        noise_id = id(noise)
        if noise_id in self._noise_cache:
            return self._noise_cache[noise_id]
        
        self._ensure_qiskit()
        Kraus = self._qiskit["Kraus"]  # type: ignore
        
        instr = {
            gate: (Kraus(list(k1)).to_instruction(), Kraus(list(k2)).to_instruction())
            for gate, (k1, k2) in noise.items()
        }
        self._noise_cache[noise_id] = instr
        return instr
    
    def _to_statevector(self, state: np.ndarray, n_qubits: int) -> np.ndarray:
        """Convert state to Qiskit qubit ordering."""
        if n_qubits <= 1:
            return state.copy()
        reshaped = state.reshape([2] * n_qubits)
        return reshaped.transpose(list(reversed(range(n_qubits)))).reshape(-1)
    
    def _build_circuit(
        self,
        circuit: Circuit,
        init_sv: np.ndarray,
        noise_instr: dict,
        insertions: Dict[Tuple[int, int], int],
    ):
        """Build Qiskit circuit with noise and Pauli insertions."""
        self._ensure_qiskit()
        QuantumCircuit, UnitaryGate = self._qiskit["QuantumCircuit"], self._qiskit["UnitaryGate"]  # type: ignore
        
        n = circuit.n_qubits
        qc = QuantumCircuit(n)
        qc.set_statevector(init_sv)

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

                if name in noise_instr:
                    instr1, instr2 = noise_instr[name]
                    qc.append(instr1, [q0])
                    qc.append(instr2, [q1])

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
        
        self._ensure_qiskit()
        Operator, Pauli, Statevector = self._qiskit["Operator"], self._qiskit["Pauli"], self._qiskit["Statevector"]  # type: ignore

        init_sv = self._to_statevector(initial_state, circuit.n_qubits)
        pauli_obs = Operator(Pauli(observable[::-1]))
        noise_instr = self._get_noise_instructions(noise)
        
        circuits = []
        for insertions in insertions_list:
            qc = self._build_circuit(circuit, init_sv, noise_instr, insertions)
            qc.save_statevector()
            circuits.append(qc)
        
        result = self._qiskit["AerSimulator"].run(circuits, shots=1).result()  # type: ignore
        
        values = []
        for idx in range(len(circuits)):
            state = result.data(idx)["statevector"]
            sv = state if isinstance(state, Statevector) else Statevector(state)
            values.append(float(np.real(sv.expectation_value(pauli_obs))))
        
        return values


class CirqSimulator(Backend):
    """Cirq density-matrix simulation backend."""

    def __init__(self, n_threads: int = 0):
        self.n_threads = n_threads  # Thread control not exposed by Cirq simulator.
        self._cirq = None
        self._simulator = None
        self._kraus_cache: Dict[int, dict] = {}

    def _ensure_cirq(self):
        """Lazy import of Cirq."""
        if self._cirq is not None:
            return

        try:
            import cirq
        except ImportError as exc:
            raise ImportError("cirq required. Install with: pip install cirq") from exc

        self._cirq = cirq
        if not hasattr(cirq, "DensityMatrixSimulator"):
            raise RuntimeError("Cirq DensityMatrixSimulator unavailable; upgrade Cirq.")
        self._simulator = cirq.DensityMatrixSimulator()

    def _make_kraus_channel(self, ops: List[np.ndarray], key: str):
        self._ensure_cirq()
        cirq = self._cirq
        try:
            return cirq.KrausChannel(ops, key=key)
        except TypeError:
            return cirq.KrausChannel(ops)

    def _get_kraus_ops(self, noise: dict) -> dict:
        """Convert noise model to Cirq Kraus channels (cached)."""
        if not noise:
            return {}

        noise_id = id(noise)
        if noise_id in self._kraus_cache:
            return self._kraus_cache[noise_id]

        self._ensure_cirq()

        channels = {}
        for gate, (k1, k2) in noise.items():
            channels[gate] = (
                self._make_kraus_channel(list(k1), f"{gate}_q0"),
                self._make_kraus_channel(list(k2), f"{gate}_q1"),
            )

        self._kraus_cache[noise_id] = channels
        return channels

    def _to_cirq_state(self, state: np.ndarray, n_qubits: int) -> np.ndarray:
        """Convert state to Cirq's little-endian ordering."""
        if n_qubits <= 1:
            return state.copy()
        reshaped = state.reshape([2] * n_qubits)
        return reshaped.transpose(list(reversed(range(n_qubits)))).reshape(-1)

    def _pauli_gate(self, s: int):
        """Return Cirq Pauli gate for index s."""
        self._ensure_cirq()
        cirq = self._cirq
        if s == 1:
            return cirq.X
        if s == 2:
            return cirq.Y
        if s == 3:
            return cirq.Z
        return None

    def _matrix_gate(self, matrix: np.ndarray):
        self._ensure_cirq()
        cirq = self._cirq
        if hasattr(cirq, "MatrixGate"):
            return cirq.MatrixGate(matrix)
        if hasattr(cirq, "SingleQubitMatrixGate"):
            return cirq.SingleQubitMatrixGate(matrix)
        raise RuntimeError("Cirq matrix gate unavailable; upgrade Cirq.")

    def _build_circuit(
        self,
        circuit: Circuit,
        qubits: list,
        noise_channels: dict,
        insertions: Dict[Tuple[int, int], int],
    ):
        """Build Cirq circuit with noise and Pauli insertions."""
        self._ensure_cirq()
        cirq = self._cirq

        moments = []

        for l, layer in enumerate(circuit.layers):
            layer_ops = []

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

            if layer_ops:
                moments.append(cirq.Moment(layer_ops))

            noise_ops = []
            for gate in layer:
                if isinstance(gate.content, np.ndarray):
                    continue

                name = gate.content
                if name not in noise_channels:
                    continue

                q0, q1 = qubits[gate.qubits[0]], qubits[gate.qubits[1]]
                chan0, chan1 = noise_channels[name]

                noise_ops.append(chan0.on(q0))
                noise_ops.append(chan1.on(q1))

                for q_idx, q in [(gate.qubits[0], q0), (gate.qubits[1], q1)]:
                    s = insertions.get((l, q_idx), 0)
                    pauli = self._pauli_gate(s)
                    if pauli is not None:
                        noise_ops.append(pauli.on(q))

            if noise_ops:
                for op in noise_ops:
                    moments.append(cirq.Moment([op]))

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

        self._ensure_cirq()
        cirq = self._cirq

        n = circuit.n_qubits
        qubits = cirq.LineQubit.range(n)

        init_state = self._to_cirq_state(initial_state, n)
        noise_channels = self._get_kraus_ops(noise)
        pauli_op = self._pauli_operator(observable)

        values = []
        for insertions in insertions_list:
            cirq_circuit = self._build_circuit(circuit, qubits, noise_channels, insertions)

            result = self._simulator.simulate(
                cirq_circuit,
                initial_state=init_state,
                qubit_order=qubits,
            )

            final_state = getattr(result, "final_density_matrix", None)
            if final_state is None:
                final_state = getattr(result, "final_state_vector", None)
            if callable(final_state):
                final_state = final_state()

            values.append(self._compute_expectation(final_state, pauli_op))

        return values


class StimClifford(Backend):
    """Stim-backed stabilizer simulator for Clifford circuits with Pauli noise."""

    def __init__(self):
        self._stim = None
        self._noise_cache: Dict[int, dict] = {}
        self._gate_cache: Dict[Tuple[str, Tuple[int, ...]], "stim.Circuit"] = {}

    def _ensure_stim(self):
        if self._stim is not None:
            return
        try:
            import stim  # type: ignore
        except ImportError as exc:
            raise ImportError("stim required. Install with: pip install stim") from exc
        self._stim = stim

    def _gate_circuit(self, op: str, targets: Tuple[int, ...]):
        self._ensure_stim()
        stim = self._stim
        key = (op, targets)
        if key not in self._gate_cache:
            circ = stim.Circuit()
            circ.append(op, list(targets))
            self._gate_cache[key] = circ
        return self._gate_cache[key]

    def _pauli_channel_lambda(self, probs: np.ndarray, pauli: str) -> float:
        if pauli == "I":
            return 1.0
        p_i, p_x, p_y, p_z = [float(x) for x in probs]
        if pauli == "X":
            return p_i + p_x - p_y - p_z
        if pauli == "Y":
            return p_i - p_x + p_y - p_z
        if pauli == "Z":
            return p_i - p_x - p_y + p_z
        raise ValueError(f"Unknown Pauli: {pauli}")

    def _pauli_on_qubit(self, pauli, qubit: int) -> str:
        if hasattr(pauli, "xs") and hasattr(pauli, "zs"):
            x = bool(pauli.xs[qubit])
            z = bool(pauli.zs[qubit])
            if x and z:
                return "Y"
            if x:
                return "X"
            if z:
                return "Z"
            return "I"

        pauli_str = str(pauli)
        if pauli_str.startswith("-i"):
            pauli_str = pauli_str[2:]
        elif pauli_str.startswith("i"):
            pauli_str = pauli_str[1:]
        elif pauli_str.startswith(("-", "+")):
            pauli_str = pauli_str[1:]
        return pauli_str[qubit]

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

    def _get_noise_probs(self, noise: dict) -> dict:
        if not noise:
            return {}
        noise_id = id(noise)
        if noise_id in self._noise_cache:
            return self._noise_cache[noise_id]

        probs = {}
        for gate, (k1, k2) in noise.items():
            probs[gate] = (kraus_to_probs(k1), kraus_to_probs(k2))
        self._noise_cache[noise_id] = probs
        return probs

    def _apply_ops(self, pauli, ops: List[Tuple[str, Tuple[int, ...]]]):
        for op, targets in ops:
            pauli = pauli.after(self._gate_circuit(op, targets))
        return pauli

    def _apply_gate(self, pauli, gate: Gate):
        if isinstance(gate.content, np.ndarray):
            seq = getattr(gate, "clifford_seq", None)
            if seq is None:
                raise ValueError("Stim backend requires clifford_seq on 1Q gates.")
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

        self._ensure_stim()
        stim = self._stim

        n_qubits = circuit.n_qubits
        if len(observable) != n_qubits:
            raise ValueError("Observable length must match circuit.n_qubits.")

        bits = self._basis_bits(initial_state, n_qubits)
        noise_probs = self._get_noise_probs(noise)

        values = []
        for insertions in insertions_list:
            pauli = stim.PauliString(observable)
            scale = 1.0
            for layer_idx in range(len(circuit.layers) - 1, -1, -1):
                layer = circuit.layers[layer_idx]
                for gate in layer:
                    if isinstance(gate.content, np.ndarray):
                        pauli = self._apply_gate(pauli, gate)
                        continue

                    name = gate.content
                    q0, q1 = int(gate.qubits[0]), int(gate.qubits[1])
                    if name in noise_probs:
                        probs0, probs1 = noise_probs[name]
                        for qubit, probs in ((q0, probs0), (q1, probs1)):
                            s = int(insertions.get((layer_idx, qubit), 0))
                            if s == 1:
                                pauli = self._apply_ops(pauli, [("X", (qubit,))])
                            elif s == 2:
                                pauli = self._apply_ops(pauli, [("Y", (qubit,))])
                            elif s == 3:
                                pauli = self._apply_ops(pauli, [("Z", (qubit,))])

                            pauli_kind = self._pauli_on_qubit(pauli, qubit)
                            scale *= self._pauli_channel_lambda(probs, pauli_kind)

                    pauli = self._apply_gate(pauli, gate)

            values.append(scale * self._basis_expectation(pauli, bits))

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
