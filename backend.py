# backend.py
"""
Simulation backend abstraction for PEC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

import numpy as np

from constants import DEFAULT_QISKIT_BATCH_SIZE, QISKIT_METHOD
from pec_shared import Circuit


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
    
    def __init__(self, batch_size: int = DEFAULT_QISKIT_BATCH_SIZE):
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
            "AerSimulator": AerSimulator(method=QISKIT_METHOD),
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


class QsimCirq(Backend):
    """Google qsim backend via Cirq (fastest CPU simulator)."""
    
    def __init__(self, n_threads: int = 0):
        self.n_threads = n_threads  # 0 = use all available
        self._cirq = None
        self._simulator = None
        self._kraus_cache: Dict[int, dict] = {}
    
    def _ensure_cirq(self):
        """Lazy import of Cirq and qsim."""
        if self._cirq is not None:
            return
        
        try:
            import cirq
            import qsimcirq
        except ImportError as exc:
            raise ImportError(
                "cirq and qsimcirq required. Install with: pip install cirq qsimcirq"
            ) from exc
        
        self._cirq = cirq
        
        # Configure qsim with thread count
        options = qsimcirq.QSimOptions(
            cpu_threads=self.n_threads if self.n_threads > 0 else 0,
            verbosity=0,
        )
        self._simulator = qsimcirq.QSimSimulator(options)
    
    def _get_kraus_ops(self, noise: dict) -> dict:
        """Convert noise model to Cirq Kraus channels (cached)."""
        if not noise:
            return {}
        
        noise_id = id(noise)
        if noise_id in self._kraus_cache:
            return self._kraus_cache[noise_id]
        
        self._ensure_cirq()
        cirq = self._cirq
        
        channels = {}
        for gate, (k1, k2) in noise.items():
            # Convert numpy arrays to Cirq KrausChannel
            channels[gate] = (
                cirq.KrausChannel(list(k1), key=f"{gate}_q0"),
                cirq.KrausChannel(list(k2), key=f"{gate}_q1"),
            )
        
        self._kraus_cache[noise_id] = channels
        return channels
    
    def _to_cirq_state(self, state: np.ndarray, n_qubits: int) -> np.ndarray:
        """Convert state to Cirq qubit ordering (same as ours, big-endian)."""
        # Cirq uses big-endian by default, same as our convention
        return state.copy()
    
    def _pauli_gate(self, s: int):
        """Return Cirq Pauli gate for index s."""
        self._ensure_cirq()
        cirq = self._cirq
        if s == 1:
            return cirq.X
        elif s == 2:
            return cirq.Y
        elif s == 3:
            return cirq.Z
        return None
    
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
                    # 1Q Haar-random gate
                    q = qubits[gate.qubits[0]]
                    layer_ops.append(cirq.MatrixGate(gate.content).on(q))
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
            
            # Add noise + insertions after 2Q gates
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
                
                # Pauli insertions
                for q_idx, q in [(gate.qubits[0], q0), (gate.qubits[1], q1)]:
                    s = insertions.get((l, q_idx), 0)
                    pauli = self._pauli_gate(s)
                    if pauli is not None:
                        noise_ops.append(pauli.on(q))
            
            if noise_ops:
                # Add noise ops one at a time to avoid moment conflicts
                for op in noise_ops:
                    moments.append(cirq.Moment([op]))
        
        return cirq.Circuit(moments)
    
    def _compute_expectation(
        self,
        final_state: np.ndarray,
        observable: str,
        n_qubits: int,
    ) -> float:
        """Compute expectation value of Pauli observable."""
        # Build Pauli operator matrix
        pauli_mats = {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
        }
        
        # Tensor product (observable is in qubit order: q0 q1 q2 ...)
        op = np.array([[1]], dtype=complex)
        for char in observable:
            op = np.kron(op, pauli_mats[char])
        
        # Expectation value: <psi|O|psi>
        return float(np.real(np.conj(final_state) @ op @ final_state))
    
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
        
        values = []
        for insertions in insertions_list:
            cirq_circuit = self._build_circuit(circuit, qubits, noise_channels, insertions)
            
            # Simulate with initial state
            result = self._simulator.simulate(
                cirq_circuit,
                initial_state=init_state,
            )
            
            final_state = result.final_state_vector
            exp_val = self._compute_expectation(final_state, observable, n)
            values.append(exp_val)
        
        return values




def create_backend(
    name: str = "qiskit_statevector",
    batch_size: int = DEFAULT_QISKIT_BATCH_SIZE,
    shots: int = 1024,
    n_threads: int = 0,
) -> Backend:
    """Factory function to create backend instances."""
    if name == "qiskit_statevector":
        return QiskitStatevector(batch_size=batch_size)
    elif name == "qsim" or name == "qsim_cirq":
        return QsimCirq(n_threads=n_threads)
    else:
        raise ValueError(f"Unknown backend: {name}")