import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.extensions import UnitaryGate


class QuantumProcessor:
    """Quantum processing unit for n-qubit operations."""
    
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.simulator = Aer.get_backend('statevector_simulator')
        self.qr = QuantumRegister(n_qubits)
        self.cr = ClassicalRegister(n_qubits)
        
        # Parameters for variational circuits
        self.theta = Parameter('θ')
        self.phi = Parameter('φ')
        self.lambda_param = Parameter('λ')
        
        # Error correction parameters
        self.syndrome_qr = QuantumRegister(3, 'syndrome')
        self.syndrome_cr = ClassicalRegister(3, 'syndrome_meas')
    
    def create_quantum_circuit(self) -> QuantumCircuit:
        """Create a base quantum circuit."""
        return QuantumCircuit(self.qr, self.cr)
    
    def create_error_corrected_circuit(self) -> QuantumCircuit:
        """Create a circuit with error correction capabilities."""
        return QuantumCircuit(
            self.qr,
            self.syndrome_qr,
            self.cr,
            self.syndrome_cr
        )
    
    def apply_bit_flip_code(self, circuit: QuantumCircuit, qubit: int) -> None:
        """Apply 3-qubit bit flip code."""
        # Encode logical qubit
        circuit.cx(qubit, qubit + 1)
        circuit.cx(qubit, qubit + 2)
        
        # Add syndrome measurement
        circuit.cx(qubit, self.syndrome_qr[0])
        circuit.cx(qubit + 1, self.syndrome_qr[1])
        circuit.cx(qubit + 2, self.syndrome_qr[2])
        
        # Measure syndrome
        for i in range(3):
            circuit.measure(self.syndrome_qr[i], self.syndrome_cr[i])
    
    def apply_phase_flip_code(self, circuit: QuantumCircuit, qubit: int) -> None:
        """Apply 3-qubit phase flip code."""
        # Encode logical qubit
        circuit.h([qubit, qubit + 1, qubit + 2])
        circuit.cx(qubit, qubit + 1)
        circuit.cx(qubit, qubit + 2)
        
        # Add syndrome measurement
        circuit.h([qubit, qubit + 1, qubit + 2])
        circuit.cx(qubit, self.syndrome_qr[0])
        circuit.cx(qubit + 1, self.syndrome_qr[1])
        circuit.cx(qubit + 2, self.syndrome_qr[2])
        circuit.h([qubit, qubit + 1, qubit + 2])
        
        # Measure syndrome
        for i in range(3):
            circuit.measure(self.syndrome_qr[i], self.syndrome_cr[i])
    
    def create_adiabatic_circuit(
        self,
        initial_hamiltonian: Operator,
        final_hamiltonian: Operator,
        steps: int = 100
    ) -> QuantumCircuit:
        """Create adiabatic state preparation circuit."""
        circuit = self.create_quantum_circuit()
        
        # Initialize in ground state of initial Hamiltonian
        initial_state = Statevector.from_operator(initial_hamiltonian)
        circuit.initialize(initial_state, self.qr)
        
        # Adiabatic evolution
        for t in range(steps):
            s = t / (steps - 1)  # Adiabatic parameter
            # Interpolate between Hamiltonians
            current_h = (1 - s) * initial_hamiltonian + s * final_hamiltonian
            # Convert to unitary for small time step
            dt = 0.01
            unitary = UnitaryGate((-1j * dt * current_h).exp())
            circuit.append(unitary, self.qr)
        
        return circuit
    
    def create_variational_ansatz(
        self,
        depth: int,
        entanglement: str = 'full'
    ) -> QuantumCircuit:
        """Create a variational ansatz with specified depth."""
        circuit = self.create_quantum_circuit()
        
        for d in range(depth):
            # Single-qubit rotations
            for i in range(self.n_qubits):
                theta = Parameter(f'θ_{d}_{i}')
                phi = Parameter(f'φ_{d}_{i}')
                lambda_param = Parameter(f'λ_{d}_{i}')
                circuit.u3(theta, phi, lambda_param, i)
            
            # Entangling layers
            if entanglement == 'full':
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        circuit.cx(i, j)
            elif entanglement == 'linear':
                for i in range(self.n_qubits - 1):
                    circuit.cx(i, i + 1)
            elif entanglement == 'circular':
                for i in range(self.n_qubits):
                    circuit.cx(i, (i + 1) % self.n_qubits)
        
        return circuit
    
    def measure_pauli_expectation(
        self,
        circuit: QuantumCircuit,
        pauli_string: str
    ) -> float:
        """Measure expectation value of Pauli string operator."""
        meas_circuit = circuit.copy()
        
        # Apply basis rotations
        for i, p in enumerate(pauli_string):
            if p == 'X':
                meas_circuit.h(i)
            elif p == 'Y':
                meas_circuit.sdg(i)
                meas_circuit.h(i)
        
        # Measure in computational basis
        meas_circuit.measure_all()
        result = execute(meas_circuit, self.simulator).result()
        counts = result.get_counts()
        
        # Calculate expectation value
        expectation = 0.0
        total = sum(counts.values())
        for state, count in counts.items():
            # Count parity of 1s
            parity = (-1) ** sum(int(b) for b in state)
            expectation += parity * count / total
        
        return float(expectation)
    
    def measure_entanglement_entropy(
        self,
        circuit: QuantumCircuit,
        subsystem: List[int]
    ) -> float:
        """Measure entanglement entropy of a subsystem."""
        # Get final state
        state = execute(circuit, self.simulator).result().get_statevector()
        dm = DensityMatrix(state)
        
        # Partial trace over complement of subsystem
        reduced_dm = dm.partial_trace(sorted(set(range(self.n_qubits)) - set(subsystem)))
        
        # Calculate von Neumann entropy
        eigenvals = np.real(np.linalg.eigvals(reduced_dm.data))
        entropy = -sum(l * np.log2(l) for l in eigenvals if l > 1e-10)
        
        return float(entropy)
    
    def analyze_circuit_properties(
        self,
        circuit: QuantumCircuit
    ) -> Dict[str, float]:
        """Analyze various properties of the quantum circuit."""
        properties = {}
        
        # Measure all Pauli correlations
        for p1 in ['I', 'X', 'Y', 'Z']:
            for p2 in ['I', 'X', 'Y', 'Z']:
                if p1 != 'I' or p2 != 'I':
                    pauli_string = 'I' * (self.n_qubits - 2) + p1 + p2
                    properties[f'corr_{p1}{p2}'] = self.measure_pauli_expectation(
                        circuit, pauli_string
                    )
        
        # Measure entanglement properties
        for i in range(1, self.n_qubits):
            subsystem = list(range(i))
            properties[f'entropy_{i}'] = self.measure_entanglement_entropy(
                circuit, subsystem
            )
        
        # Add tomography measurements
        measurements = self.measure_state_tomography(circuit)
        for basis, counts in measurements.items():
            total = sum(counts.values())
            for state, count in counts.items():
                properties[f'{basis}_{state}'] = count / total
        
        return properties
    
    def create_bell_pair(self) -> QuantumCircuit:
        """Create a Bell pair for quantum entanglement."""
        circuit = self.create_quantum_circuit()
        circuit.h(0)  # Put first qubit in superposition
        circuit.cx(0, 1)  # CNOT to entangle qubits
        return circuit
    
    def create_ghz_state(self) -> QuantumCircuit:
        """Create a GHZ state for multi-qubit entanglement."""
        circuit = self.create_quantum_circuit()
        circuit.h(0)  # Put first qubit in superposition
        # Entangle all qubits
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
        return circuit
    
    def apply_quantum_fourier_transform(
        self,
        circuit: QuantumCircuit,
        qubits: Tuple[int, ...]
    ) -> None:
        """Apply Quantum Fourier Transform to specified qubits."""
        n = len(qubits)
        for i in range(n):
            circuit.h(qubits[i])
            for j in range(i + 1, n):
                theta = np.pi / float(2 ** (j - i))
                circuit.cp(theta, qubits[j], qubits[i])
    
    def apply_inverse_quantum_fourier_transform(
        self,
        circuit: QuantumCircuit,
        qubits: Tuple[int, ...]
    ) -> None:
        """Apply inverse QFT to specified qubits."""
        n = len(qubits)
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                theta = -np.pi / float(2 ** (j - i))
                circuit.cp(theta, qubits[j], qubits[i])
            circuit.h(qubits[i])
    
    def create_variational_circuit(
        self,
        params: Dict[str, float]
    ) -> QuantumCircuit:
        """Create a variational quantum circuit."""
        circuit = self.create_quantum_circuit()
        
        # Layer 1: Single-qubit rotations
        for i in range(self.n_qubits):
            circuit.u3(
                params.get(f'theta_{i}', 0.0),
                params.get(f'phi_{i}', 0.0),
                params.get(f'lambda_{i}', 0.0),
                i
            )
        
        # Layer 2: Entanglement
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
        circuit.cx(self.n_qubits - 1, 0)  # Close the loop
        
        # Layer 3: Single-qubit rotations
        for i in range(self.n_qubits):
            circuit.u3(
                params.get(f'theta2_{i}', 0.0),
                params.get(f'phi2_{i}', 0.0),
                params.get(f'lambda2_{i}', 0.0),
                i
            )
        
        return circuit
    
    def quantum_phase_estimation(
        self,
        unitary_circuit: QuantumCircuit,
        precision_qubits: int = 3
    ) -> QuantumCircuit:
        """Perform quantum phase estimation."""
        # Create extended circuit with precision qubits
        precision_qr = QuantumRegister(precision_qubits, 'precision')
        precision_cr = ClassicalRegister(precision_qubits, 'measurement')
        circuit = QuantumCircuit(precision_qr, self.qr, precision_cr)
        
        # Initialize precision qubits in superposition
        for i in range(precision_qubits):
            circuit.h(precision_qr[i])
        
        # Apply controlled unitary operations
        for i in range(precision_qubits):
            power = 2 ** i
            for _ in range(power):
                circuit.compose(
                    unitary_circuit.control(),
                    qubits=[precision_qr[i]] + list(range(self.n_qubits)),
                    inplace=True
                )
        
        # Apply inverse QFT to precision qubits
        self.apply_inverse_quantum_fourier_transform(
            circuit,
            tuple(range(precision_qubits))
        )
        
        # Measure precision qubits
        for i in range(precision_qubits):
            circuit.measure(precision_qr[i], precision_cr[i])
        
        return circuit
    
    def measure_state_tomography(
        self,
        circuit: QuantumCircuit
    ) -> Dict[str, float]:
        """Perform quantum state tomography."""
        measurements = {}
        
        # X-basis measurements
        x_circuit = circuit.copy()
        for i in range(self.n_qubits):
            x_circuit.h(i)
        x_circuit.measure_all()
        x_result = execute(x_circuit, self.simulator).result()
        measurements['x_basis'] = x_result.get_counts()
        
        # Y-basis measurements
        y_circuit = circuit.copy()
        for i in range(self.n_qubits):
            y_circuit.sdg(i)
            y_circuit.h(i)
        y_circuit.measure_all()
        y_result = execute(y_circuit, self.simulator).result()
        measurements['y_basis'] = y_result.get_counts()
        
        # Z-basis measurements
        z_circuit = circuit.copy()
        z_circuit.measure_all()
        z_result = execute(z_circuit, self.simulator).result()
        measurements['z_basis'] = z_result.get_counts()
        
        return measurements
    
    def encode_classical_data(
        self,
        data: torch.Tensor,
        add_entanglement: bool = True
    ) -> QuantumCircuit:
        """Encode classical data into quantum state."""
        circuit = self.create_quantum_circuit()
        
        # Normalize data
        normalized_data = data / torch.norm(data)
        
        # Encode data into quantum state
        for i in range(min(len(normalized_data), self.n_qubits)):
            angle = float(normalized_data[i]) * np.pi
            circuit.ry(angle, i)
        
        if add_entanglement:
            # Create entanglement
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
        
        return circuit
    
    def quantum_feature_map(
        self,
        data: torch.Tensor,
        add_interference: bool = True
    ) -> torch.Tensor:
        """Map classical features to quantum feature space."""
        circuit = self.encode_classical_data(data)
        
        if add_interference:
            # Add quantum interference
            for i in range(self.n_qubits):
                circuit.h(i)
                circuit.phase(np.pi/4, i)
            
            # Apply QFT for enhanced feature mapping
            self.apply_quantum_fourier_transform(
                circuit,
                tuple(range(min(4, self.n_qubits)))
            )
        
        # Execute circuit and perform tomography
        measurements = self.measure_state_tomography(circuit)
        
        # Convert measurements to feature vector
        features = []
        for basis in ['x_basis', 'y_basis', 'z_basis']:
            counts = measurements[basis]
            total = sum(counts.values())
            for state, count in counts.items():
                features.append(count / total)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def quantum_enhanced_similarity(
        self,
        vec1: torch.Tensor,
        vec2: torch.Tensor
    ) -> float:
        """Calculate similarity using quantum interference."""
        # Create circuit for similarity calculation
        circuit = self.create_quantum_circuit()
        
        # Create Bell pair for enhanced measurement
        bell_circuit = self.create_bell_pair()
        circuit.compose(bell_circuit, inplace=True)
        
        # Create variational circuit for similarity computation
        params = {
            'theta_0': float(torch.norm(vec1)),
            'phi_0': float(torch.norm(vec2)),
            'lambda_0': float(torch.dot(vec1, vec2))
        }
        var_circuit = self.create_variational_circuit(params)
        circuit.compose(var_circuit, inplace=True)
        
        # Apply QFT for enhanced similarity detection
        self.apply_quantum_fourier_transform(
            circuit,
            tuple(range(min(4, self.n_qubits)))
        )
        
        # Execute circuit and perform tomography
        measurements = self.measure_state_tomography(circuit)
        
        # Calculate similarity from measurements
        z_counts = measurements['z_basis']
        total = sum(z_counts.values())
        similarity = z_counts.get('0' * self.n_qubits, 0) / total
        
        return float(similarity)
    
    def detect_quantum_resonance(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor
    ) -> Tuple[float, Dict[str, float]]:
        """Detect quantum resonance between states."""
        # Create resonance detection circuit
        circuit = self.create_quantum_circuit()
        
        # Create GHZ state for enhanced resonance detection
        ghz_circuit = self.create_ghz_state()
        circuit.compose(ghz_circuit, inplace=True)
        
        # Create variational circuit for resonance detection
        params = {
            'theta_0': float(torch.norm(state1)),
            'phi_0': float(torch.norm(state2)),
            'lambda_0': float(torch.dot(state1, state2))
        }
        var_circuit = self.create_variational_circuit(params)
        circuit.compose(var_circuit, inplace=True)
        
        # Apply QFT for enhanced resonance detection
        self.apply_quantum_fourier_transform(
            circuit,
            tuple(range(min(4, self.n_qubits)))
        )
        
        # Execute circuit and perform tomography
        measurements = self.measure_state_tomography(circuit)
        
        # Calculate resonance metrics from measurements
        z_counts = measurements['z_basis']
        total = sum(z_counts.values())
        resonance_score = z_counts.get('0' * self.n_qubits, 0) / total
        
        # Calculate additional metrics
        x_counts = measurements['x_basis']
        y_counts = measurements['y_basis']
        
        metrics = {
            'coherence': sum(
                abs(c1/total - c2/total)
                for c1, c2 in zip(z_counts.values(), x_counts.values())
            ),
            'entanglement': sum(
                abs(c1/total - c2/total)
                for c1, c2 in zip(z_counts.values(), y_counts.values())
            ),
            'phase_alignment': sum(
                c/total * i/len(z_counts)
                for i, c in enumerate(z_counts.values())
            ),
            'state_purity': sum(
                (c/total) ** 2 for c in z_counts.values()
            )
        }
        
        return resonance_score, metrics
    
    def quantum_phase_estimation_enhanced(
        self,
        unitary_circuit: QuantumCircuit,
        precision_qubits: int = 3,
        num_iterations: int = 1,
        controlled_rotations: bool = True
    ) -> QuantumCircuit:
        """Enhanced quantum phase estimation with iterative phase refinement.
        
        Args:
            unitary_circuit: Circuit implementing the unitary operation
            precision_qubits: Number of qubits for phase estimation
            num_iterations: Number of iterative refinement steps
            controlled_rotations: Whether to use controlled rotation gates
        """
        # Create extended circuit with precision qubits
        precision_qr = QuantumRegister(precision_qubits, 'precision')
        precision_cr = ClassicalRegister(precision_qubits, 'measurement')
        circuit = QuantumCircuit(precision_qr, self.qr, precision_cr)
        
        # Initialize precision qubits in superposition
        for i in range(precision_qubits):
            circuit.h(precision_qr[i])
        
        # Iterative phase estimation
        for iteration in range(num_iterations):
            # Apply controlled unitary operations with increasing powers
            for i in range(precision_qubits):
                power = 2 ** (i + iteration * precision_qubits)
                
                if controlled_rotations:
                    # Add controlled phase rotations for improved precision
                    circuit.cp(np.pi / (2 ** i), precision_qr[i], self.qr[0])
                
                # Apply controlled unitary
                for _ in range(power):
                    circuit.compose(
                        unitary_circuit.control(),
                        qubits=[precision_qr[i]] + list(range(self.n_qubits)),
                        inplace=True
                    )
            
            # Apply inverse QFT to precision qubits
            self.apply_inverse_quantum_fourier_transform(
                circuit,
                tuple(range(precision_qubits))
            )
            
            if iteration < num_iterations - 1:
                # Reset precision qubits for next iteration
                for i in range(precision_qubits):
                    circuit.reset(precision_qr[i])
                    circuit.h(precision_qr[i])
        
        # Final measurements
        for i in range(precision_qubits):
            circuit.measure(precision_qr[i], precision_cr[i])
        
        return circuit
    
    def create_variational_circuit_enhanced(
        self,
        params: Dict[str, float],
        entanglement_pattern: str = 'full',
        num_layers: int = 2,
        add_zz_entanglement: bool = False
    ) -> QuantumCircuit:
        """Create enhanced variational quantum circuit.
        
        Args:
            params: Dictionary of variational parameters
            entanglement_pattern: Type of qubit connectivity ('full', 'linear', 'circular')
            num_layers: Number of variational layers
            add_zz_entanglement: Whether to add ZZ entangling gates
        """
        circuit = QuantumCircuit(self.n_qubits)
        
        # Define rotation gates for each layer
        rotations = ['rx', 'ry', 'rz']
        
        for layer in range(num_layers):
            # Single qubit rotations
            for i in range(self.n_qubits):
                for rot in rotations:
                    param_name = f'{rot[1]}_{layer}_{i}'
                    if param_name in params:
                        angle = params[param_name]
                    else:
                        angle = params.get(f'theta_{i}', 0.0)
                    
                    if rot == 'rx':
                        circuit.rx(angle, i)
                    elif rot == 'ry':
                        circuit.ry(angle, i)
                    else:  # rz
                        circuit.rz(angle, i)
            
            # Entangling gates based on pattern
            if entanglement_pattern == 'full':
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        circuit.cx(i, j)
                        if add_zz_entanglement:
                            # Add ZZ interaction
                            circuit.rz(params.get(f'zz_{layer}_{i}_{j}', np.pi/4), j)
                            circuit.cx(i, j)
            
            elif entanglement_pattern == 'linear':
                for i in range(self.n_qubits - 1):
                    circuit.cx(i, i + 1)
                    if add_zz_entanglement:
                        circuit.rz(params.get(f'zz_{layer}_{i}', np.pi/4), i + 1)
                        circuit.cx(i, i + 1)
            
            elif entanglement_pattern == 'circular':
                for i in range(self.n_qubits):
                    next_qubit = (i + 1) % self.n_qubits
                    circuit.cx(i, next_qubit)
                    if add_zz_entanglement:
                        circuit.rz(params.get(f'zz_{layer}_{i}', np.pi/4), next_qubit)
                        circuit.cx(i, next_qubit)
            
            # Optional barrier between layers
            circuit.barrier()
        
        return circuit
    
    def measure_state_tomography_enhanced(
        self,
        circuit: QuantumCircuit,
        add_error_mitigation: bool = True,
        shots: int = 1024
    ) -> Dict[str, Dict[str, float]]:
        """Perform enhanced state tomography with error mitigation.
        
        Args:
            circuit: Quantum circuit to measure
            add_error_mitigation: Whether to apply error mitigation
            shots: Number of measurement shots
        """
        # Create measurement circuits for each basis
        measurements = {}
        bases = ['z', 'x', 'y']
        
        for basis in bases:
            # Create copy of circuit for measurement
            meas_circuit = circuit.copy()
            
            # Add basis transformation
            if basis == 'x':
                for qubit in range(self.n_qubits):
                    meas_circuit.h(qubit)
            elif basis == 'y':
                for qubit in range(self.n_qubits):
                    meas_circuit.sdg(qubit)
                    meas_circuit.h(qubit)
            
            # Add measurements
            cr = ClassicalRegister(self.n_qubits)
            meas_circuit.add_register(cr)
            meas_circuit.measure_all()
            
            # Execute circuit
            counts = self.execute_circuit(meas_circuit, shots=shots)
            
            # Store raw measurements
            measurements[f'{basis}_basis'] = {
                state: count / shots
                for state, count in counts.items()
            }
            
            if add_error_mitigation:
                # Apply error mitigation
                mitigated_counts = self._apply_error_mitigation(counts)
                measurements[f'{basis}_basis_mitigated'] = {
                    state: count / shots
                    for state, count in mitigated_counts.items()
                }
        
        return measurements
    
    def _apply_error_mitigation(
        self,
        counts: Dict[str, int]
    ) -> Dict[str, int]:
        """Apply error mitigation to measurement counts.
        
        Args:
            counts: Raw measurement counts
        """
        # Simple error mitigation: threshold small probabilities
        threshold = 0.05
        total_counts = sum(counts.values())
        
        mitigated_counts = {}
        for state, count in counts.items():
            prob = count / total_counts
            if prob > threshold:
                mitigated_counts[state] = count
            
        # Renormalize
        total_mitigated = sum(mitigated_counts.values())
        if total_mitigated > 0:
            scale = total_counts / total_mitigated
            mitigated_counts = {
                state: int(count * scale)
                for state, count in mitigated_counts.items()
            }
        
        return mitigated_counts
    
    def quantum_feature_projection(
        self,
        features: torch.Tensor,
        projection_type: str = 'hybrid',
        add_nonlinearity: bool = True
    ) -> torch.Tensor:
        """Project classical features into quantum state space.
        
        Args:
            features: Input feature tensor
            projection_type: Type of projection ('quantum', 'hybrid', 'variational')
            add_nonlinearity: Whether to add quantum nonlinearity
        """
        circuit = self.create_quantum_circuit()
        
        if projection_type == 'quantum':
            # Encode features directly into quantum states
            for i in range(min(len(features), self.n_qubits)):
                circuit.ry(float(features[i]) * np.pi, i)
                if i > 0:
                    circuit.cx(i-1, i)
                
            if add_nonlinearity:
                # Add quantum nonlinearity via controlled operations
                for i in range(self.n_qubits - 1):
                    circuit.crz(float(features[i % len(features)]), i, i+1)
                
        elif projection_type == 'hybrid':
            # Create quantum-classical hybrid encoding
            # First layer: quantum encoding
            for i in range(min(len(features), self.n_qubits)):
                circuit.rx(float(features[i]) * np.pi / 2, i)
                circuit.rz(float(features[i]) * np.pi, i)
            
            # Entanglement layer
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i+1)
            circuit.cx(self.n_qubits - 1, 0)
            
            # Second layer: nonlinear transformation
            if add_nonlinearity:
                for i in range(self.n_qubits):
                    circuit.ry(float(torch.tanh(features[i % len(features)])) * np.pi, i)
                
        elif projection_type == 'variational':
            # Create variational quantum circuit for feature projection
            params = {f'theta_{i}': float(f) for i, f in enumerate(features)}
            var_circuit = self.create_variational_circuit_enhanced(
                params,
                entanglement_pattern='circular',
                num_layers=2
            )
            circuit.compose(var_circuit, inplace=True)
            
            if add_nonlinearity:
                # Add parameterized controlled rotations
                for i in range(self.n_qubits - 1):
                    param_val = float(torch.sigmoid(features[i % len(features)]))
                    circuit.crx(param_val * np.pi, i, i+1)
        
        # Apply QFT for enhanced feature interaction
        if add_nonlinearity:
            self.apply_quantum_fourier_transform(
                circuit,
                tuple(range(min(3, self.n_qubits)))
        
        # Get quantum state
        state_vector = self.get_statevector(circuit)
        
        # Convert to tensor
        return torch.tensor(state_vector.data, dtype=torch.complex64)
    
    def quantum_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        num_heads: int = 1,
        add_entanglement: bool = True
    ) -> torch.Tensor:
        """Calculate quantum-enhanced attention weights.
        
        Args:
            query: Query tensor
            key: Key tensor
            num_heads: Number of attention heads
            add_entanglement: Whether to add entanglement between heads
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        head_dim = query.size(-1) // num_heads
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, num_heads, head_dim)
        key = key.view(batch_size, seq_len, num_heads, head_dim)
        
        attention_weights = torch.zeros(
            (batch_size, num_heads, seq_len, seq_len),
            device=query.device
        )
        
        for b in range(batch_size):
            for h in range(num_heads):
                # Create quantum circuits for this head
                head_circuit = self.create_quantum_circuit()
                
                # Create entangled state if requested
                if add_entanglement:
                    bell_circuit = self.create_bell_pair()
                    head_circuit.compose(bell_circuit, inplace=True)
                
                for i in range(seq_len):
                    for j in range(seq_len):
                        # Create variational circuit for attention computation
                        q_norm = float(torch.norm(query[b, i, h]))
                        k_norm = float(torch.norm(key[b, j, h]))
                        qk_dot = float(torch.dot(query[b, i, h], key[b, j, h]))
                        
                        params = {
                            'theta_0': q_norm,
                            'phi_0': k_norm,
                            'lambda_0': qk_dot / (q_norm * k_norm + 1e-8)
                        }
                        
                        var_circuit = self.create_variational_circuit_enhanced(
                            params,
                            entanglement_pattern='full',
                            num_layers=2
                        )
                        
                        # Combine circuits
                        circuit = head_circuit.copy()
                        circuit.compose(var_circuit, inplace=True)
                        
                        # Apply QFT for enhanced attention
                        self.apply_quantum_fourier_transform(
                            circuit,
                            tuple(range(min(4, self.n_qubits)))
                        )
                        
                        # Measure quantum state
                        measurements = self.measure_state_tomography_enhanced(
                            circuit,
                            add_error_mitigation=True
                        )
                        
                        # Calculate attention weight from measurements
                        z_basis = measurements['z_basis_mitigated']
                        attention_weights[b, h, i, j] = z_basis.get(
                            '0' * self.n_qubits,
                            0.0
                        )
        
        # Apply softmax normalization
        attention_weights = torch.softmax(
            attention_weights / np.sqrt(head_dim),
            dim=-1
        )
        
        return attention_weights

    def quantum_enhanced_similarity_v2(
        self,
        vec1: torch.Tensor,
        vec2: torch.Tensor,
        similarity_type: str = 'quantum',
        add_interference: bool = True
    ) -> float:
        """Calculate enhanced quantum similarity between vectors.
        
        Args:
            vec1: First input vector
            vec2: Second input vector
            similarity_type: Type of similarity calculation ('quantum', 'hybrid', 'interference')
            add_interference: Whether to add quantum interference effects
        """
        # Create base circuit
        circuit = self.create_quantum_circuit()
        
        # Create Bell pair for enhanced measurement
        bell_circuit = self.create_bell_pair()
        circuit.compose(bell_circuit, inplace=True)
        
        if similarity_type == 'quantum':
            # Create variational circuit for similarity computation
            params = {
                'theta_0': float(torch.norm(vec1)),
                'phi_0': float(torch.norm(vec2)),
                'lambda_0': float(torch.dot(vec1, vec2))
            }
            var_circuit = self.create_variational_circuit_enhanced(
                params,
                entanglement_pattern='full',
                num_layers=2,
                add_zz_entanglement=True
            )
            circuit.compose(var_circuit, inplace=True)
            
        elif similarity_type == 'hybrid':
            # Encode vectors into quantum states
            for i in range(min(len(vec1), self.n_qubits // 2)):
                circuit.ry(float(vec1[i]) * np.pi, i)
            for i in range(min(len(vec2), self.n_qubits // 2)):
                circuit.ry(float(vec2[i]) * np.pi, i + self.n_qubits // 2)
            
            # Add entanglement between states
            for i in range(self.n_qubits // 2):
                circuit.cx(i, i + self.n_qubits // 2)
            
        elif similarity_type == 'interference':
            # Create superposition of both vectors
            circuit.h(0)  # Control qubit
            
            # Controlled encoding of vectors
            for i in range(min(len(vec1), self.n_qubits - 1)):
                circuit.cry(float(vec1[i]) * np.pi, 0, i + 1)
            
            circuit.x(0)  # Flip control
            for i in range(min(len(vec2), self.n_qubits - 1)):
                circuit.cry(float(vec2[i]) * np.pi, 0, i + 1)
            circuit.x(0)  # Flip back
        
        if add_interference:
            # Apply QFT for interference effects
            self.apply_quantum_fourier_transform(
                circuit,
                tuple(range(min(4, self.n_qubits)))
        
        # Measure quantum state with error mitigation
        measurements = self.measure_state_tomography_enhanced(
            circuit,
            add_error_mitigation=True
        )
        
        # Calculate similarity from measurements
        z_basis = measurements['z_basis_mitigated']
        similarity = z_basis.get('0' * self.n_qubits, 0.0)
        
        return float(similarity)

    def execute_circuit(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        optimization_level: int = 1
    ) -> Dict[str, int]:
        """Execute quantum circuit and return measurement counts.
        
        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots
            optimization_level: Circuit optimization level (0-3)
        """
        from qiskit import execute, Aer
        
        # Get simulator backend
        backend = Aer.get_backend('qasm_simulator')
        
        # Execute circuit
        job = execute(
            circuit,
            backend=backend,
            shots=shots,
            optimization_level=optimization_level
        )
        
        # Get and return counts
        return job.result().get_counts()

    def get_statevector(
        self,
        circuit: QuantumCircuit,
        decimals: int = 10
    ) -> Statevector:
        """Get the statevector representation of a quantum circuit.
        
        Args:
            circuit: Quantum circuit
            decimals: Number of decimal places to round to
        """
        from qiskit import Aer, execute
        
        # Get statevector simulator
        backend = Aer.get_backend('statevector_simulator')
        
        # Execute circuit
        job = execute(circuit, backend)
        result = job.result()
        
        # Get statevector
        statevector = result.get_statevector()
        
        # Round small values to zero
        threshold = 10**(-decimals)
        statevector = np.array([
            0 if abs(x) < threshold else x
            for x in statevector
        ])
        
        return Statevector(statevector)


class QuantumTokenizer:
    """Quantum-enhanced tokenizer with similarity calculation capabilities."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_qubits: int = 4,
        use_quantum_encoding: bool = True
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        self.use_quantum_encoding = use_quantum_encoding
        
        # Initialize quantum processor
        self.quantum_processor = QuantumProcessor(n_qubits)
        
        # Initialize classical embeddings
        self.embeddings = nn.Parameter(
            torch.randn(vocab_size, embedding_dim)
        )
        
        # Quantum projection layer
        if use_quantum_encoding:
            self.quantum_projection = nn.Sequential(
                nn.Linear(embedding_dim, 2 ** n_qubits),
                nn.LayerNorm(2 ** n_qubits),
            nn.GELU()
        )
    
    def encode(
        self,
        tokens: torch.Tensor,
        return_quantum_states: bool = False
    ) -> torch.Tensor:
        """Encode tokens into embeddings with optional quantum enhancement."""
        # Get classical embeddings
        embeddings = F.embedding(tokens, self.embeddings)
        
        if self.use_quantum_encoding:
            # Project to quantum dimension
            quantum_features = self.quantum_projection(embeddings)
            
            # Apply quantum encoding
            batch_size, seq_len, _ = embeddings.shape
            quantum_states = []
            
            for b in range(batch_size):
                for s in range(seq_len):
                    # Project features to quantum state
                    quantum_state = self.quantum_processor.quantum_feature_projection(
                        quantum_features[b, s],
                        projection_type='hybrid',
                        add_nonlinearity=True
                    )
                    quantum_states.append(quantum_state)
            
            quantum_states = torch.stack(quantum_states)
            quantum_states = quantum_states.view(batch_size, seq_len, -1)
            
            if return_quantum_states:
                return quantum_states
            
            # Project back to embedding dimension
            return self.quantum_projection[0].weight.t() @ quantum_states.unsqueeze(-1)
        
        return embeddings
    
    def calculate_similarity(
        self,
        tokens1: torch.Tensor,
        tokens2: torch.Tensor,
        similarity_type: str = 'quantum'
    ) -> torch.Tensor:
        """Calculate similarity between token sequences."""
        # Get quantum states
        states1 = self.encode(tokens1, return_quantum_states=True)
        states2 = self.encode(tokens2, return_quantum_states=True)
        
        # Calculate similarities
        batch_size = states1.size(0)
        seq_len1 = states1.size(1)
        seq_len2 = states2.size(1)
        
        similarities = torch.zeros(
            (batch_size, seq_len1, seq_len2),
            device=states1.device
        )
        
        for b in range(batch_size):
            for i in range(seq_len1):
                for j in range(seq_len2):
                    similarities[b, i, j] = self.quantum_processor.quantum_enhanced_similarity_v2(
                        states1[b, i],
                        states2[b, j],
                        similarity_type=similarity_type
                    )
        
        return similarities
    
    def batch_encode(
        self,
        batch_tokens: List[torch.Tensor],
        padding: bool = True,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of token sequences with padding."""
        if max_length is None:
            max_length = max(len(tokens) for tokens in batch_tokens)
        
        batch_size = len(batch_tokens)
        padded_tokens = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
            device=batch_tokens[0].device
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.float,
            device=batch_tokens[0].device
        )
        
        for i, tokens in enumerate(batch_tokens):
            length = min(len(tokens), max_length)
            padded_tokens[i, :length] = tokens[:length]
            attention_mask[i, :length] = 1.0
        
        # Encode padded tokens
        embeddings = self.encode(padded_tokens)
        
        return {
            'input_ids': padded_tokens,
            'attention_mask': attention_mask,
            'embeddings': embeddings
        } 