import stim
import numpy as np
from ldpc import bposd_decoder
import scipy.sparse as sp
from ldpc.mod2 import nullspace
from ldpc.mod2 import rank


def ldpc_x_memory(H_matrix, logical_support, p):

    num_checks, num_data = H_matrix.shape
    data_qubits = list(range(num_data))
    ancilla_qubits = list(range(num_data, num_data + num_checks))
    
    c = stim.Circuit()

    c.append("R", data_qubits + ancilla_qubits)

    c.append("X_ERROR", data_qubits, p)

    
    for check_idx in range(num_checks):
        ancilla = ancilla_qubits[check_idx]
        
        # Identify which data qubits are involved in this check
        # (Where the matrix row is 1)
        targets = np.where(H_matrix[check_idx] == 1)[0]
        
        c.append("H", ancilla)
        
        # CZ between Ancilla and Data propagates Z information to the X-basis of the Ancilla
        for q in targets:
            c.append("CZ", [ancilla, q])
            
        c.append("H", ancilla)
        c.append("M", ancilla)
        
        # Compares this measurement to the expected value (deterministic 0 in absence of noise)
        c.append("DETECTOR", [stim.target_rec(-1)], [check_idx])

    c.append("M", data_qubits)
    
    obs_targets = []
    for q_idx in logical_support:
        # Calculate relative offset from the end of the measurement record
        rel_index = -(num_data - q_idx)
        obs_targets.append(stim.target_rec(rel_index))
        
    c.append("OBSERVABLE_INCLUDE", obs_targets, 0)

    return c

def shapes_to_parity_matrix(width, height, row_shapes):
    """
    Converts 2D stabilizer shapes into a Parity Check Matrix (H).
    
    Args:
        width (int): Lattice width (L). Periodic boundaries.
        height (int): Lattice height (H). Open boundaries.
        row_shapes (list): A list of shapes. Index i corresponds to the shape 
                           used for lattice row i. If len(row_shapes) < height,
                           it cycles through them (Quasi-Cyclic).
                           Shape format: List of (dr, dc) tuples.
                           
    Returns:
        np.array: The binary parity check matrix H.
                  Rows = Stabilizers, Cols = Physical Qubits.
    """
    num_qubits = width * height
    stabilizer_rows = []

    # Iterate through every possible "root" position (r, c) on the lattice
    for r in range(height):
        current_shape = row_shapes[r % len(row_shapes)]
        
        # Find how "tall" the shape is.
        max_dr = max(dr for dr, dc in current_shape)
        
        # If the shape hangs off the bottom of the lattice, we don't place it.
        # (This creates the "logical degrees of freedom" at the boundaries)
        if r + max_dr >= height:
            continue
            
        for c in range(width):
            # Create a new row for the H-matrix (initially all zeros)
            h_row = np.zeros(num_qubits, dtype=int)
            
            # Apply the shape offsets to find involved qubits
            for dr, dc in current_shape:
                target_r = r + dr
                # Wrap column index (Periodic Boundary Condition)
                target_c = (c + dc) % width 
                
                # Map 2D (r, c) -> 1D index
                qubit_idx = target_r * width + target_c
                
                h_row[qubit_idx] = 1
            
            stabilizer_rows.append(h_row)

    return np.array(stabilizer_rows)

def decode_with_bposd(syndrome, H_matrix, error_rate=0.01, max_iter=30, osd_order=10):
    """
    Decodes the syndrome using the BP+OSD algorithm.
    
    Args:
        syndrome (np.array): The binary syndrome vector (measurements from ancilla).
        H_matrix (np.array): The binary parity check matrix.
        error_rate (float): The estimated physical error rate (p) for channel log-likelihood ratios.
        max_iter (int): Maximum iterations for Belief Propagation.
        osd_order (int): The order for Ordered Statistics Decoding (higher = more accurate but slower).
        
    Returns:
        np.array: The predicted error vector (correction) for the data qubits.
    """
    bpd = bposd_decoder(
        H_matrix,
        error_rate=error_rate,
        max_iter=max_iter,
        bp_method='ms',          # Min-Sum is commonly used for stability
        ms_scaling_factor=0.625, # Standard scaling factor for Min-Sum
        osd_method='osd_cs',     # OSD with Combination Sweep
        osd_order=osd_order      # Controls the search depth in OSD
    )
    

    predicted_error = bpd.decode(syndrome)
    
    return predicted_error


import numpy as np
import stim

def calculate_logical_error_rate(circuit, H_matrix, logical_support, num_shots=1000, p_error=0.01):
    sampler = circuit.compile_detector_sampler()
    syndromes_batch, actual_observables_batch = sampler.sample(
        shots=num_shots, 
        separate_observables=True
    )
    
    num_logical_errors = 0
    
    print(f"Running decoding for {num_shots} shots...")

    for i in range(num_shots):
        syndrome = syndromes_batch[i].astype(int)
        
        actual_flip = actual_observables_batch[i][0]
        
        correction = decode_with_bposd(syndrome, H_matrix, error_rate=p_error)
        

        predicted_flip = np.sum(correction[logical_support]) % 2
        
        if predicted_flip != actual_flip:
            num_logical_errors += 1

    ler = num_logical_errors / num_shots
    return ler


L = 50
H = 8   

shapes = [
    [ (0, 0),
    (2, -1),
    (2, 0),   
    (2, 1)], 
    [ (0, 0),
    (2, -1),
    (2, 0),   
    (2, 1)], 
    [ (0, 0),
    (2, -1),
    (1, 1),   
    (2, 1)],
    [ (0, 0),
    (2, -1),
    (1, 1),   
    (2, 1)], 
    [ (0, 0),
    (2, -1),
    (1, -1),   
    (1, 1)],
    [ (0, 0),
    (2, -1),
    (1, 1),   
    (2, 1)],
    ]

H_matrix = shapes_to_parity_matrix(width=L, height=H, row_shapes=shapes)

print(f"Lattice: {H}x{L} = {H*L} physical qubits")
print(f"Matrix Shape: {H_matrix.shape}")
print(f"Number of Stabilizers: {H_matrix.shape[0]}")
Hrank = rank(H_matrix)
k = 2*L
print(f"Logical Qubits (k): {k}")

p = 0.1

logical_ops = [r * L for r in range(H)]
circuit = ldpc_x_memory(H_matrix, logical_ops, p)
#print(repr(circuit))
sampler = circuit.compile_detector_sampler()
syndrome_batch = sampler.sample(shots=1)
single_syndrome = syndrome_batch[0].astype(int)

correction = decode_with_bposd(single_syndrome, H_matrix, error_rate=p)

print(f"Syndrome: {single_syndrome}")
print(f"Predicted Correction: {correction}")

logical_error_rate = calculate_logical_error_rate(
    circuit=circuit,
    H_matrix=H_matrix,
    logical_support=logical_ops,
    num_shots=1000,
    p_error=p
)


print(f"Physical Error Rate (p):{p}")
print(f"Logical Error Rate (LER): {logical_error_rate:.4f}")