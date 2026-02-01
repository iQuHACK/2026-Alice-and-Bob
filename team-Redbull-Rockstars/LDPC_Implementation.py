import stim
import sinter
import numpy as np
from ldpc import bposd_decoder
import scipy.sparse as sp
from ldpc.mod2 import nullspace
from ldpc.mod2 import rank
import matplotlib.pyplot as plt

# d=4 cat-repetition code benchmark from section 5.1
D4_P = [0.001, 0.001, 0.002, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.011,
        0.014, 0.018, 0.024, 0.031, 0.041, 0.053, 0.069, 0.090, 0.118, 0.153, 0.200]
D4_LER = [0.000010, 0.000000, 0.000000, 0.000040, 0.000030, 0.000020, 0.000060, 0.000060, 0.000190, 0.000300,
          0.000580, 0.000970, 0.001510, 0.002850, 0.004830, 0.007810, 0.013590, 0.023400, 0.038450, 0.064570, 0.105680]

def ldpc_x_memory(H_matrix, logical_support_indices, p):
    num_checks, num_data = H_matrix.shape
    data_qubits = list(range(num_data))
    ancilla_qubits = list(range(num_data, num_data + num_checks))
    
    c = stim.Circuit()

    c.append("R", data_qubits + ancilla_qubits)

    c.append("X_ERROR", data_qubits, p)

    for check_idx in range(num_checks):
        ancilla = ancilla_qubits[check_idx]
        targets = np.where(H_matrix[check_idx] == 1)[0]
        
        c.append("H", ancilla)
        for q in targets:
            c.append("CZ", [ancilla, q])
        c.append("H", ancilla)
        c.append("M", ancilla)
        
        c.append("DETECTOR", [stim.target_rec(-1)], [check_idx])

    c.append("M", data_qubits)
    
    
    rec_targets = []
    for q_idx in logical_support_indices:

        relative_index = q_idx - num_data 
        rec_targets.append(stim.target_rec(relative_index))
        
    c.append("OBSERVABLE_INCLUDE", rec_targets, 0)
    
    return c

def shapes_to_parity_matrix(width, height, row_shapes):
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
    bpd = bposd_decoder(
        H_matrix,
        error_rate = float(error_rate),
        max_iter=max_iter,
        bp_method='ms',          # Min-Sum is commonly used for stability
        ms_scaling_factor=0.625, # Standard scaling factor for Min-Sum
        osd_method='osd_cs',     # OSD with Combination Sweep
        osd_order=osd_order      # Controls the search depth in OSD
    )


    predicted_error = bpd.decode(syndrome)

    return predicted_error


class BPOSDCompiledDecoder(sinter.CompiledDecoder):
    """Compiled decoder for a specific H_matrix configuration."""

    def __init__(self, H_matrix, logical_support, error_rate, num_detectors, num_observables, max_iter=30, osd_order=10):
        self.H_matrix = H_matrix
        self.logical_support = logical_support
        self.error_rate = error_rate
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.max_iter = max_iter
        self.osd_order = osd_order

        # Pre-create the bposd_decoder instance for reuse
        self.bpd = bposd_decoder(
            H_matrix,
            error_rate=float(error_rate),
            max_iter=max_iter,
            bp_method='ms',
            ms_scaling_factor=0.625,
            osd_method='osd_cs',
            osd_order=osd_order
        )

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
        num_shots = bit_packed_detection_event_data.shape[0]
        num_det_bytes = (self.num_detectors + 7) // 8

        # Output: one bit per observable, bit-packed
        num_obs_bytes = (self.num_observables + 7) // 8
        predictions = np.zeros((num_shots, num_obs_bytes), dtype=np.uint8)

        for shot_idx in range(num_shots):
            # Unpack the detection events for this shot
            packed_row = bit_packed_detection_event_data[shot_idx, :num_det_bytes]
            syndrome_bits = np.unpackbits(packed_row, bitorder='little')[:self.num_detectors]
            syndrome = syndrome_bits.astype(int)

            # Decode using BPOSD
            correction = self.bpd.decode(syndrome)

            # Compute logical flip
            predicted_flip = np.sum(correction[self.logical_support]) % 2

            # Pack the prediction (single observable)
            if predicted_flip:
                predictions[shot_idx, 0] = 1

        return predictions


class CustomBPOSDDecoder(sinter.Decoder):
    """Sinter decoder wrapper for ldpc bposd_decoder."""

    def __init__(self, H_matrix, logical_support, error_rate, max_iter=30, osd_order=10):
        self.H_matrix = H_matrix
        self.logical_support = logical_support
        self.error_rate = error_rate
        self.max_iter = max_iter
        self.osd_order = osd_order

    def compile_decoder_for_dem(self, *, dem):
        num_detectors = dem.num_detectors
        num_observables = dem.num_observables
        return BPOSDCompiledDecoder(
            H_matrix=self.H_matrix,
            logical_support=self.logical_support,
            error_rate=self.error_rate,
            num_detectors=num_detectors,
            num_observables=num_observables,
            max_iter=self.max_iter,
            osd_order=self.osd_order
        )


def generate_tasks(H_matrix, logical_ops, physical_errors):
    """Generate sinter.Task for each physical error rate."""
    for p in physical_errors:
        circuit = ldpc_x_memory(H_matrix, logical_ops, float(p))
        yield sinter.Task(
            circuit=circuit,
            decoder=f'bposd_p{p:.6f}',
            json_metadata={'p': float(p)},
        )


def create_custom_decoders(H_matrix, logical_support, physical_errors):
    """Create dict mapping decoder names to instances."""
    return {
        f'bposd_p{p:.6f}': CustomBPOSDDecoder(H_matrix, logical_support, float(p))
        for p in physical_errors
    }


def extract_results(results):
    """Extract (physical_errors, logical_errors) from sinter.TaskStats list."""
    sorted_results = sorted(results, key=lambda r: r.json_metadata['p'])
    physical_errors = [r.json_metadata['p'] for r in sorted_results]
    logical_errors = [r.errors / r.shots if r.shots > 0 else 0.0 for r in sorted_results]
    return physical_errors, logical_errors


def calculate_logical_error_rate(circuit, H_matrix, logical_support, num_shots=1000, p_error=0.01):
    """Legacy function for single-threaded decoding."""
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


if __name__ == "__main__":
    import multiprocessing

    L = 50
    H = 8

    shapes = [
        [(0, 0),
         (2, -1),
         (2, 0),
         (2, 1)],
        [(0, 0),
         (2, -1),
         (2, 0),
         (2, 1)],
        [(0, 0),
         (2, -1),
         (1, 1),
         (2, 1)],
        [(0, 0),
         (2, -1),
         (1, 1),
         (2, 1)],
        [(0, 0),
         (2, -1),
         (1, -1),
         (1, 1)],
        [(0, 0),
         (2, -1),
         (1, 1),
         (2, 1)],
    ]

    H_matrix = shapes_to_parity_matrix(width=L, height=H, row_shapes=shapes)

    print(f"Lattice: {H}x{L} = {H*L} physical qubits")
    print(f"Matrix Shape: {H_matrix.shape}")
    print(f"Number of Stabilizers: {H_matrix.shape[0]}")
    Hrank = rank(H_matrix)
    k = 2 * L
    print(f"Logical Qubits (k): {k}")

    logical_ops = [r * L for r in range(H)]
    physical_errors = np.logspace(-3, np.log10(0.15), 21).astype(float)

    NUM_WORKERS = max(4, multiprocessing.cpu_count())
    NUM_SHOTS = 1_000_000

    print(f"Using {NUM_WORKERS} workers for {NUM_SHOTS} shots per error rate")

    custom_decoders = create_custom_decoders(H_matrix, logical_ops, physical_errors)
    tasks = list(generate_tasks(H_matrix, logical_ops, physical_errors))

    results = sinter.collect(
        num_workers=NUM_WORKERS,
        tasks=tasks,
        custom_decoders=custom_decoders,
        max_shots=NUM_SHOTS,
        print_progress=True,
    )

    p_values, logical_errors = extract_results(results)

    print(logical_errors)
    plt.figure(figsize=(8, 6))
    plt.loglog(p_values, logical_errors, '-o', label=f'CA Code (H={H}, L={L})')
    plt.loglog(D4_P, D4_LER, '-s', label='Cat-Rep Code (d=4)')
    plt.loglog(p_values, p_values, '--', color='gray', label='Breakeven (y=x)')
    plt.xlabel("Physical Error Rate (p)")
    plt.ylabel("Logical Error Rate (LER)")
    plt.title("Bit-Flip Error Correction Performance")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.xlim(1e-3, 1.5e-1)
    plt.savefig("test.png")