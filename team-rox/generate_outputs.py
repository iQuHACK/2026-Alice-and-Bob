
import stim
import pymatching
import sinter
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Helper to safe print ---
def safe_print(msg):
    try:
        print(msg)
    except Exception:
        pass

# --- Task 2.1 ---
print("\n=== Task 2.1: Error Detection Output ===")
try:
    circuit = stim.Circuit()
    circuit.append("R", [0, 1, 2])
    circuit.append("H", [0])
    circuit.append("CX", [0, 1])
    circuit.append("X_ERROR", [0, 1], 0.1)
    circuit.append("CX", [0, 2])
    circuit.append("CX", [1, 2])
    circuit.append("M", [2])
    circuit.append("DETECTOR", [stim.target_rec(-1)])
    circuit.append("M", [0, 1])
    # Note: Observable index 0
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-2)], 0)

    print(f"Num detectors: {circuit.num_detectors}")
    print(f"Num observables: {circuit.num_observables}")

    sampler = circuit.compile_detector_sampler()
    stats = sampler.sample(shots=100000)
    print(f"Stats shape: {stats.shape}")

    if stats.shape[1] > 1:
        valid = ~stats[:, 0]
        simulated_logical_error_rate = np.mean(stats[valid, 1]) if np.any(valid) else 0.0
        print(f"Simulated Logical Error Rate from {np.sum(valid)} valid shots: {simulated_logical_error_rate}")
    elif circuit.num_observables > 0:
        # If stats only returns detectors, maybe observables are implicit?
        # But compile_detector_sampler usually returns both.
        # We will skip valid calc if not present.
        print("Warning: Observable column missing in stats.")
    
    p = 0.1
    print(f"Theoretical Success Probability: {(1-p)**2 + p**2}")

except Exception as e:
    print(f"Task 2.1 failed: {e}")

print("========================================")

# --- Task 3 ---
print("\n=== Task 3: Repetition Code Output ===")
try:
    def decode_repetition_code(meas):
        data_bits = [int(x) for x in meas[0]]
        stab_bits = [int(x) for x in meas[1]]
        n = len(data_bits)
        correction = [0] * n
        curr = 0
        for i, s in enumerate(stab_bits):
            curr = (curr + s) % 2
            correction[i+1] = curr
        corrected = [(d+c)%2 for d, c in zip(data_bits, correction)]
        return 1 if sum(corrected) > n/2 else 0

    def repetition_code_circuit(n: int, p: float = 0.1) -> stim.Circuit:
        c = stim.Circuit()
        c.append("R", range(2*n-1))
        data_qubits = [2*i for i in range(n)]
        c.append("X_ERROR", data_qubits, p)
        for i in range(n-1):
            c.append("CX", [2*i, 2*i+1])
            c.append("CX", [2*i+2, 2*i+1])
            c.append("M", [2*i+1])
        c.append("M", data_qubits)
        return c

    def simulate_circuit(circuit, n, num_shots=10000):
        sampler = circuit.compile_sampler()
        samples = sampler.sample(shots=num_shots)
        results = {}
        for s in samples:
            stabs = "".join(str(int(b)) for b in s[:n-1])
            data = "".join(str(int(b)) for b in s[n-1:])
            key = (data, stabs)
            results[key] = results.get(key, 0) + 1
        return results

    def logical_error_rate(results, logical_prepared=0):
        errs = 0
        total = 0
        for (data, stabs), count in results.items():
            if decode_repetition_code((data, stabs)) != logical_prepared:
                errs += count
            total += count
        return errs / total if total > 0 else 0

    error_probabilities = np.logspace(-2, np.log10(0.5), 5) 
    distances = [3, 5] # Reduced
    print("Simulating Repetition Code Threshold (Manual)...")
    plt.figure(figsize=(10, 6))
    for d in distances:
        logical_errors = []
        print(f"  d={d}: ", end="")
        for p in error_probabilities:
            circuit = repetition_code_circuit(d, p)
            results = simulate_circuit(circuit, d, num_shots=2000) # Small shots
            p_L = logical_error_rate(results)
            logical_errors.append(p_L)
            print(f"p={p:.3f}->{p_L:.3f} ", end="")
        print()
        plt.loglog(error_probabilities, logical_errors, 'o-', label=f'd={d}')

    plt.xlabel("Physical Error Probability p")
    plt.ylabel("Logical Error Probability p_L")
    plt.title("Repetition Code Threshold Simulation")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.savefig("repetition_code_threshold.png")
    print("Saved repetition_code_threshold.png")

except Exception as e:
    print(f"Task 3 failed: {e}")
print("==================================")

# --- Task 4 ---
print("\n=== Task 4: Sinter Benchmarks Output ===")
try:
    def generate_rep_code_bit_flips(d, noise):
        c = stim.Circuit.generated("repetition_code:memory", distance=d, rounds=3*d, after_clifford_depolarization=noise)
        return stim.Circuit(str(c).replace("DEPOLARIZE1", "X_ERROR"))

    tasks = [
        sinter.Task(
            circuit=generate_rep_code_bit_flips(d, noise),
            json_metadata={'d': d, 'p': noise},
        )
        for d in [3, 5]
        for noise in [0.001, 0.01, 0.05, 0.1]
    ]
    print("Collecting Sinter stats...")
    stats = sinter.collect(num_workers=4, tasks=tasks, decoders=['pymatching'], max_shots=5000, max_errors=500)

    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    ax.loglog()
    ax.set_title("Cat Repetition Code Threshold (Sinter)")
    ax.grid(which='major')
    plt.savefig("sinter_threshold.png")
    print("Saved sinter_threshold.png")

except Exception as e:
    print(f"Task 4 failed: {e}")
print("========================================")

# --- Task 5 ---
print("\n=== Task 5: Hamming Code Output ===")
try:
    def hamming_7_4_x_memory(p):
        c = stim.Circuit()
        c.append("R", range(10))
        c.append("X_ERROR", range(7), p)
        c.append("H", [7])
        for q in [0, 2, 4, 6]: c.append("CX", [q, 7])
        c.append("H", [7])
        c.append("M", [7])
        c.append("DETECTOR", [stim.target_rec(-1)])
        c.append("H", [8])
        for q in [1, 2, 5, 6]: c.append("CX", [q, 8])
        c.append("H", [8])
        c.append("M", [8])
        c.append("DETECTOR", [stim.target_rec(-1)])
        c.append("H", [9])
        for q in [3, 4, 5, 6]: c.append("CX", [q, 9])
        c.append("H", [9])
        c.append("M", [9])
        c.append("DETECTOR", [stim.target_rec(-1)])
        c.append("M", range(7))
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-7)], 0)
        return c

    hamming_tasks = [sinter.Task(
        circuit=hamming_7_4_x_memory(p),
        json_metadata={'p': p, 'd': 3}
    ) for p in [0.001, 0.01, 0.1]]

    print("Collecting Hamming stats...")
    hamming_stats = sinter.collect(num_workers=4, tasks=hamming_tasks, decoders=['pymatching'], max_shots=5000)

    for stat in hamming_stats:
        print(f"p={stat.json_metadata['p']}: Logical Error Rate = {stat.errors / stat.shots}")

except Exception as e:
    print(f"Task 5 failed: {e}")
print("===================================")
