
import stim
import os

# Ensure we are in the right directory
# (User is running from team-Ritesh_Roshan_Sahoo usually)

def save_diagram(circuit, filename, type="timeline-svg"):
    print(f"Generating {filename}...")
    with open(filename, "w") as f:
        print(circuit.diagram(type), file=f)
    print(f"Saved {filename}")

# --- Task 2.1: Error Detection ---
circuit_ed = stim.Circuit()
circuit_ed.append("R", [0, 1, 2])
circuit_ed.append("H", [0])
circuit_ed.append("CX", [0, 1])
circuit_ed.append("X_ERROR", [0, 1], 0.1)
circuit_ed.append("CX", [0, 2])
circuit_ed.append("CX", [1, 2])
circuit_ed.append("M", [2])
# Annotations for diagram clarity
circuit_ed.append("DETECTOR", [stim.target_rec(-1)], [0, 0, 0]) 
circuit_ed.append("M", [0, 1])
circuit_ed.append("OBSERVABLE_INCLUDE", [stim.target_rec(-2)], 0)

save_diagram(circuit_ed, "circuit_error_detection.svg")


# --- Task 3: Repetition Code (d=3) ---
# We use a slightly cleaner version for the diagram (no noise for clarity, or low noise)
def repetition_code_circuit(n: int) -> stim.Circuit:
    c = stim.Circuit()
    c.append("R", range(2*n-1))
    # data_qubits = [0, 2, 4] for n=3
    # ancilla = [1, 3]
    
    # 3 rounds
    for round in range(3):
        c.append("TICK")
        for i in range(n-1):
            # Sybdrome Extraction
            # 2*i is data, 2*i+1 ancilla, 2*i+2 data
            c.append("CX", [2*i, 2*i+1])
            c.append("CX", [2*i+2, 2*i+1])
        c.append("M", [1, 3])
        c.append("DETECTOR", [stim.target_rec(-2), stim.target_rec(-1)])
        
    c.append("TICK")
    c.append("M", [0, 2, 4])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-3)], 0)
    return c

circuit_rep = repetition_code_circuit(n=3)
save_diagram(circuit_rep, "circuit_repetition_d3.svg")


# --- Task 5: Hamming Code ---
def hamming_circuit() -> stim.Circuit:
    c = stim.Circuit()
    c.append("R", range(10))
    # Stabilizers
    # 7 is anc for 0,2,4,6
    c.append("H", [7])
    for q in [0, 2, 4, 6]: c.append("CX", [q, 7])
    c.append("H", [7])
    c.append("TICK")
    
    # 8 is anc for 1,2,5,6
    c.append("H", [8])
    for q in [1, 2, 5, 6]: c.append("CX", [q, 8])
    c.append("H", [8])
    c.append("TICK")
    
    # 9 is anc for 3,4,5,6
    c.append("H", [9])
    for q in [3, 4, 5, 6]: c.append("CX", [q, 9])
    c.append("H", [9])
    c.append("TICK")
    
    c.append("M", [7, 8, 9])
    c.append("DETECTOR", [stim.target_rec(-3), stim.target_rec(-2), stim.target_rec(-1)])
    
    c.append("M", range(7))
    return c

circuit_hamming = hamming_circuit()
save_diagram(circuit_hamming, "circuit_hamming.svg")
