# Core Task: Classical to Quantum Codes

In this submission, we explore the adaptation of the classical [7,4,3] Hamming Code to the quantum setting to protect against bit-flip errors in biased noise architectures (cat qubits).

## Contents
- `notebook.ipynb`: Implementation and benchmarking of the Hamming Code using Stim.
- `presentation.pdf`: (Placeholder for slides)

## Results
The Hamming Code successfully detects and corrects single bit-flip errors.
Efficiency: Encodes 4 logical qubits in 7 physical qubits (Rate 4/7), which is superior to the Repetition Code (Rate 1/3) for the same distance (d=3).
