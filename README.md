# Iterative MIMO Detection with Turbo Decoding

A Python implementation of joint MIMO (Multiple-Input Multiple-Output) detection and turbo decoding for wireless communication systems. This project implements soft-input soft-output (SISO) iterative receivers that achieve near-capacity performance on multiple-antenna channels.

## Overview

This repository contains implementations of:

- **MIMO Detector**: Soft-output MIMO detector using max-log-MAP algorithm for 2×2 MIMO systems with QPSK modulation
- **Turbo Encoder**: Rate-1/2 (7,5) systematic recursive convolutional code (SRCC) based turbo encoder with puncturing
- **BCJR Decoder**: Bahl-Cocke-Jelinek-Raviv algorithm implementation for soft-output decoding
- **Iterative Receiver**: Joint iterative MIMO detection and turbo decoding with extrinsic information exchange

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSMITTER                                   │
├─────────────────────────────────────────────────────────────────┤
│  Info Bits → Turbo Encoder → Channel Interleaver → QPSK Mapper  │
│                                                    ↓            │
│                                              2×2 MIMO TX        │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Channel (H, noise)
┌─────────────────────────────────────────────────────────────────┐
│                    RECEIVER (Iterative)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐      La      ┌──────────────────┐            │
│  │    MIMO      │ ←─────────── │  Turbo Decoder   │            │
│  │   Detector   │              │  (BCJR × 2)      │            │
│  │  (Soft Out)  │ ───────────→ │                  │            │
│  └──────────────┘      Le      └──────────────────┘            │
│         ↑                              ↑                        │
│    Y (received)              Channel De-interleaver             │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### MIMO Detection
- 2×2 MIMO configuration (2 TX antennas, 2 RX antennas)
- QPSK modulation (2 bits per symbol)
- Gray-coded constellation mapping
- Soft-output LLR (Log-Likelihood Ratio) computation
- Max-star algorithm for accurate APP (A Posteriori Probability) computation

### Turbo Coding
- Rate-1/2 parallel concatenated convolutional code
- Generator polynomials: G = [7, 5] (octal)
- Systematic recursive convolutional code (SRCC) encoders
- Configurable interleaver patterns
- Puncturing for rate matching

### BCJR Decoder
- Full MAP (Maximum A Posteriori) decoding
- Optional max-log-MAP approximation for reduced complexity
- Trellis termination support
- Soft-output for both information and parity bits

## File Structure

```
├── FinalProject.py              # Main implementation with full system
├── LDPCFinalProject.py          # Alternative implementation
├── CourseProjectCodeReview.py   # Code review version with test cases
├── Combined.py                  # Combined implementation
├── turbo_decoder_final.py       # Standalone turbo decoder
├── turbo_decoder_inprogress.py  # Development version
└── Course_Project_Test Set and Code Review Input/
    ├── Course Project Code Review Input.txt
    └── Test Set 2_Iterative_MIMO_Turbo_detector.txt
```

## Installation

### Requirements
- Python 3.7+
- NumPy
- Matplotlib (for BER plotting)

```bash
pip install numpy matplotlib
```

## Usage

### Basic MIMO Detection

```python
import numpy as np
from FinalProject import compute_ld_le

# Channel matrix (2×2 complex)
H = np.array([[-0.9 + 2j, 0.2 + 2j],
              [-0.3 - 1j, 2.5 + 0.5j]])

# Received signal
Y = np.array([1 + 1.5j, 2 - 0.2j]).reshape(-1, 1)

# A priori LLRs (initially zeros)
La = np.array([0, 0, 0, 0], dtype=float)

# Compute soft outputs
Ld, Le = compute_ld_le(H, Y, La, Es=4, EbN0=2)
print(f"Soft output LLRs: {Ld}")
print(f"Extrinsic LLRs: {Le}")
```

### Turbo Encoding

```python
from FinalProject import encoder75, interleaver

# Information bits
info_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])

# Interleaver pattern
intlv_pattern = np.array([12, 5, 9, 2, 10, 7, 1, 14, 6, 11, 3, 8, 4, 13]) - 1

# Encode
encoded = encoder75(info_bits, intlv_pattern)
```

### Full Iterative Receiver

```python
from FinalProject import process_mimo_decoder

# Channel matrices for each transmission
H = [np.array([[1.8+1.5j, 0.4-0.2j],
               [1.0+0.3j, 2.2-0.9j]]),
     # ... more channel matrices
    ]

# Received signals
Y = np.array([[-0.9-1.0j, -2.0-1.4j],
              [-0.3-1.7j,  0.2+2.0j],
              # ... more received signals
             ])

# Run iterative detection/decoding
output = process_mimo_decoder(H, Y, Es=4, EbN0=2)
```

## System Parameters

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| TX Antennas | Mt | 2 | Number of transmit antennas |
| RX Antennas | Nr | 2 | Number of receive antennas |
| Modulation | - | QPSK | Quadrature Phase Shift Keying |
| Bits/Symbol | Mc | 2 | QPSK uses 2 bits per symbol |
| Code Rate | R | 1/2 | After puncturing |
| Generator Poly | G | [7, 5]₈ | Octal representation |
| Eb/N0 | - | 2 dB | Energy per bit to noise ratio |

## QPSK Constellation Mapping

```
        Im
         │
   01    │    00
  (-1+j) │  (1+j)
─────────┼─────────→ Re
   11    │    10
  (-1-j) │  (1-j)
         │
```

| Bits (b₀b₁) | Symbol |
|-------------|--------|
| 00 | 1 + j |
| 01 | -1 + j |
| 10 | 1 - j |
| 11 | -1 - j |

## Algorithm Details

### MIMO Soft-Output Detection

The detector computes LLRs using:

```
L(bₖ) = La(bₖ) + ln[Σ exp(-||Y-Hs||²/2σ² + Σⱼ≠ₖ bⱼLa(bⱼ)/2)] - ln[...]
                 s:bₖ=1                        s:bₖ=0
```

The max-star operation is used for numerical stability:
```
max*(x, y) = max(x, y) + ln(1 + exp(-|x - y|))
```

### BCJR Algorithm

The decoder computes forward (α), backward (β), and branch (γ) metrics:

1. **Branch metrics**: γₖ(s', s) = uₖ·La/2 + Lc·Σ(2cᵢ-1)rᵢ/2
2. **Forward recursion**: αₖ(s) = max*{αₖ₋₁(s') + γₖ₋₁(s', s)}
3. **Backward recursion**: βₖ(s) = max*{βₖ₊₁(s') + γₖ(s, s')}
4. **LLR computation**: L(uₖ) = max*{αₖ + γₖ + βₖ₊₁}|u=1 - max*{...}|u=0

## Test Cases

The repository includes test vectors for validation:

### Test Case 1: MIMO Detector
- Input: H₁, Y₁, La=[0,0,0,0]
- Expected Ld and Le values provided in test files

### Test Case 2: Iterative Receiver
- 2 outer iterations × 2 inner turbo iterations
- Expected intermediate LLR values at each stage
- Final soft output validation

## Performance

BER (Bit Error Rate) simulation results can be generated using:

```python
# Run BER simulation
snr_values = np.array([1, 1.5, 2, 2.5, 3, 3.5])  # dB
# ... simulation code in FinalProject.py
```

## References

1. Hochwald, B.M. and ten Brink, S., "Achieving near-capacity on a multiple-antenna channel," IEEE Trans. Communications, 2003.
2. Berrou, C., Glavieux, A., and Thitimajshima, P., "Near Shannon limit error-correcting coding and decoding: Turbo-codes," IEEE ICC, 1993.
3. Bahl, L.R., Cocke, J., Jelinek, F., and Raviv, J., "Optimal decoding of linear codes for minimizing symbol error rate," IEEE Trans. Information Theory, 1974.

## License

This project was developed for educational purposes as part of a communications systems course.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Bug fixes
- Performance optimizations
- Additional modulation schemes
- Extended MIMO configurations
