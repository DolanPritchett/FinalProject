# Iterative MIMO-Turbo Receiver for 5G Wireless Systems

## What This Project Does

This project simulates a **near-optimal wireless receiver** that combines two key technologies used in modern 5G systems:

1. **MIMO Detection** - Decoding signals from multiple antennas simultaneously
2. **Turbo/LDPC Decoding** - Error correction codes that approach theoretical limits

By having these components exchange "soft" reliability information iteratively, the receiver achieves significantly better performance than processing them separately.

**Key Result:** At 2 dB SNR, the system achieves ~1% bit error rate, compared to ~20% without iterative processing.

---

## Why It Matters

Modern wireless systems (5G, WiFi 6) use both multiple antennas and advanced error correction. The challenge is that optimal joint processing is computationally prohibitive. This project implements a practical iterative approach that trades a few iterations for near-optimal performance.

---

## Quick Start

```python
from FinalProject import process_mimo_decoder, encoder75

# Encode information bits
info_bits = np.random.randint(0, 2, 1280)
encoded = encoder75(info_bits)

# Simulate channel and decode
output_llrs = process_mimo_decoder(H_matrices, received_signals, Es=4, EbN0=2)
decoded_bits = (output_llrs > 0).astype(int)
```

---

## System Overview

```
Transmitter:  Info Bits → Turbo Encoder → Interleaver → QPSK → 2×2 MIMO
                                                              ↓
                                                         Channel + Noise
                                                              ↓
Receiver:     Decoded Bits ← Turbo Decoder ←→ MIMO Detector (iterative)
```

---

## Key Components

| Component | Function | Algorithm |
|-----------|----------|-----------|
| Turbo Encoder | Error correction (rate 1/2) | (7,5) SRCC with puncturing |
| MIMO Detector | Soft symbol detection | Max-log-MAP with max* |
| BCJR Decoder | Trellis-based soft decoding | Forward-backward algorithm |

---

## Performance

![BER Curve Placeholder]

| Eb/N0 (dB) | Turbo BER | LDPC BER |
|------------|-----------|----------|
| 1.0 | 20.0% | 19.7% |
| 2.0 | 6.8% | 10.9% |
| 2.5 | 1.1% | 4.2% |
| 3.0 | <0.01% | 0.08% |

---

## Project Structure

```
├── FinalProject.py          # Main simulation (encoder + iterative receiver)
├── turbo_decoder_final.py   # BCJR algorithm implementation
├── Combined.py              # Integration testing
└── CourseProjectCodeReview.py  # Validation against test vectors
```

---

## Technical Details

<details>
<summary>Click to expand system parameters</summary>

- **MIMO Configuration:** 2×2 (2 TX, 2 RX antennas)
- **Modulation:** QPSK (2 bits/symbol)
- **Code Rate:** 1/2 (after puncturing)
- **Generator Polynomials:** G = [7, 5]₈ (octal)
- **Iterations:** 4 outer (MIMO↔Turbo) × 8 inner (Turbo decoder)

</details>

---

## References

- Hochwald & ten Brink, "Achieving near-capacity on a multiple-antenna channel," IEEE Trans. Comm., 2003
- Berrou et al., "Near Shannon limit error-correcting coding: Turbo-codes," IEEE ICC, 1993

---

*Course project for ECE 7670 - Digital Communications*
