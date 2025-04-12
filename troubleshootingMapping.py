import numpy as np
M = 2
Mc = 2

def maxStar(x,y):
    x, y = float(x), float(y)
    return (max(x, y) + np.log1p(np.exp(-abs(x - y))))

def qpsk_mapping(bits):
    """
    Maps an array of bits to QPSK symbols.
    Assumes the input array has an even length.
    """
    if len(bits) % 2 != 0:
        raise ValueError("Input bit array length must be even for QPSK mapping.")
    bits = np.array(bits).reshape(-1, 2)  # Group bits into pairs
    return (1 - 2 * bits[:, 1]) + 1j * (1 - 2 * bits[:, 0]) # QPSK mapping


received_seq = [-1.6-0.4j,  2. -0.j ]

toEncode = np.array([0,0,0,1])

Encoded = qpsk_mapping(toEncode)
print("Encoded: ", Encoded)
S = Encoded.reshape(-1,1) # Reshape to column vector
print("s: ", S)