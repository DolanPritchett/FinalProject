import numpy as np

def compute_ld_le(H, Y, La):
    M = 2
    Mc = 2
    R = 1 / 2
    Es = 4
    EbN0 = 2
    sigma2 = (Es / 2) * (2 / (R * M * Mc)) * (10 ** (-EbN0 / 10))

    num_sequences = 2 ** (M * Mc - 1)
    num_bits = int(np.ceil(np.log2(num_sequences)))
    ints = np.arange(num_sequences)
    bit_sequences = ((ints[:, None] & (1 << np.arange(num_bits)[::-1])) > 0).astype(int)

    Ld = np.zeros(M * Mc, dtype=float)
    Le = np.zeros(M * Mc, dtype=float)

    def maxStar(x, y):
        x, y = float(x), float(y)
        return max(x, y) + np.log1p(np.exp(-abs(x - y)))

    def qpsk_mapping(bits):
        if len(bits) % 2 != 0:
            raise ValueError("Input bit array length must be even for QPSK mapping.")
        bits = np.array(bits).reshape(-1, 2)
        return np.array((1 - 2 * bits[:, 1]) + 1j * (1 - 2 * bits[:, 0]))

    for k in range(M * Mc):
        PlusSeq = np.zeros((num_sequences, M * Mc), dtype=int)
        MinusSeq = np.zeros((num_sequences, M * Mc), dtype=int)
        SPlus = np.zeros((num_sequences, M), dtype=complex)
        SMinus = np.zeros((num_sequences, M), dtype=complex)

        for i in range(num_sequences):
            PlusSeq[i] = np.insert(bit_sequences[i], k, 1)
            MinusSeq[i] = np.insert(bit_sequences[i], k, 0)
            SPlus[i] = qpsk_mapping(PlusSeq[i])
            SMinus[i] = qpsk_mapping(MinusSeq[i])

        PlusAccum = -1 / (2 * sigma2) * np.linalg.norm(Y - H @ SPlus[0].reshape(-1, 1)) ** 2
        MinusAccum = -1 / (2 * sigma2) * np.linalg.norm(Y - H @ SMinus[0].reshape(-1, 1)) ** 2

        for j in range(M * Mc):
            if j != k:
                PlusAccum += 0.5 * -(1 - 2 * PlusSeq[0][j]) * La[j]
                MinusAccum += 0.5 * -(1 - 2 * MinusSeq[0][j]) * La[j]

        for i in range(1, num_sequences):
            PlusSumAccum = 0
            MinusSumAccum = 0
            for j in range(M * Mc):
                if j != k:
                    PlusSumAccum += -(1 - 2 * PlusSeq[i][j]) * La[j]
                    MinusSumAccum += -(1 - 2 * MinusSeq[i][j]) * La[j]
            PlusAccum = maxStar(
                PlusAccum,
                -1 / (2 * sigma2) * np.linalg.norm(Y - H @ SPlus[i].reshape(-1, 1)) ** 2 + 0.5 * PlusSumAccum,
            )
            MinusAccum = maxStar(
                MinusAccum,
                -1 / (2 * sigma2) * np.linalg.norm(Y - H @ SMinus[i].reshape(-1, 1)) ** 2 + 0.5 * MinusSumAccum,
            )

        Ld[k] = La[k] + PlusAccum - MinusAccum
        Le[k] = Ld[k] - La[k]

    return Ld, Le

# Example usage
H = np.array([[0.5 + 1.1j, 0.2 - 0.6j], [-1.4 + 0.6j, 0.2 - 1.0j]])
Y = np.array([-1.6 - 0.4j, 2.0 - 0.0j]).reshape(-1, 1)
La = np.array([0,0,0,0], dtype=float)

Ld, Le = compute_ld_le(H, Y, La)
print("======== Test Case 1 ============")
print("Expected Output from MIMO Detector when La=[0, 0, 0, 0]")
print("Ld: ", Ld)
print("Le: ", Le)

La = np.array([1.2, -0.5, -1.5, 2], dtype=float)
Ld, Le = compute_ld_le(H, Y, La)

print("Expected Output from MIMO Detector when La=[1.2, -0.5, -1.5, 2]")
print("Ld: ", Ld)
print("Le: ", Le)

H = np.array(
 [[ 0.7-0.9j, -0.5-1.1j],
 [-0.6-0.5j, -2.1+1.7j]])

Y = np.array([-1.4-0.5j, -0.8-0.7j]).reshape(-1, 1)

La = np.array([0,0,0,0], dtype=float)

Ld, Le = compute_ld_le(H, Y, La)
print("======== Test Case 2 ============")
print("Expected Output from MIMO Detector when La=[0, 0, 0, 0]")
print("Ld: ", Ld)
print("Le: ", Le)

La = np.array([1.2, -0.5, -1.5, 2], dtype=float)
Ld, Le = compute_ld_le(H, Y, La)

print("Expected Output from MIMO Detector when La=[1.2, -0.5, -1.5, 2]")
print("Ld: ", Ld)
print("Le: ", Le)

H = np.array(
 [[0.1-0.7j,-1.8-0.8j],
 [-0.2+0j,0.2+1j]])

Y = np.array([-0.7+0j,-0.9+0.1j]).reshape(-1, 1)

La = np.array([0,0,0,0], dtype=float)

Ld, Le = compute_ld_le(H, Y, La)
print("======== Test Case 3 ============")
print("Expected Output from MIMO Detector when La=[0, 0, 0, 0]")
print("Ld: ", Ld)
print("Le: ", Le)

La = np.array([1.2, -0.5, -1.5, 2], dtype=float)
Ld, Le = compute_ld_le(H, Y, La)

print("Expected Output from MIMO Detector when La=[1.2, -0.5, -1.5, 2]")
print("Ld: ", Ld)
print("Le: ", Le)