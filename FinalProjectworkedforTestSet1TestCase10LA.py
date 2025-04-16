import numpy as np
M = 2
Mc = 2
N = 2
R = 1/2
Es = 4
EbN0 = 2
sigma2 = (Es/2)*(N/(R*M*Mc))*(10**(-EbN0/10))
print("sigma2: ", sigma2)
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
    return np.array((1 - 2 * bits[:, 1]) + 1j * (1 - 2 * bits[:, 0]))  # QPSK mapping

num_sequences = 2**(M * Mc-1)# Number of sequences
#print("num_sequences: ", num_sequences)
num_bits = int(np.ceil(np.log2(num_sequences)))  # Number of bits per sequence
ints = np.arange(num_sequences) # Generate the integers
bit_sequences = ((ints[:, None] & (1 << np.arange(num_bits)[::-1])) > 0).astype(int)# Convert to binary with fixed length

print(bit_sequences)

H = np.array([[ 0.5+1.1j,  0.2-0.6j],[-1.4+0.6j,  0.2-1.j ]])
Y = np.array([-1.6-0.4j,  2. -0.j ]).reshape(-1, 1)
print("Y: ", Y)
La = np.array([0,0,0,0], dtype=float)
Ld = np.array([0,0,0,0], dtype=float)

for k in range (M*Mc):
    print("k: ", k)
    PlusSeq = np.zeros((num_sequences, M*Mc), dtype=int)  # Initialize PlusSeq
    #print("PlusSeq: ", PlusSeq.shape)
    MinusSeq = np.zeros((num_sequences, M*Mc), dtype=int)  # Initialize MinusSeq
    SPlus = np.zeros((num_sequences, M), dtype=complex)  # Initialize SPlus
    SMinus = np.zeros((num_sequences, M), dtype=complex)  # Initialize SMinus
    for i in range(num_sequences):
        PlusSeq[i] = np.insert(bit_sequences[i], k, 1)
        MinusSeq[i] = np.insert(bit_sequences[i], k, 0)
        SPlus[i] = qpsk_mapping(PlusSeq[i])
        SMinus[i] = qpsk_mapping(MinusSeq[i])
    #print("PlusSeq: ", PlusSeq)
    #print('Splus: ', SPlus)
    i = 0
    PlusSumAccum = 0
    MinusSumAccum = 0
    for j in range (M*Mc):
            if j != k:
                PlusSumAccum += -(1 - 2*PlusSeq[i][j])*La[j]
                MinusSumAccum += -(1 - 2*MinusSeq[i][j])*La[j]
    PlusAccum = -1 / (2 * sigma2) * np.linalg.norm(Y - H @ SPlus[i].reshape(-1, 1))**2 + 0.5 * PlusSumAccum
    MinusAccum = -1 / (2 * sigma2) * np.linalg.norm(Y - H @ SMinus[i].reshape(-1, 1))**2 + 0.5 * MinusSumAccum
    print()
    #print("PlusAccum = ", PlusAccum, " = -1 * (2 * ", sigma2,"* np.linalg.norm(", Y," - ", H," @ ",SPlus[i].reshape(-1, 1),"**2 + 0.5 * ", PlusSumAccum)
    #print("H @ SPlus[i].reshape(-1, 1): ", H @ SPlus[i].reshape(-1, 1))
    for i in range(1, num_sequences):
        PlusSumAccum = 0
        MinusSumAccum = 0
        for j in range (M*Mc):
            if j != k:
                PlusSumAccum += -(1 - 2*PlusSeq[i][j])*La[j]
                MinusSumAccum += -(1 - 2*MinusSeq[i][j])*La[j]
        #print("SPlus[i]",SPlus[i])
        PlusAccum = maxStar(PlusAccum, -1 / (2 * sigma2) * np.linalg.norm(Y - H @ SPlus[i].reshape(-1, 1))**2 + 0.5 * PlusSumAccum)
        MinusAccum = maxStar(MinusAccum, -1 / (2 * sigma2) * np.linalg.norm(Y - H @ SMinus[i].reshape(-1, 1))**2 + 0.5 * MinusSumAccum)
        print("PlusAccum = ", PlusAccum, " = -1 * (2 * ", sigma2,"* np.linalg.norm(", Y," - ", H," @ ",SPlus[i].reshape(-1, 1),"**2 + 0.5 * ", PlusSumAccum)
    print(k, "PlusAccum: ", PlusAccum, "MinusAccum: ", MinusAccum)                        
    Ld[k] = La[k] + PlusAccum - MinusAccum

print("Ld: ", Ld)