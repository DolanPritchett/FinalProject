import numpy as np
M = 2
Mc = 2

num_sequences = 2**(M * Mc-1)# Number of sequences
num_bits = int(np.ceil(np.log2(num_sequences)))  # Number of bits per sequence
ints = np.arange(num_sequences) # Generate the integers
bit_sequences = ((ints[:, None] & (1 << np.arange(num_bits)[::-1])) > 0).astype(int)# Convert to binary with fixed length

print(bit_sequences)
for k in range (M*Mc):
    PlusSeq = np.zeros((num_sequences, M*Mc), dtype=int)  # Initialize PlusSeq
    MinusSeq = np.zeros((num_sequences, M*Mc), dtype=int)  # Initialize MinusSeq
    for i in range(num_sequences):
        PlusSeq[i] = np.insert(bit_sequences[i], k, 1)
        MinusSeq[i] = np.insert(bit_sequences[i], k, 0)
    for i in range(num_sequences):
        for j in range (M*Mc):
            if j != k:
                SumAccum += PlusSeq[i][j]*La[j]
        PlusAccum = -1(2*sigma2)*np.linalg.norm(Y - H @ S)**2 + .5*SumAccum
    MinusAccum = MinusSeq[0]
    for i in range(num_sequences):
       