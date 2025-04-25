import numpy as np
import h5py
import os
import sys
import scipy
import copy
import numpy.matlib
from numpy import vectorize, arange
import pybind_11_ldpc_setup as pbe

def generate_complex_array(rows, cols, scale):
    real = np.random.normal(loc=0, scale=scale, size=(rows, cols))
    imag = np.random.normal(loc=0, scale=scale, size=(rows, cols))
    return real + 1j * imag

def interleaver(intlv_pattern, in_seq):    
    if len(intlv_pattern) != len(in_seq):
        print(f'ERROR: interleaver pattern length is not matched with input sequence')
        return None
        
    out_seq = []
    for i in range(len(in_seq)):        
        out_seq += [in_seq[intlv_pattern[i]]]
    out_seq = np.array(out_seq)
    return out_seq

def de_interleaver(intlv_pattern, in_seq):    
    if len(intlv_pattern) != len(in_seq):
        print(f'ERROR: interleaver pattern length is not matched with input sequence')
        return None
    
    de_intlv_pattern = np.zeros(len(intlv_pattern), dtype=int)
    for i, index in enumerate(intlv_pattern):
        de_intlv_pattern[index] = i

    out_seq = []
    for i in range(len(in_seq)):               
        out_seq += [in_seq[de_intlv_pattern[i]]]        
    out_seq = np.array(out_seq)
    return out_seq

def qpsk_mapping(bits):
        if len(bits) % 2 != 0:
            raise ValueError("Input bit array length must be even for QPSK mapping.")
        bits = np.array(bits).reshape(-1, 2)
        return np.array((1 - 2 * bits[:, 1]) + 1j * (1 - 2 * bits[:, 0]))

def compute_ld_le(H, Y, La, Es, EbN0):


    num_sequences = 2 ** (M * Mc - 1)
    num_bits = int(np.ceil(np.log2(num_sequences)))
    ints = np.arange(num_sequences)
    bit_sequences = ((ints[:, None] & (1 << np.arange(num_bits)[::-1])) > 0).astype(int)

    Ld = np.zeros(M * Mc, dtype=float)
    Le = np.zeros(M * Mc, dtype=float)

    def maxStar(x, y):
        x, y = float(x), float(y)
        return max(x, y) + np.log1p(np.exp(-abs(x - y)))

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

def process_mimo_decoder(H, Y, Es, EbN0):
    Ld = np.zeros(Y.size * 2, dtype=float)
    Le = np.zeros(Y.size * 2, dtype=float)
    print('Le.size:',Le.size)
    interleavedForMIMO = np.zeros(Y.size * 2, dtype=float)

    for outer_iter in range(4):  # Outer loop iterates 2 times
        print(f"Outer iteration {outer_iter + 1}\n")
        for i in range(len(H)):
            matrix = H[i]
            Recd = Y[i].reshape(-1, 1)
            Ld[4 * i : 4 * i + 4], Le[4 * i : 4 * i + 4] = compute_ld_le(
                matrix, Recd, interleavedForMIMO[4 * i : 4 * i + 4], Es, EbN0
            )

        
        deinterleavedLd = de_interleaver(channel_interleaver_pattern, Ld)
        deinterleavedLe = de_interleaver(channel_interleaver_pattern, Le)

        received_seq = deinterleavedLe
        inv_recd_seq = - received_seq
        prior = inv_recd_seq.tolist()
        for inner_iter in range(100):
            out_APP_c = pbe.ldpc_decoder(address, prior, LDPC_CODELEN)
            out_APP_c = np.array(out_APP_c, dtype=np.double)
            out_APP_m = pbe.ldpc_Ext(address, out_APP_c.tolist())
            out_APP_m = np.array(out_APP_m, dtype=np.double)
            hard_code = (out_APP_m < 0)
            prior = out_APP_c.tolist()
        ForMIMO = np.array(inv_recd_seq) - np.array(prior)  # Fixed subtraction
        interleavedForMIMO = interleaver(channel_interleaver_pattern, ForMIMO).tolist()
    return hard_code   


file_name = "./LDPC_CODE/5G_LDPC_M10_N20_Z128_Q2_nonVer.txt"
LDPC_CHKLEN = 1280
LDPC_CODELEN = 2560

address = pbe.ldpc_initial(LDPC_CHKLEN, LDPC_CODELEN, file_name)
LDPC_INFOLEN = pbe.get_ldpc_infoLen(address)
print("LDPC_CHKLEN=", LDPC_CHKLEN, "LDPC_CODELEN=", LDPC_CODELEN, "LDPC_INFOLEN=", LDPC_INFOLEN)

Es=4

M = 2
Mc = 2
R = 1 / 2

Mc = 2  # QPSK
Mt = 2
Nr = 2
Code_R = 0.5


# MODELING FOR TURBO CODES
H = [generate_complex_array(2, 2, np.sqrt(0.5)) for _ in range(int(LDPC_CODELEN/4))]
#H = [np.eye(2) for _ in range(641)]
#print('H[0]:',H[0])

''' do the modeling for the turbo'''
#intlv_pattern = np.random.permutation(np.arange(0, 1282))
channel_interleaver_pattern = np.random.permutation(np.arange(0, LDPC_CODELEN))

import matplotlib.pyplot as plt
length_u = LDPC_INFOLEN
u = np.random.randint(0, 2, length_u)
code = pbe.ldpc_encoder(address, u, LDPC_INFOLEN, LDPC_CODELEN)
code = np.array(code)

Intcode = interleaver(channel_interleaver_pattern, code)  # Interleaved output
QPSKcode = qpsk_mapping(Intcode)  # BPSK Mapping: 0 → -1, 1 → +1
snr_values = np.array([ 1, 2, 3, 4, 5, 6])  # SNR values in dB
BER = np.zeros(len(snr_values), dtype=float)
for j in range(snr_values.size):
    EbN0=snr_values[j]
    sigma2 = (Es / 2) * (Nr / (Code_R * Mt * Mc)) * (10 ** (-EbN0 / 10))

    EsN0 = EbN0 - 10 * np.log10(Nr / (Code_R * Mt * Mc))  # in dB
    EsN0 = np.round(10 ** (EsN0 / 10), decimals=4)
    Y = np.zeros((int(LDPC_CODELEN/4),2), dtype=complex)  # Initialize Y with zeros
    for i in range(int(LDPC_CODELEN/4)):
        Y[i] = (H[i] @ QPSKcode[2*i:2*i+2].reshape(-1,1) + generate_complex_array(2, 1, np.sqrt(sigma2))).reshape(1,2)
        Y = np.array(Y)
    Output = process_mimo_decoder(H, Y, Es, EbN0)
    #MappedOutput = qpsk_mapping(Output)  # BPSK Mapping: 0 → -1, 1 → +1
    #print('MappedOutput[0:10]:',MappedOutput[0:10])
    #print('Y[0:10]:',Y[0:10])
    #print('output.size:',Output.size)
    #print('u.size:',u.size)
    #Output = 2*u-1
    BER[j] = np.sum((Output[0:length_u])!=u)/length_u
print(f'BER: {BER}')

# Plot BER vs SNR
plt.figure(figsize=(8, 6))
plt.plot(snr_values, np.log10(BER), marker='o', linestyle='-', color='b', label='BER')
plt.xlabel('SNR (Linear Domain)', fontsize=12)
plt.ylabel('log10(BER)', fontsize=12)
plt.title('BER vs SNR', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.show()
pbe.ldpc_clear(address)
"""
SNR_dB = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4])
min_wec = 100
for i in range(len(SNR_dB)):
    wec = 0  # word error number
    bec = 0  # bit error number
    bec_mesg = 0  # message bit error number
    tot = 0  # total block number
    while(wec < min_wec and tot < 10000):
        # generated message
        
        
        noise_var = 10**((-SNR_dB[i])/10)
        noise = np.random.multivariate_normal(np.zeros(2), noise_var * np.eye(2), size=LDPC_CODELEN).view(np.complex128)
        noise = np.squeeze(noise)
        r = mod_code + np.real(noise)
        prior = 2*r/noise_var
        prior = prior.tolist()
        out_APP_c = pbe.ldpc_decoder(address, prior, LDPC_CODELEN)
        out_APP_c = np.array(out_APP_c, dtype=np.double)
        out_APP_m = pbe.ldpc_Ext(address, out_APP_c.tolist())
        out_APP_m = np.array(out_APP_m, dtype=np.double)
        hard_code = (out_APP_c<0)
        bit_err = sum(abs(hard_code-code))
        bit_mesg_err = sum(abs((out_APP_m<0)-mesg))
        bec = bec + bit_err
        bec_mesg = bec_mesg + bit_mesg_err
        if bit_mesg_err != 0:
            wec = wec + 1
        tot = tot + 1
    ber = bec / tot / len(code)
    ber_mesg = bec_mesg / tot / len(mesg)
    wer = wec / tot
    print('SNR(dB) = ', SNR_dB[i], 'tot = ', tot, 'ber = ', ber, 'ber_mesg = ', ber_mesg, 'wer = ', wer)

pbe.ldpc_clear(address)
"""