import numpy as np
from turbo_decoder.turbo_decoder_final import BCJR_decoder2, interleaver, de_interleaver
from Matlab_Python_LDPC_update.Matlab_Python_LDPC.Python_version.main import pbe

# Parameters
M = 2  # Number of TX antennas
N = 2  # Number of RX antennas
Mc = 2  # Bits per QPSK symbol
LDPC_CHKLEN = 1280
LDPC_CODELEN = 2560
LDPC_INFOLEN = 1280
num_outer_iter = 4
num_inner_iter = 8
SNR_dB = 2  # Example SNR value
gen_poly = [0o37, 0o21]
intlv_pattern = np.array([5, 7, 16, 4, 1, 19, 10, 15, 3, 20, 12, 8, 14, 2, 17, 11, 18, 6, 13, 9]) - 1

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
    return (1 - 2 * bits[:, 1]) + 1j * (1 - 2 * bits[:, 0])  # QPSK mapping

# Initialize LDPC
file_name = "./LDPC_CODE/5G_LDPC_M10_N20_Z128_Q2_nonVer.txt"
address = pbe.ldpc_initial(LDPC_CHKLEN, LDPC_CODELEN, file_name)

# Channel and noise setup
H = (np.random.randn(N, M) + 1j * np.random.randn(N, M))/2**(.5)  # Channel matrix
sigma2 = 10 ** (-SNR_dB / 10)
noise = np.sqrt(sigma2 / 2) * (np.random.randn(N) + 1j * np.random.randn(N))

# Generate random message and encode
message = np.random.randint(0, 2, LDPC_INFOLEN)
encoded_bits = pbe.ldpc_encoder(address, message, LDPC_INFOLEN, LDPC_CODELEN)

# QPSK mapping
S = qpsk_mapping(encoded_bits)

# Transmit signal through the channel
Y = S @ H + noise

# Iterative MIMO-Turbo-LDPC decoding
for outer_iter in range(num_outer_iter):
    print(f"Outer Iteration {outer_iter + 1}")
    
    # MIMO Detection
    LLR_mimo = np.zeros(LDPC_CODELEN)  # Initialize LLR_mimo
    for i in range(LDPC_CODELEN):
        for k in range(M*Mc):
            num_sequences = 2**(M * Mc-1)# Number of sequences
            num_bits = int(np.ceil(np.log2(num_sequences)))  # Number of bits per sequence
            ints = np.arange(num_sequences) # Generate the integers
            bit_sequences = ((ints[:, None] & (1 << np.arange(num_bits)[::-1])) > 0).astype(int)# Convert to binary with fixed length
        # Compute LLR_mimo using the formula
       
            term1 = -1 / (2 * sigma2) * np.linalg.norm(transmitted_signal - H @ s) ** 2
            
    # Turbo Decoding
    for inner_iter in range(num_inner_iter):
        print(f"  Inner Iteration {inner_iter + 1}")
        La = LLR_mimo if inner_iter == 0 else ext_llr
        llr, _, _, _, _, _ = BCJR_decoder2(gen_poly, True, False, True, La, SNR_dB, transmitted_signal)
        ext_llr = llr - La  # Extrinsic LLRs for Turbo decoding
    
    # LDPC Decoding
    decoded_bits = pbe.ldpc_decoder(address, ext_llr.tolist(), LDPC_CODELEN)
    hard_decision = np.array(decoded_bits) < 0
    bit_errors = np.sum(hard_decision != message)
    print(f"  Bit Errors: {bit_errors}")

# Cleanup LDPC
pbe.ldpc_clear(address)
