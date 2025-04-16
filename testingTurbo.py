import numpy as np
from numpy import array
import commpy.channelcoding.convcode as cc
import commpy.channelcoding.turbo as turbo
from commpy.utilities import RandomInterleaver

# Define parameters
memory = array([2])
g_matrix = array([[7, 5]])  # G(D) = [1+D^2, 1]
feedback = np.array([[7]])
trellis = cc.Trellis(memory, g_matrix, feedback=feedback, code_type='rsc')
interleaver = RandomInterleaver(5000)  # Interleaver for 5000 bits
num_iterations = 10
noise_variance = 0.5  # Example noise variance for AWGN channel

# Generate random input sequence
msg_bits = np.random.randint(0, 2, 5000)

# Turbo encode the sequence
encoded = turbo.turbo_encode(msg_bits, trellis, trellis, interleaver)

# Simulate AWGN channel
received = np.array(encoded) + np.random.normal(0, np.sqrt(noise_variance), len(encoded))

# Separate received systematic and parity bits
sys_bits = received[::3]
parity1_bits = received[1::3]
parity2_bits = received[2::3]

# Turbo decode the sequence
decoded_bits = turbo.turbo_decode(sys_bits, parity1_bits, parity2_bits, trellis, noise_variance, num_iterations, interleaver)

# Calculate bit error rate (BER)
bit_errors = np.sum(msg_bits != decoded_bits)
ber = bit_errors / len(msg_bits)

print(f"Bit Error Rate (BER): {ber:.6f}")