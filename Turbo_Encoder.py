import numpy as np

def encoder75(InputSequence):
    

    def SRCCencoder(InputSequence, *args):
        TerminatingBits = np.zeros(args[0].size-1, dtype=int) #initialize the array for terminating bits
        AppendedSequence = np.append(InputSequence,TerminatingBits).astype(int) # append the initialized but not populated teminating bits to the input sequence
        Outputs = np.zeros([AppendedSequence.size, len(args)], dtype=int)
        ShiftRegister = np.zeros(args[0].size-1, dtype=int)
        # ShiftRegister[0] = InputSequence[0]
        for j in range(AppendedSequence.size):
            if j >= InputSequence.size:
                AppendedSequence[j] = np.bitwise_xor.reduce(args[0][1:] * ShiftRegister)
                
            Outputs[j,0] = AppendedSequence[j]
            #print('output[:,0]:', Outputs[:,0])
            #print('appended sequence:', AppendedSequence[j])
            RecursiveT0 = np.bitwise_xor.reduce(np.append(AppendedSequence[j],args[0][1:] * ShiftRegister))
            components = np.append(RecursiveT0,ShiftRegister)
            
            for i in range(1,len(args)):
                
                Outputs[j,i] = np.bitwise_xor.reduce(components*args[i])

            #Feedback = args[0]*components
            ShiftRegister[1:ShiftRegister.size] = ShiftRegister[0:ShiftRegister.size-1]
            ShiftRegister[0] = RecursiveT0

        return Outputs[:,0],Outputs[:,1]

    def TurboEncoder(InputSequence, Interleaver, Encoder, gen_poly):
        poly1, poly2 = bitfield(gen_poly[0]), bitfield(gen_poly[1])
        Outputs = np.zeros([InputSequence.size + poly1.size -1, 3], dtype=int)
        Outputs[:,0],Outputs[:,1] = Encoder(InputSequence,poly1,poly2)
        #print(f'Outputs[:,0]: {Outputs[:,0]}')
        #print(f'Outputs[:,1]: {Outputs[:,1]}')
        Interleaved = Interleaver(intlv_pattern, Outputs[:,0])
        Waste, temp = Encoder(Interleaved, poly1,poly2)
        Outputs[:,2] = temp[0:InputSequence.size+poly1.size -1]
        #print(f'Outputs[:,2]: {Outputs[:,2]}')
        return Outputs

    def interleaver(intlv_pattern,in_seq):    
            if len(intlv_pattern)!=len(in_seq):
                print(f'ERROR: interleaver pattern length is not matched with input sequence')
                return None
                
            out_seq=[]
            for i in range(len(in_seq)):        
                out_seq+=[in_seq[intlv_pattern[i]]]
            out_seq=np.array(out_seq)
            return out_seq

    def puncturing(systematic, parity1, parity2, punc_mat):
            """
            Punctures the input systematic, parity1, and parity2 arrays based on the puncturing matrix.

            Args:
                systematic (np.ndarray): The systematic bits array.
                parity1 (np.ndarray): The parity 1 bits array.
                parity2 (np.ndarray): The parity 2 bits array.
                punc_mat (np.ndarray): The puncturing matrix.

            Returns:
                np.ndarray: The punctured output sequence.
            """
            punc_vec = punc_mat.flatten()  # Flatten the puncturing matrix into a 1D array
            punctured_out = []

            for i in range(len(systematic)):
                # Append values based on the puncturing pattern
                if punc_vec[(3 * i) % len(punc_vec)]:
                    punctured_out.append(systematic[i])
                if punc_vec[(3 * i + 1) % len(punc_vec)]:
                    punctured_out.append(parity1[i])
                if punc_vec[(3 * i + 2) % len(punc_vec)]:
                    punctured_out.append(parity2[i])

            return np.array(punctured_out)

    intlv_pattern = np.array([12, 5, 9, 2, 10, 7, 1, 14, 6, 11, 3, 8, 4, 13])
    intlv_pattern = intlv_pattern - 1
    punc_matrix = np.array([[True, True], [True, False], [False, True]]).T

    def bitfield(n):
                return np.array([int(digit) for digit in bin(n)[2:]])

    gen_poly = [0o7, 0o5]

    before_punc = TurboEncoder(InputSequence, interleaver, SRCCencoder, gen_poly)
    punctured = puncturing(before_punc[:,0], before_punc[:,1], before_punc[:,2], punc_matrix)


    return punctured

InputSequence = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])
punctured = encoder75(InputSequence)
print(f'Punctured Output: {punctured}')

random_array = np.random.permutation(np.arange(0, 8))
random_array2 = np.random.permutation(np.arange(0, 4))
print(f'Random Array: {random_array}')
print(f'Random Array 2: {random_array2}')

h_real = np.random.normal(loc=0, scale=np.sqrt(0.5))
h_imag = np.random.normal(loc=0, scale=np.sqrt(0.5))

# Construct the complex number
h_ij = h_real + 1j * h_imag

print("h_ij:", h_ij)
print("Magnitude squared:", abs(h_ij)**2)

def generate_complex_array(rows, cols):
    real = np.random.normal(loc=0, scale=np.sqrt(0.5), size=(rows, cols))
    imag = np.random.normal(loc=0, scale=np.sqrt(0.5), size=(rows, cols))
    return real + 1j * imag

# Create a list of 10 such arrays
array_list = [generate_complex_array(2, 2) for _ in range(10)]

# Example: print the first one
print("First 4x4 complex array:")
print(array_list[0],array_list[1])