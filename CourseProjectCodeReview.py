import numpy as np

def compute_ld_le(H, Y, La):
    M = 2
    Mc = 2
    R = 1 / 2
    Es = 2
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

H = np.array(
 [[ -0.9 + 2j, 0.2 + 2j],
 [-0.3-1j, 2.5+0.5j]])

Y = np.array([1+1.5j, 2-0.2j]).reshape(-1, 1)

La = np.array([0,0,0,0], dtype=float)

Ld, Le = compute_ld_le(H, Y, La)
print("Course Project Code Review Input 1")
print("Expected Output from MIMO Detector when La=[0, 0, 0, 0]")
print("Ld: ", Ld)
print("Le: ", Le)
"""
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
"""
import numpy as np

def BCJR_decoder2(gen_poly,srcc_en,max_log_map_en,term_en,La,EsN0,received_seq):

    #Conversion of Binary to Python Integer List: '0b1101'=> [1,1,0,1]
    #the Leftest bit is LSB. ex) 1101 => G(D)=1+D+D^3 (not 1+D^2+D^3)
    def bitfield(n):
        return [int(digit) for digit in bin(n)[2:]] # [2:] to chop off the "0b" part

    #Conversion of Binary Integer List to Decimal: [1,1,0,1] => 11 
    #the Leftest bit is LSB. ex) [1,1,0,1] => 11 (not 7!!)
    def intconv(n):
        intout=0
        for idx, digit in enumerate(n):
            #print(f'idx:{idx}, digit:{digit}')
            #intout+=digit*2**(len(n)-idx)
            intout+=digit*2**idx    
        return intout

    def max_exp2(x,y):
        #return max(x,y)+np.log(1+np.exp(-np.abs(x-y)))
        return np.round(max(x,y)+np.log(1+np.exp(-np.abs(x-y))),decimals=4)                
    def max_exp(max_log_map_en, x):
        if max_log_map_en==True:
            return max(x)
        else:
            return x[0] if len(x)==1 else max_exp2(x[-1],max_exp(max_log_map_en,x[0:-1]))

    def fsm_table_srcc_gen(v,bitpoly):
        state_idx=np.linspace(0,2**v-1,2**v,dtype=int)
        state_reg=[bitfield(state_idx[i]) for i in range(2**v)]

        for i in range(len(state_idx)):
            state_reg[i]=[0] * (v-len(state_reg[i])) + state_reg[i] if len(state_reg[i]) < v else state_reg[i]  #polynomial bitwitdh matching

        fsm_table=[['none' for i in range(2**v)] for j in range(2**v)]

        for idx, shft_reg_in in enumerate(state_reg):        
            for i in range(2):
                shft_reg=np.asarray(shft_reg_in)
                in_state=intconv(shft_reg)
                #print(f'input:{i}, in state:{in_state}, shift reg:{shft_reg}')    
                j1=i  #SRCC
                #Feedback Loop    
                temp=i*bitpoly[0][0]
                for j in range(v):        
                    temp=temp^shft_reg[j]*bitpoly[0][j+1]    
                
                #Forward Path
                j2=temp*bitpoly[1][0]
                for j in range(v):        
                    j2=j2^shft_reg[j]*bitpoly[1][j+1]
                #print(f'temp:{temp}, j2:{j2}')   
                #Register Shift
                shft_reg[1:shft_reg.size]=shft_reg[0:shft_reg.size-1]
                shft_reg[0]=temp

                #FSM Output        
                out_state=intconv(shft_reg)
                #print(f'out state:{out_state}, shift reg:{shft_reg}')  
                fsm_table[in_state][out_state]=[i, j1, j2]            
        
        return fsm_table

    def fsm_table_ff_gen(v,bitpoly):
        state_idx=np.linspace(0,2**v-1,2**v,dtype=int)
        state_reg=[bitfield(state_idx[i]) for i in range(2**v)]

        for i in range(len(state_idx)):
            state_reg[i]=[0] * (v-len(state_reg[i])) + state_reg[i] if len(state_reg[i]) < v else state_reg[i]  #polynomial bitwitdh matching

        fsm_table=[['none' for i in range(2**v)] for j in range(2**v)]

        for idx, shft_reg_in in enumerate(state_reg):    
            for i in range(2):        
                shft_reg=np.asarray(shft_reg_in)
                in_state=intconv(shft_reg)
                #print(f'in state:{in_state}, shift reg:{shft_reg}')    
                #Feedforward
                j1=i*bitpoly[0][0]
                j2=i*bitpoly[1][0]
                for j in range(len(bitpoly[0])-1):
                    j1=j1^shft_reg[j]*bitpoly[0][j+1]
                    j2=j2^shft_reg[j]*bitpoly[1][j+1]
                
                #Register Shift
                shft_reg[1:shft_reg.size]=shft_reg[0:shft_reg.size-1]
                shft_reg[0]=i        

                #FSM Output
                out_state=intconv(shft_reg)
                #print(f'out state:{out_state}, shift reg:{shft_reg}')   
                fsm_table[in_state][out_state]=[i, j1, j2]
        
        return fsm_table

    def bcjr_branch_metric(La,u_l,EsN0,rx_vec,code_vec):
        #BCJR branch metric
        Lc=4*EsN0        
        n=len(code_vec) #code length at each trellis section    
        u_l2=2*(u_l-0.5)
        #print(f'u_l:{u_l}, u_l2:{u_l2}')
        corr=0.0
        for i in range(n):
            corr+=2*(code_vec[i]-0.5)*rx_vec[i]
        #Maximizing log-likelihood function is equivalent to finding 'maximum' path metric
        #cal_metric=2*(codeword[1]-0.5)*received_seq[0]+2*(codeword[2]-0.5)*received_seq[1]    
        #gamma=u_l2*(La/2)+(Lc/2)*(corr)
        gamma=u_l2*(La/2)+(corr/2)

        return gamma

    #Generalized Gamma
    def bcjr_gamma_table_gen2(La,EsN0,h,m,v,fsm_table,received_seq):        
        #gamma_table dimension: time index x leaving state x arriving state
        gamma_table=[[['none' for i in range(2**v)] for j in range(2**v)] for k in range(h+m)] 

        for k in range(h+m): #time index
            for i in range(2**v): #leaving state index
                for j in range(2**v): #arriving state index
                    if fsm_table[i][j]!='none':                                                        
                        #print(f'k index:{k}, La_ul:{La[k]}')
                        #print(f'k index:{k},{i},{j}')
                        if len(La)<h:
                            print(f'ERROR: the number of La is smaller than information sequence')
                            exit()
                        elif k<len(La):
                            La_ul=La[k]                            
                        else:  #if all or some termination bits don't have a priori LLR
                            print(f'La is shorter than input sequence and put zero value in place')
                            La_ul=0
                        u_l=fsm_table[i][j][0]
                        rx_vec= received_seq[k]
                        code_vec=fsm_table[i][j][1:3]                                                                
                        #gamma_table[k][i][j]=bcjr_branch_metric(La_ul,u_l,EsN0,rx_vec,code_vec)
                        gamma_table[k][i][j]=np.round(bcjr_branch_metric(La_ul,u_l,EsN0,rx_vec,code_vec),decimals=4)
                        
        return gamma_table

    #Generalized Alpha Table Generation
    def bcjr_alpha_table_gen2(h,m,v,max_log_map_en,fsm_table,gamma_table):
        #alpha_table dimension: time index x state index
        alpha_table=[['none' for i in range(2**v)] for k in range(h+m)]
        for k in range(h+m):
            for i in range(2**v):
                #alpha table intialization
                if k==0:
                    if i==0: #alpha(s0) at t=0
                        alpha_table[k][i]=0
                    else:    #alpha values of other states at t=0
                        alpha_table[k][i]=-50            
                
                #alpha table generation
                else:
                    max_list=[]
                    #for j in range (len(gamma_table[0])):
                    for j in range (2**v):
                        if fsm_table[j][i]!='none':
                            max_list.append(alpha_table[k-1][j]+gamma_table[k-1][j][i])
                        #if gamma_table[k-1][j][i]!='none':
                        #    max_list.append(alpha_table[k-1][j]+gamma_table[k-1][j][i])
                    alpha_table[k][i]=max_exp(max_log_map_en,max_list)                        
                    #if len(max_list)==0:
                    #    alpha_table[k][i]=-500
                    #else:                
                        #alpha_table[k][i]=max_exp(max_list[0],max_list[1])
                        #alpha_table[k][i]=max_exp(max_list)
        return alpha_table

    #Generalized Beta Table Generation
    def bcjr_beta_table_gen2(h,m,v,max_log_map_en,term_en,fsm_table,gamma_table):
        #beta_table dimension: time index x state index
        beta_table=[['none' for i in range(2**v)] for k in range(h+m+1)]

        for k in range(h+m,0,-1):
            for i in range(2**v):
                #beta table intialization
                if k==h+m:
                    if i==0:  #beta(s0) at t=h+m
                        beta_table[k][i]=0
                    else:
                        beta_table[k][i]=-50 if term_en==True else 0
                
                #beta table generation
                else:
                    max_list=[]
                    for j in range (len(gamma_table[0])):
                        if fsm_table[i][j]!='none':
                            max_list.append(beta_table[k+1][j]+gamma_table[k][i][j])
                        #print(k,i,j)
                        #if gamma_table[k][i][j]!='none':
                        #    max_list.append(beta_table[k+1][j]+gamma_table[k][i][j])
                    #print(f'max list length:{len(max_list)}')                
                    beta_table[k][i]=max_exp(max_log_map_en,max_list)

                    #if len(max_list)==0:
                    #    beta_table[k][i]=-500
                    #else:                
                        #beta_table[k][i]=max_exp(max_list[0],max_list[1])
                        #beta_table[k][i]=max_exp(max_list)
        return beta_table

    n=len(gen_poly) #Code parameter n

    bitpoly=[bitfield(gen_poly[i]) for i in range(n)]
    v=max( len(bitpoly[i]) for i in range(n) )-1 #memory size

    for i in range(n):
        bitpoly[i]=[0] * (v+1-len(bitpoly[i])) + bitpoly[i] if len(bitpoly[i]) < v+1 else bitpoly[i]  #polynomial bitwitdh matching

    if srcc_en==True:
        fsm_table=fsm_table_srcc_gen(v,bitpoly)
    else:
        fsm_table=fsm_table_ff_gen(v,bitpoly)
    
    m=v
    h=len(received_seq)-m    

    gamma_table=bcjr_gamma_table_gen2(La,EsN0,h,m,v,fsm_table,received_seq)
    alpha_table=bcjr_alpha_table_gen2(h,m,v,max_log_map_en,fsm_table,gamma_table)
    beta_table=bcjr_beta_table_gen2(h,m,v,max_log_map_en,term_en,fsm_table,gamma_table)

    ## LLR Generation
    llr_info = np.zeros(h + m)  # LLR for information bits
    llr_parity = np.zeros(h + m)  # LLR for parity bits

    for k in range(h + m):
        positive_list_info = []
        negative_list_info = []
        positive_list_parity = []
        negative_list_parity = []

        for i in range(2**v):
            for j in range(2**v):
                if gamma_table[k][i][j] != 'none':
                    if fsm_table[i][j][0] == 1:  # Positive info bit
                        positive_list_info.append(alpha_table[k][i] + gamma_table[k][i][j] + beta_table[k + 1][j])
                    elif fsm_table[i][j][0] == 0:  # Negative info bit
                        negative_list_info.append(alpha_table[k][i] + gamma_table[k][i][j] + beta_table[k + 1][j])

                    if fsm_table[i][j][2] == 1:  # Positive parity bit
                        positive_list_parity.append(alpha_table[k][i] + gamma_table[k][i][j] + beta_table[k + 1][j])
                    elif fsm_table[i][j][2] == 0:  # Negative parity bit
                        negative_list_parity.append(alpha_table[k][i] + gamma_table[k][i][j] + beta_table[k + 1][j])

        llr_info[k] = max_exp(max_log_map_en, positive_list_info) - max_exp(max_log_map_en, negative_list_info)
        llr_parity[k] = max_exp(max_log_map_en, positive_list_parity) - max_exp(max_log_map_en, negative_list_parity)

    decod_seq = np.where(llr_info > 0.0, 1, 0)

    # Return both LLR arrays
    return llr_info, llr_parity, decod_seq, fsm_table, gamma_table, alpha_table, beta_table

def interleaver(intlv_pattern,in_seq):    
    if len(intlv_pattern)!=len(in_seq):
        print(f'ERROR: interleaver pattern length is not matched with input sequence')
        return None
        
    out_seq=[]
    for i in range(len(in_seq)):        
        out_seq+=[in_seq[intlv_pattern[i]]]
    out_seq=np.array(out_seq)
    return out_seq

def de_interleaver(intlv_pattern,in_seq):    
    if len(intlv_pattern)!=len(in_seq):
        print(f'ERROR: interleaver pattern length is not matched with input sequence')
        return None
    
    de_intlv_pattern=np.zeros(len(intlv_pattern),dtype=int)
    for i,index in enumerate(intlv_pattern):
        de_intlv_pattern[index]=i
    #print(f'de-interleaver pattern:\n{de_intlv_pattern}')

    out_seq=[]
    for i in range(len(in_seq)):               
        out_seq+=[in_seq[de_intlv_pattern[i]]]        
    out_seq=np.array(out_seq)
    return out_seq

def depuncturing(received_seq,code_block_len,num_pccc,punc_mat):
    # code_block_len: the length of code block
    # ex) n,k,m convolutional code, if k is 1000 and m is 5, and R=1/3
    # then code_block_len is k+m=1005 and total code length is (k+m)*(1/R)
    # num_pccc: the number of constituent codes

    k=code_block_len*(num_pccc+1) 
    #total code length without puncturing if each constituent code has R=1/2

    punc_vec=punc_mat.flatten()
    depunc_out=np.array([])

    j=0
    for i in range(k):
        if punc_vec[i%len(punc_vec)]==True:
            depunc_out=np.append(depunc_out,received_seq[j])
            j+=1
        else:
            depunc_out=np.append(depunc_out, 0.0)
    
    return depunc_out

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
print("\n======     Course Project Code Review Input 2    ======\n")
H = [
    np.array([[-0.9 + 2j, 0.2 + 2j],
              [-0.3 - 1j, 2.5 + 0.5j]]),

    np.array([[0 + 1j, -0.1 + 1j],
              [0.5 + 1j, 0 + 0.5j]]),

    np.array([[-0.9 + 0.7j, 1 - 0.2j],
              [0.5 + 0.5j, 0.3 + 0.2j]]),

    np.array([[-0.6 - 0.5j, -0.8 + 1j],
              [0.6 - 2j, -0.7 + 0.2j]])
]

Y = np.array([
    [1 + 1.5j, 2 - 0.2j],
    [-2 + 1j, -0.6 + 1.4j],
    [0.6 - 0.2j, 1.3 - 0.5j],
    [-1 + 0.8j, -1 - 0.4j]
])


Ld = np.zeros(Y.size*2, dtype=float)
Le = np.zeros(Y.size*2, dtype=float)
interleavedForMIMO = np.zeros(Y.size*2, dtype=float)

for outer_iter in range(1):  # Outer loop iterates 2 times
    print(f"Outer iteration {outer_iter + 1}\n")
    for i in range(len(H)):
        matrix = H[i]
        Recd = Y[i].reshape(-1, 1)
        Ld[4*i:4*i+4], Le[4*i:4*i+4] = compute_ld_le(matrix, Recd, interleavedForMIMO[4*i:4*i+4])

    #print("Ld", Ld)

    channel_interleaver_pattern = np.array([3, 8, 14, 1, 5, 4, 10, 9, 11, 16, 15, 12, 13, 6, 7, 2])
    channel_interleaver_pattern = channel_interleaver_pattern - 1
    deinterleavedLd = de_interleaver(channel_interleaver_pattern, Ld)
    deinterleavedLe = de_interleaver(channel_interleaver_pattern, Le)

    #print("deinterleaved Ld", deinterleavedLd)
    print("MIMO detector output extrinsic LLR(de-interleaved, La2):\n", deinterleavedLe)

    srcc_en=True
    max_log_map_en=False
    dec1_term_en=True
    dec2_term_en=False
    gen_poly=[0o7,0o5]
    intlv_pattern=np.array([2,  1,  7, 5,  3,  6,  8,  4])
    intlv_pattern=intlv_pattern-1
    punc_matrix=np.array([[True, True],[True, False],[False,True]]).T
    punc_en=True
    num_pccc=2

    received_seq = deinterleavedLe

    code_block_len=8 #12(info)+2(term)

     #EsN0=EbN0-10log10(k)-10log10(R) in which k bits per symbol and R is code rate
     #EsN0=10**(-1/10) #-1dB, EbN0=2dB, 
    #Eb/N0=2 db (following the definition given in the paper)

    Mc=2 #QPSK
    Mt=2
    Nr=2
    Code_R = 0.5

    Es=2 # in linear
    #Es here is total energy(not average energy per symbol) through Tx antenna
    #Es = (energy per symbol * Mt)
    EbN0=2 #in dB
    sigma2=(Es/2)*(Nr/(Code_R*Mt*Mc))*(10**(-EbN0/10))

    EsN0=EbN0-10*np.log10(Nr/(Code_R*Mt*Mc)) #in dB

    EsN0=np.round(10**(EsN0/10),decimals=4) #-1dB, EbN0=2dB, 
     #'''




    num_iter=2

    if punc_en==True:
        depunc_out=depuncturing(received_seq,code_block_len,num_pccc,punc_matrix)
        received_seq=depunc_out

    Lc= 4*EsN0

    #received_seq_tmp=received_seq.reshape(code_block_len,num_pccc+1)
    received_seq_tmp=received_seq.reshape(code_block_len,-1)

    received_seq1=received_seq_tmp[:,0:2]
    #print(f'received_seq1:\n{received_seq1}')
    intlevd_received_infobit=interleaver(intlv_pattern,received_seq_tmp[:,0])
    #received_seq2=np.concatenate((intlevd_received_infobit.reshape(info_block_len,1),received_seq_tmp[:,2].reshape(info_block_len,1)),axis=1)
    received_seq2=np.concatenate((np.expand_dims(intlevd_received_infobit, axis=1),np.expand_dims(received_seq_tmp[:,2], axis=1)),axis=1)

    for i in range(num_iter):
        print(f'\ninner iteration number:{i+1}\n')

        ## Decoder 1 ##    
        if i==0:
            La=np.zeros(len(received_seq1))
        else:
            La_tmp=ext_llr
            La=de_interleaver(intlv_pattern,La_tmp)
        #print(f'La to decoder1:{La}')
        received_seq_input=received_seq1
        llr,llr_parity, decod_seq,fsm_table,gamma_table, alpha_table, beta_table=BCJR_decoder2(gen_poly,srcc_en,max_log_map_en,dec1_term_en,La,EsN0,received_seq_input)
        #ext_llr=llr-La-Lc*received_seq_tmp[:,0]
        ext_llr=llr-La-received_seq_tmp[:,0]
        Le1Pl1=llr_parity-received_seq_tmp[:,1]  
        #print(f'Extrinsic LLR from decoder1:\n{np.round(ext_llr,decimals=4)}')
        print(f'Extrinsic LLR ul from decoder1:\n{np.round(ext_llr,4)}')

        print(f'Extrinsic LLR pl from decoder1:\n{np.round(Le1Pl1,4)}')
        #print(f'Extrinsic LLR from decoder1:\n{np.round(ext_llr,4)}')
        #print(f'Prior LLR for decoder 1:\n{La}')
        #print(f'Full LLR from decoder 1:\n{llr}')
        #print(f'input to decoder 1:\n{Lc*received_seq_tmp[:,0]}')
        #fsm_gamma_alpha_beta_table_disp(gen_poly,fsm_table,gamma_table,alpha_table,beta_table)

        ## Decoder 2 ##
        La_tmp=ext_llr
        La=interleaver(intlv_pattern,La_tmp)
        
        #print(f'La to decoder2:{La}')
        received_seq_input=received_seq2
        llr,llr_parity, decod_seq,fsm_table,gamma_table, alpha_table, beta_table=BCJR_decoder2(gen_poly,srcc_en,max_log_map_en,dec2_term_en,La,EsN0,received_seq_input)
        #ext_llr=llr-La-Lc*intlevd_received_infobit
        ext_llr=llr-La-intlevd_received_infobit
        ext_llr_ul12 = llr-intlevd_received_infobit
        Le12Ul = de_interleaver(intlv_pattern,ext_llr_ul12)
        Le2Pl2=llr_parity-received_seq_tmp[:,2]
        #print(' Turbo decoder 2 llr(ul):\n', np.round(llr, 4))
        
        #print('Turbo Decoder 2 llr(pl):\n', np.round(llr_parity, 4))
        #print(f'Turbo Decoder 2 ext_llr_ul2:\n{np.round(ext_llr,4)}')
        #print(f'Turbo Decoder 2 ext_llr_ul12:\n{np.round(ext_llr_ul12,4)}')
        #print(f'Turbo Decoder 2 ext_llr_pl2:\n{np.round(Le2Pl2,4)}')
        #print('received_seq_tmp[:,2]',received_seq_tmp[:,2])
        #print(f'ext_LLr_parity from decoder2:\n{ext_llr_parity}')
        #print(f'Extrinsic LLR from decoder2:\n{np.round(ext_llr,4)}')
        #print(f'Prior LLR for decoder 2:\n{La}')
        #print(f'Full LLR from decoder 2:\n{llr}')
        #print(f'input to decoder 2:\n{Lc*intlevd_received_infobit}')
        #fsm_gamma_alpha_beta_table_disp(gen_poly,fsm_table,gamma_table,alpha_table,beta_table)

        siso_out=de_interleaver(intlv_pattern,llr)
        #print(f'full LLR from decoder2:\n{np.round(llr, 4)}')
        #print(f'full LLR from decoder2:\n{np.round(llr,4)}')
        #print(f'full LLR from decoder2:\n{np.round(llr,decimals=4)}')
        #print(f'deinterleaved Soft-output L-values:\n{np.round(siso_out, 4)}')
        #print(f'deinterleaved Soft-output L-values:\n{np.round(siso_out,decimals=4)}')    
        
        #print('Le12Ul:\n', np.round(Le12Ul, 4))
        #print('Le1Pl1:\n', np.round(Le1Pl1, 4))
        #print('Le2Pl2:\n', np.round(Le2Pl2, 4))  
        BeforeInterleaving = puncturing(Le12Ul, Le1Pl1, Le2Pl2, punc_matrix)
        #print(f'Before Interleaving:\n{np.round(BeforeInterleaving, 4)}')
        interleavedForMIMO = interleaver(channel_interleaver_pattern, BeforeInterleaving)
        print(f'Turbo Decoder 2 Extrinsic LLR, after channel interleaving, to be passed into the MIMO detector:\n{np.round(interleavedForMIMO, 4)}')

