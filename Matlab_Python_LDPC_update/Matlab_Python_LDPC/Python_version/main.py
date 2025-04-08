import numpy as np
import h5py
import os
import sys
import scipy
import copy
import numpy.matlib
from numpy import vectorize, arange
import pybind_11_ldpc_setup as pbe


file_name = "./LDPC_CODE/5G_LDPC_M10_N20_Z128_Q2_nonVer.txt"
LDPC_CHKLEN = 1280
LDPC_CODELEN = 2560

address = pbe.ldpc_initial(LDPC_CHKLEN, LDPC_CODELEN, file_name)
LDPC_INFOLEN = pbe.get_ldpc_infoLen(address)
print("LDPC_CHKLEN=", LDPC_CHKLEN, "LDPC_CODELEN=", LDPC_CODELEN, "LDPC_INFOLEN=", LDPC_INFOLEN)


SNR_dB = np.array([0.01, 0.05, 0.1, 0.5, 1, 2, 4])
min_wec = 100
for i in range(len(SNR_dB)):
    wec = 0  # word error number
    bec = 0  # bit error number
    bec_mesg = 0  # message bit error number
    tot = 0  # total block number
    while(wec < min_wec and tot < 10000):
        # generated message
        info_len = LDPC_INFOLEN
        mesg = np.random.randint(0, 2, info_len)
        code = pbe.ldpc_encoder(address, mesg, LDPC_INFOLEN, LDPC_CODELEN)
        code = np.array(code)
        mod_code = 1 - code*2
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