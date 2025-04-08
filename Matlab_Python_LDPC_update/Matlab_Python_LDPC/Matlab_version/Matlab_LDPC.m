addpath './mex'
ldpc_checkLen = 128*10;
ldpc_codeLen = 128*20;
ldpc_qnumber = 2.0;
code_name = './forClass/5G_LDPC_M10_N20_Z128_Q2_nonVer.txt';
% Initialization of parameters and H matrix
[address,ldpc_infoLen] = ldpc_mex_initial_CAPI([ldpc_checkLen ldpc_codeLen ldpc_qnumber], code_name);

SNR_dB = [0.01 0.05 0.1 0.5 1 2 4];
min_wec = 100;
for i=1:length(SNR_dB)
    wec = 0; % word error number
    bec = 0; % bit error number
    bec_mesg = 0; % mesg bit error number
    tot = 0; % total block number
    while(wec<min_wec && tot < 10000)
        % generated message
        info_len = ldpc_infoLen;
        mesg = double(rand(info_len,1)>0.5);
        code = ldpcEncoder_CAPI(address, mesg);
        mod_code = 1 - code*2;
        noise_var = 10^(-SNR_dB(i)/10);
        r = mod_code + sqrt(noise_var)*randn(size(mod_code));
        prior = 2*r/noise_var;
        out_APP_c = ldpcDecoder_CAPI(address, prior);
        hard_code = (out_APP_c<0);
        out_APP_m = ldpc_Ext_CAPI(address, out_APP_c);
        hard_mesg = (out_APP_m<0);
        bit_mesg_err = sum(abs(hard_mesg-mesg));
        bit_err = sum(abs(hard_code-code));
        bec = bec + bit_err;
        bec_mesg = bec_mesg + bit_mesg_err;
        if bit_mesg_err ~= 0
            wec = wec + 1;
        end
        tot = tot + 1;
    end
    ber = bec / tot / length(code);
    ber_mesg = bec_mesg / tot / length(mesg);
    wer = wec / tot;
    fprintf('SNR(dB) = %f, tot = %f, ber = %f, ber_mesg = %f, wer = %f\n', SNR_dB(i), tot, ber, ber_mesg, wer);
end
