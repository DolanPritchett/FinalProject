%{
mex qam_demod_To_symLLR_CAPI.cpp
mex qam_demod_CAPI.cpp
mex ldpcDecoder_CAPI.cpp bLDPC.cpp function4Binary.cpp
mex ldpcEncoder_CAPI.cpp bLDPC.cpp function4Binary.cpp
mex ldpc_mex_initial_CAPI.cpp bLDPC.cpp function4Binary.cpp
mex clear_ldpc_CAPI.cpp bLDPC.cpp function4Binary.cpp
%}

mex nbldpcDecoder_CAPI_sym.cpp nbLDPC.cpp GFq.cpp gf.cpp function.cpp
mex nbldpcDecoder_CAPI.cpp nbLDPC.cpp GFq.cpp gf.cpp function.cpp
mex nbldpcEncoder_CAPI.cpp nbLDPC.cpp GFq.cpp gf.cpp function.cpp
mex nbldpc_mex_initial_CAPI.cpp nbLDPC.cpp GFq.cpp gf.cpp function.cpp
mex clear_nbldpc_CAPI.cpp nbLDPC.cpp GFq.cpp gf.cpp function.cpp
mex bitLLR_symLLR.cpp nbLDPC.cpp GFq.cpp gf.cpp function.cpp
mex symLLR_bitLLR.cpp nbLDPC.cpp GFq.cpp gf.cpp function.cpp