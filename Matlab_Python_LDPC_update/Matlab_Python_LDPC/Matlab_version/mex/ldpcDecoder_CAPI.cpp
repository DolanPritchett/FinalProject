/*==========================================================
 * mexcpp.cpp - example in MATLAB External Interfaces
 *
 * Illustrates how to use some C++ language features in a MEX-file.
 *
 * The routine simply defines a class, constructs a simple object,
 * and displays the initial values of the internal variables. It
 * then sets the data members of the object based on the input given
 * to the MEX-file and displays the changed values.
 *
 * This file uses the extension .cpp. Other common C++ extensions such
 * as .C, .cc, and .cxx are also supported.
 *
 * The calling syntax is:
 *
 *              mexcpp( num1, num2 )
 *
 * This is a MEX-file for MATLAB.
 * Copyright 1984-2018 The MathWorks, Inc.
 *
 *========================================================*/

#include <iostream>
#include <fstream>
#include "mex.h"
#include "bLDPC.hpp"


/* The gateway function. */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
     
     uint64_t* ptr = static_cast<uint64_t*>(mxGetData(prhs[0]));
     qcldpc* ldpc_ptr = reinterpret_cast<qcldpc*>(*ptr);
     
     double *LeBits = mxGetPr(prhs[1]);
     int codeLen_bit = mxGetM(prhs[1]);
     int bitsGF = 1;
     int infoLen_bit = (*ldpc_ptr).infoLen*bitsGF;
     
	 double *inLLR = new double[codeLen_bit];
	 double *outLLR = new double[codeLen_bit];
	 int *vhat = new int[codeLen_bit];
     for(int i = 0; i < codeLen_bit; ++i) inLLR[i] = LeBits[i];
	 (*ldpc_ptr).log_Decoder(vhat, inLLR, outLLR, 100);
     
     
     plhs[0] = mxCreateDoubleMatrix(codeLen_bit, 1, mxREAL);
     double *output_APP_c = mxGetPr(plhs[0]);
     //for(int i = 0; i < codeLen_bit; ++i)
     //    output_APP_c[i] = 0 - outLLR[1][i];
     for(int i = 0; i < codeLen_bit; ++i)
         output_APP_c[i] = outLLR[i];
     
     
	 delete[]inLLR;
	 delete[]outLLR;
     delete[]vhat;
}
