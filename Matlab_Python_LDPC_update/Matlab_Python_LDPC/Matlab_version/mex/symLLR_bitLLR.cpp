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
#include "nbldpc.h"
#include <math.h>

vector<gfe> GF;
vector<int> GFindex;
vector<int> alpha_to;
vector<int> index_of;

/* The gateway function. */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    
	double *LeSym = mxGetPr(prhs[0]);
	int codeLen_sym = mxGetM(prhs[0]);
    double *qnumber = mxGetPr(prhs[1]);
	int Q = *qnumber;
	int Q_bits = (int)log2(Q);
	int Len_sym = codeLen_sym / Q;

    printf("%2d\n",Len_sym);
	double *LeBit = new double[Len_sym*Q_bits];
	for (int iSym = 0; iSym < Len_sym; ++iSym)
	{
		for (int idx = 0; idx < Q_bits; ++idx)
		{
			double LLR1 = 0; double LLR0 = 0;
			for (int i = 0; i < Q; ++i)
			{
				if ((i&(1 << idx)) != 0)
				{
					LLR1 = LLR1 + exp(LeSym[iSym*Q + i]);
					int ii = i ^ (1 << idx);
					LLR0 = LLR0 + exp(LeSym[iSym*Q + ii]);
				}
			}
			double LLR = log(LLR1 / LLR0);
			LeBit[iSym*Q_bits + idx] = LLR;
		}
	}

	plhs[0] = mxCreateDoubleMatrix(Len_sym*Q_bits, 1, mxREAL);
	double *output_LeBit = mxGetPr(plhs[0]);
	for (int i = 0; i < Len_sym*Q_bits; ++i)
		output_LeBit[i] = LeBit[i];

	delete[]LeBit;
}
