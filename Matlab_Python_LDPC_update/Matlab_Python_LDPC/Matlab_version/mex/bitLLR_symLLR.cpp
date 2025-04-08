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
    
	double *LeBit = mxGetPr(prhs[0]);
	int codeLen_bit = mxGetM(prhs[0]);
    double *qnumber = mxGetPr(prhs[1]);
	int Q = *qnumber;
	int Q_bits = (int)log2(Q);
	int Len_sym = codeLen_bit / Q_bits;

	double *LeBit_input = new double[Len_sym*Q_bits];
	for (int i = 0; i < Len_sym*Q_bits; ++i)
		LeBit_input[i] = LeBit[i];

	double *LeSym = new double[Len_sym*Q];
	for (int i = 0; i < Len_sym; ++i)
	{
		for (int j = 0; j < Q; ++j)
		{
			double LLR = 0;
			for (int k = 0; k < Q_bits; ++k)
			{
				if ((j&(1 << k)) != 0)
					LLR = LLR + LeBit_input[i*Q_bits + k];
			}
			LeSym[i*Q + j] = LLR;
		}
	}

	plhs[0] = mxCreateDoubleMatrix(Len_sym*Q, 1, mxREAL);
	double *output_LeSym = mxGetPr(plhs[0]);
	for (int i = 0; i < Len_sym*Q; ++i)
		output_LeSym[i] = LeSym[i];
    
	delete[]LeSym;
}
