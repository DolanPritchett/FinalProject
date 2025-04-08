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
#include <math.h>

/* The gateway function. */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    
	double *abs_diff = mxGetPr(prhs[0]);
	int Len = mxGetM(prhs[0]);
    double *qnum = mxGetPr(prhs[1]);
	int Q = (int)(*qnum);
    int sym_Len = Len/Q;
	double *variance = mxGetPr(prhs[2]);
	double n_0 = *variance;
	int n_bits = log2(Q);

	double *p0 = new double[sym_Len*log2(Q)];
	double *p1 = new double[sym_Len*log2(Q)];
	double *llr= new double[sym_Len*log2(Q)];
	for (int i = 0; i < sym_Len*log2(Q); ++i) p0[i] = p1[i] = 0;
	double p_sym = 0;
	for (int i = 0; i < sym_Len; ++i)
	{
		for (int j = 0; j < Q; ++j)
		{
			p_sym = exp(-pow(abs_diff[i*Q+j], 2) / n_0);
			for (int m_index = 0; m_index < n_bits; ++m_index)
			{
				if ((j&(1 << m_index)) == 0)
					p0[i*n_bits + m_index] = p0[i*n_bits + m_index] + p_sym;
				else
					p1[i*n_bits + m_index] = p1[i*n_bits + m_index] + p_sym;
			}
		}
	}
	for (int i = 0; i < sym_Len*log2(Q); ++i) llr[i] = log(p0[i] / p1[i]);

	delete[]p0;
	delete[]p1;

	

	plhs[0] = mxCreateDoubleMatrix(sym_Len*n_bits, 1, mxREAL);
	double *output_llr = mxGetPr(plhs[0]);
	for (int i = 0; i < sym_Len*n_bits; ++i)
		output_llr[i] = llr[i];

	delete[]llr;
}
