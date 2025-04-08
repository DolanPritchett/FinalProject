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


vector<gfe> GF;
vector<int> GFindex;
vector<int> alpha_to;
vector<int> index_of;

/* The gateway function. */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
     genGF(GF);
     genGFindex(GFindex);
     
     uint64_t* ptr = static_cast<uint64_t*>(mxGetData(prhs[0]));
     nbLDPC* nbldpc_ptr = reinterpret_cast<nbLDPC*>(*ptr);
     
     double *LeSym = mxGetPr(prhs[1]);
     int Q = (*nbldpc_ptr).qnumber;
     int codeLen_sym = (*nbldpc_ptr).codeLen;
     int bitsGF = (int)log2((double)Q);
     int infoLen_sym= (*nbldpc_ptr).infoLen;
     
     double **inLLRGF = new double*[1<<bitsGF];
     for(int i = 0; i < (1<<bitsGF); ++i)
         inLLRGF[i] = new double[codeLen_sym];
     double **outLLRGF = new double*[1<<bitsGF];
     for(int i = 0; i < (1<<bitsGF); ++i)
         outLLRGF[i] = new double[codeLen_sym];
     double **outLLRGF_u = new double*[1<<bitsGF];
     for(int i = 0; i < (1<<bitsGF); ++i)
         outLLRGF_u[i] = new double[infoLen_sym];
     int *vhat = new int[codeLen_sym];
     
     for(int i = 0; i < codeLen_sym; ++i)
     {
		 for (int j = 0; j < (1 << bitsGF); ++j)
			 inLLRGF[j][i] = LeSym[i*Q+j];
     }
     
     /*
     cout << "**************************" << endl;
     
     ofstream out;
     out.open("inLLRGF.txt");
     for(int i = 0; i < codeLen_sym; ++i)
     {
         for(int idx = 0; idx < (1<<bitsGF); ++idx)
         {
             out << inLLRGF[idx][i] << " ";
         }
         out << endl;
     }
     out.close();
     */
     if(bitsGF==1)
     {
         for (int i = 0; i < codeLen_sym; ++i)
         {
             for (int idx = 0; idx < (1 << bitsGF); ++idx)
             {
                 if (inLLRGF[idx][i] > 30.0) inLLRGF[idx][i] = 30.0;
                 if (inLLRGF[idx][i] < -30.0) inLLRGF[idx][i] = -30.0;
             }
         }
     }
     (*nbldpc_ptr).FFT_QSPA_log(inLLRGF, outLLRGF, vhat, 100);
     
     
     /*
     os.open("vhat.txt");
     for(int i = 0; i < (codeLen_bit/elementBits); ++i)
     {
         os << vhat[i] << " ";
     }
     os.close();
	 */
     /*
     out.open("outLLRGF.txt");
     for(int i = 0; i < (codeLen_bit/bitsGF); ++i)
     {
         for(int idx = 0; idx < (1<<bitsGF); ++idx)
         {
             out << outLLRGF[idx][i] << " ";
         }
         out << endl;
     }
     out.close();
	 */
     
     
     plhs[0] = mxCreateDoubleMatrix(codeLen_sym*Q, 1, mxREAL);
     double *output_APP_c = mxGetPr(plhs[0]);
     //for(int i = 0; i < codeLen_bit; ++i)
     //    output_APP_c[i] = 0 - outLLR[1][i];
     for(int i = 0; i < codeLen_sym; ++i)
		 for(int j = 0; j < (1<<bitsGF); ++j)
			output_APP_c[i*Q+j] = outLLRGF[j][i];
     
	 /*
     os.open("output_APP_c.txt");
     for(int i = 0; i < codeLen_bit; ++i)
     {
         os << output_APP_c[i] << " ";
     }
     os.close();
	 */
     
     
     
     
     
     
     GF.clear(); GFindex.clear(); alpha_to.clear(); index_of.clear();
     
   
     
     for(int i = 0; i < (1<<bitsGF); ++i)
         delete[]inLLRGF[i];
     delete[]inLLRGF;
     for(int i = 0; i < (1<<bitsGF); ++i)
         delete[]outLLRGF[i];
     delete[]outLLRGF;
     for(int i = 0; i < (1<<bitsGF); ++i)
         delete[]outLLRGF_u[i];
     delete[]outLLRGF_u;
     delete[]vhat;
}
