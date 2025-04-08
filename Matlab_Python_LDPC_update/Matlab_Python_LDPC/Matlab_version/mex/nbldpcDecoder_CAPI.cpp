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
     
     double *LeBits = mxGetPr(prhs[1]);
     int codeLen_bit = mxGetM(prhs[1]);
     int bitsGF = (int)log2((double)(*nbldpc_ptr).qnumber);
     int infoLen_bit = (*nbldpc_ptr).infoLen*bitsGF;
     
	 /*
     ofstream os;
     os.open("LeBits.txt");
     for(int i = 0; i < codeLen_bit; ++i)
     {
         os << LeBits[i] << " ";
     }
     os.close();
	 */
     
     double **inLLR = new double*[2];
     for(int i = 0; i < 2; ++i)
         inLLR[i] = new double[codeLen_bit];
     double **outLLR = new double*[2];
     for(int i = 0; i < 2; ++i)
         outLLR[i] = new double[codeLen_bit];
     double **outLLR_u = new double*[2];
     for(int i = 0; i < 2; ++i)
         outLLR_u[i] = new double[infoLen_bit];
     double **inLLRGF = new double*[1<<bitsGF];
     for(int i = 0; i < (1<<bitsGF); ++i)
         inLLRGF[i] = new double[codeLen_bit/bitsGF];
     double **outLLRGF = new double*[1<<bitsGF];
     for(int i = 0; i < (1<<bitsGF); ++i)
         outLLRGF[i] = new double[codeLen_bit/bitsGF];
     double **outLLRGF_u = new double*[1<<bitsGF];
     for(int i = 0; i < (1<<bitsGF); ++i)
         outLLRGF_u[i] = new double[infoLen_bit/bitsGF];
     int *vhat = new int[codeLen_bit/bitsGF];
     
     for(int i = 0; i < codeLen_bit; ++i)
     {
         inLLR[0][i] = 0;
         inLLR[1][i] = LeBits[i];
     }
     

     
     for(int i = 0; i < (codeLen_bit/bitsGF); ++i)
     {
         for(int idx = 0; idx < (1<<bitsGF); ++idx)
         {
             double GFLLR = 0;
             for(int j = 0; j < bitsGF; ++j)
             {
                 if((idx & (1<<j))==0)
                     GFLLR += inLLR[0][i*bitsGF+j];
                 else
                     GFLLR += inLLR[1][i*bitsGF+j];
             }
             inLLRGF[idx][i] = GFLLR;
         }
         for(int idx = 1; idx < (1<<bitsGF); ++idx)
             inLLRGF[idx][i] = inLLRGF[idx][i] - inLLRGF[0][i];
         inLLRGF[0][i] = 0;
     }
     
     /*
     ofstream out;
     out.open("inLLRGF.txt");
     for(int i = 0; i < (codeLen_bit/bitsGF); ++i)
     {
         for(int idx = 0; idx < (1<<bitsGF); ++idx)
         {
             out << inLLRGF[idx][i] << " ";
         }
         out << endl;
     }
     out.close();
     */
     /*
     if(bitsGF==1)
     {
         for (int i = 0; i < (codeLen_bit / bitsGF); ++i)
         {
             for (int idx = 0; idx < (1 << bitsGF); ++idx)
             {
                 if (inLLRGF[idx][i] > 30.0) inLLRGF[idx][i] = 30.0;
                 if (inLLRGF[idx][i] < -30.0) inLLRGF[idx][i] = -30.0;
             }
         }
     }
     */
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
     
     
     for (int i = 0; i < codeLen_bit; ++i)
     {
        int bitLocGF = i % bitsGF;
        int symLocGF = (i - (i%bitsGF)) / bitsGF;
        int num = 0;
        double LLR0 = 0;
        double LLR1 = 0;
        double LLR = 0;
        /*
        for (int idx = 0; idx < (1 << bitsGF); ++idx)
        {
            if ((idx&(1 << bitLocGF)) != 0)
            {
                num++;
                LLR1 = outLLRGF[idx][symLocGF];
                LLR0 = outLLRGF[idx ^ (1 << bitLocGF)][symLocGF];
                LLR += LLR1 - LLR0;
            }
        }
        LLR /= num;
        */
        LLR1 = LLR0 = 0;
        for (int idx = 0; idx < (1 << bitsGF); ++idx)
        {
            if ((idx&(1 << bitLocGF)) != 0)
            {
                LLR1 += exp(outLLRGF[idx][symLocGF]);
                LLR0 += exp(outLLRGF[idx ^ (1 << bitLocGF)][symLocGF]);
            }
        }
        LLR = log(LLR1/LLR0);
        outLLR[1][i] = LLR;
        outLLR[0][i] = 0;
     }
     
     plhs[0] = mxCreateDoubleMatrix(1, codeLen_bit, mxREAL);
     double *output_APP_c = mxGetPr(plhs[0]);
     //for(int i = 0; i < codeLen_bit; ++i)
     //    output_APP_c[i] = 0 - outLLR[1][i];
     for(int i = 0; i < codeLen_bit; ++i)
         output_APP_c[i] = outLLR[1][i];
     
	 /*
     os.open("output_APP_c.txt");
     for(int i = 0; i < codeLen_bit; ++i)
     {
         os << output_APP_c[i] << " ";
     }
     os.close();
	 */
     
     
     /*
     plhs[1] = mxCreateDoubleMatrix(1, infoLen_bit, mxREAL);
     double *output_APP_u = mxGetPr(plhs[1]);
     (*nbldpc_ptr).extract_mesg(outLLRGF_u, outLLRGF);
     for (int i = 0; i < infoLen_bit; ++i)
     {
        int bitLocGF = i % bitsGF;
        int symLocGF = (i - (i%bitsGF)) / bitsGF;
        int num = 0;
        double LLR0 = 0;
        double LLR1 = 1;
        double LLR = 0;
        
        LLR1 = LLR0 = 0;
        for (int idx = 0; idx < (1 << bitsGF); ++idx)
        {
            if ((idx&(1 << bitLocGF)) != 0)
            {
                LLR1 += exp(outLLRGF_u[idx][symLocGF]);
                LLR0 += exp(outLLRGF_u[idx ^ (1 << bitLocGF)][symLocGF]);
            }
        }
        LLR = log(LLR1/LLR0);
        outLLR_u[1][i] = LLR;
        outLLR_u[0][i] = 0;
     }
     
     
     
     
     for(int i = 0; i < infoLen_bit; ++i)
         output_APP_u[i] = outLLR_u[1][i];
     */
     
     GF.clear(); GFindex.clear(); alpha_to.clear(); index_of.clear();
     for(int i = 0; i < 2; ++i)
         delete[]inLLR[i];
     delete[]inLLR;
     for(int i = 0; i < 2; ++i)
         delete[]outLLR[i];
     delete[]outLLR;
     for(int i = 0; i < 2; ++i)
         delete[]outLLR_u[i];
     delete[]outLLR_u;
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
