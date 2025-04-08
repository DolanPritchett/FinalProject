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
     
     double* doubleMesg = mxGetPr(prhs[1]);
     int mesgLen = mxGetM(prhs[1]);
     int codeLen = (*nbldpc_ptr).codeLen;
     int *mesg = new int[mesgLen];
     int *code = new int[codeLen];
     for(int i = 0; i < mesgLen; ++i)
     {
         mesg[i] = (int)doubleMesg[i];
//         printf("%d ",mesg[i]);
     }
//     printf("\n");
//     printf("$$$$$$$$$$$$$$$\n");
     (*nbldpc_ptr).Encoder(mesg, code);
//     for(int i = 0; i < codeLen; ++i)
//         if(code[i] > 1)
//            printf("%d ",code[i]);
//     printf("\n");
//     printf("%d\n",GF.size());
     plhs[0] = mxCreateDoubleMatrix(codeLen, 1, mxREAL);
     double *output_p = mxGetPr(plhs[0]);
     for(int i = 0; i < codeLen; ++i)
         output_p[i] = (double)code[i];
     
     GF.clear(); GFindex.clear(); alpha_to.clear(); index_of.clear();
     delete []mesg;
     delete []code;
}
