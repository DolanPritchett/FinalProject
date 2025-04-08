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

nbLDPC *nbldpc;

/* The gateway function. */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    
    double *input  = mxGetPr(prhs[0]);
    int checkLen = (int)input[0];
    int codeLen = (int)input[1];
    int q       = (int)input[2];
    /* get the length of the input string */
    
    char *input_buf = mxArrayToString(prhs[1]);
    
    genGF(GF);
    genGFindex(GFindex);
    
    nbldpc = new nbLDPC(checkLen, codeLen, q);
    cout << input_buf << endl;
    (*nbldpc).CreatMatrix_OL_nonbinary(input_buf);
    //(*nbldpc).CreatMatrix_OL_nonbinary_alist(input_buf);
    (*nbldpc).rearrangeXS();
    (*nbldpc).gen_G();
    printf("infoLen and codeLen\n");
    printf("%2d\n",(*nbldpc).infoLen);
    printf("%2d\n",(*nbldpc).codeLen);
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    uint64_t* ptr = static_cast<uint64_t*>(mxGetData(plhs[0]));
    *ptr = reinterpret_cast<uint64_t>(nbldpc);
    
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *infoLen = mxGetPr(plhs[1]);
    *infoLen = (*nbldpc).infoLen;
    
    GF.clear(); GFindex.clear(); alpha_to.clear(); index_of.clear();
}
