/* file: ConvEncode.c

   Description: Convolutionally encode with either NSC or RSC code.

   The calling syntax is:

      [output] = ConvEncode(input, g_encoder, [code_type] )

      output = code word

      Required inputs:
	  input  = data word
	  g_encoder = generator matrix for convolutional code
	              (If RSC, then feedback polynomial is first)
	  Optional inputs:
	  code_type = 0 for recursive systematic convolutional (RSC) code (default)
	            = 1 for non-systematic convolutional (NSC) code
				= 2 for tail-biting NSC code               

   Copyright (C) 2005-2008, Matthew C. Valenti

   Last updated on May 21, 2008

   Function ConvEncode is part of the Iterative Solutions 
   Coded Modulation Library. The Iterative Solutions Coded Modulation 
   Library is free software; you can redistribute it and/or modify it 
   under the terms of the GNU Lesser General Public License as published 
   by the Free Software Foundation; either version 2.1 of the License, 
   or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

*/
        
#include <math.h>
#include <mex.h>
#include <matrix.h>
#include <stdlib.h>

/* library of functions */
#include "convolutional.h"

/* Input Arguments */
#define INPUT       prhs[0]
#define GENENCODER  prhs[1]
#define CODETYPE    prhs[2]

/* Output Arguments */
#define OUTPUT      plhs[0]

/* main function that interfaces with MATLAB */
void mexFunction(
				 int            nlhs,
				 mxArray       *plhs[],
				 int            nrhs,
				 const mxArray *prhs[] )
{	double	*input, *g_array;
	double	*output_p;
	int      DataLength, CodeLength, i, j, index;
	int      subs[] = {1,1};
	int     *g_encoder;
	int		 nn, KK, mm, code_type, max_states;
	double   elm;
	int		*input_int, *output_int;
	int     *out0, *out1, *state0, *state1, *tail;
	int state = 0;
	int ii;

	code_type = 0; /* Default:Code is RSC with terminated trellis */

	/* Check for proper number of arguments */
	if ((nrhs < 2 )||(nlhs  > 1)) {
		mexErrMsgTxt("Usage: [output] = ConvEncode(input, g_encoder, code_type )");
	} else {
		/* first input is the data word */
		input = mxGetPr(INPUT);	
		DataLength = mxGetN(INPUT); /* number of data bits */
//		mexPrintf("DataLength %d \n", DataLength);

		/* cast the input into a vector of integers */
		input_int = calloc( DataLength, sizeof(int) );
		for (i=0;i<DataLength;i++)
			input_int[i] = (int) input[i];

		/* second input specifies the code polynomial */
	    g_array = mxGetPr(GENENCODER);	
		nn = mxGetM(GENENCODER);
		KK = mxGetN(GENENCODER);
		mm = KK - 1;
		max_states = 1 << mm;

		if ( nrhs == 3 ) {		
			/* optional third input indicates if outer is RSC, NSC or tail-biting NSC */
			code_type   = (int) *mxGetPr(CODETYPE);
		}

		/* Determine the length of the output */
		if (code_type < 2)
			CodeLength = nn*(DataLength+mm);
		else
			CodeLength = nn*DataLength;
			
		/* Convert code polynomial to binary */
		g_encoder = calloc(nn, sizeof(int) );

		for (i = 0;i<nn;i++) {
			subs[0] = i;
			for (j=0;j<KK;j++) {
				subs[1] = j;
				index = mxCalcSingleSubscript(GENENCODER, 2, subs);
				elm = g_array[index];
		//		mexPrintf(" elm = %f \n", elm);
		//		mexPrintf("   g_array[%d] = %f \n", index, g_array[index]); 
				if (elm != 0) {
					g_encoder[i] = g_encoder[i] + (int) pow(2,(KK-j-1)); 
				}
				
			}
			 mexPrintf("   g_encoder[%d] = %o %d %d\n", i, g_encoder[i], nn, KK ); 
		}

	} 

	/* create the output vector */		
	OUTPUT = mxCreateDoubleMatrix(1, CodeLength, mxREAL );
	output_p = mxGetPr(OUTPUT);	
	output_int = calloc( CodeLength, sizeof( int ) );

	/* create appropriate transition matrices */
	out0 = calloc( max_states, sizeof(int) );
	out1 = calloc( max_states, sizeof(int) );
	state0 = calloc( max_states, sizeof(int) );
	state1 = calloc( max_states, sizeof(int) );
	tail = calloc( max_states, sizeof(int) );

	if ( code_type ) {
		nsc_transit( out0, state0, 0, g_encoder, KK, nn );
		nsc_transit( out1, state1, 1, g_encoder, KK, nn );
		if (code_type == 2)
			tail[0] = -1;
	} else {
		rsc_transit( out0, state0, 0, g_encoder, KK, nn );
		rsc_transit( out1, state1, 1, g_encoder, KK, nn );
		rsc_tail( tail, g_encoder, max_states, mm );
	}

	
	if (tail[0] < 0) {
		for (ii = DataLength - KK + 1;ii<DataLength;ii++) {
			if (input_int[ii]) {
				/* Determine next state */
				state = state1[state];
			}
			else {
				/* Determine next state */
				state = state0[state];
			}
		}
	}
	//mexPrintf("state = %d \n", state);
	/* encode data bits one bit at a time */
	mexPrintf("Con1v: %d, %d, %d, %d, %d, %d, %d, %d \n",out1[0], out1[1], out1[2], out1[3], out0[0], out0[1], out0[2], out0[3]);
	
	//int outsym = -1;
	//for (int ii = 0;ii<DataLength;ii++) {
	//	//mexPrintf("state = %d ", state);
	//	if (input_int[ii]) {
	//		/* Input is a one */
	//		outsym = out1[state];  /* The output symbol */

	//							   /* Determine next state */
	//		state = state1[state];
	//	}
	//	else {
	//		/* Input is a zero */
	//		outsym = out0[state];  /* The output symbol */

	//							   /* Determine next state */
	//		state = state0[state];
	//	}

	//	//outsym = -1;

	//	

	//	/* Convert symbol to a binary vector	*/
	//	int *bin_vec;
	//	bin_vec = calloc(nn, sizeof(int));
	//	itob(bin_vec, outsym, nn);
	//	free(bin_vec);

	//	if (ii < 100)
	//		mexPrintf("{input = %d, next state = %d, bit[0] = %d, bit[1] = %d} \n", input_int[ii], state, bin_vec[0], bin_vec[1]);

	//	///* Assign to output */
	//	//for (j = 0;j<nn;j++)
	//	//	output_p[nn*i + j] = bin_vec[j];
	//}
	
	/* Encode */
	conv_encode( output_int, input_int, out0, state0, out1, state1, tail, KK, DataLength, nn );	


	/* cast to output */
    for (i=0;i<CodeLength;i++) 			
		output_p[i] = output_int[i];
//	mexPrintf("CodeLength %d \n", CodeLength);	
	/* Clean up memory */
	free( output_int );
	free( input_int );
	free( g_encoder );
	free( out0 );
	free( out1 );
	free( state0 );
	free( state1 );
	free( tail );

	return;
}
