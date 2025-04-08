#include "nbLDPC.h"
#include "function.h"
#include <fstream>
using namespace std;
#include "GFq.h"

bool shouldput;
ofstream osError;

ofstream out;


double maxLLR0 = 0, minLLR0 = 0;
int where;
void mFFT(double* input, int Qary)
{
	int m, j, k, num;
	int index1, index2;
	unsigned int mask;
	double tmp1;

	//	printf("In fft\n");

	where = 1;

	k = Qary;
	m = 0;
	while (k>1)
	{
		k = k >> 1;
		m++;
	}

	num = Qary / 2;

	for (j = 0;j<Qary;j++)
	{
		//		printf("input[%d]=%12.8f\n", j, input[j]);
	}

	for (j = 0;j<m;j++)
	{
		for (k = 0;k<num;k++)
		{
			mask = (0xff) << (j + 1);
			index1 = ((k << 1)&mask) + (k % (1 << j));
			index2 = index1 + (1 << j);

			//			printf("FFT j = %d j1 %d k =%d mask %d, index1 %d -index2 %d\n", j, k%(1<<j), k, mask, index1, index2);
			tmp1 = input[index1] + input[index2];
			input[index2] = input[index1] - input[index2];
			input[index1] = tmp1;

			/*			printf("j=%d k=%d, index1 %d=%18.14f   index2 %d=%18.14f\n",
			j, k, index1, input[index1], index2, input[index2]);*/
		}
	}
	where = 0;
}

void mIFFT(double* input, int Qary)
{
	int m, j, k, num;
	int index1, index2;
	int mask;
	int sgn;
	double tmp1;

	int l; 	//test only
	FILE *f;

	//	printf("In Ifft\n");

	where = 2;

	k = Qary;
	m = 0;
	while (k>1)
	{
		k = k >> 1;
		m++;
	}

	num = Qary / 2;

	/*	for(j=0;j<Qary;j++)
	{
	if(insgn[j]!=1&&insgn[j]!=-1)
	printf("Wrong input sign in ifft\n");
	printf("input[%d]=%12.8f\n", j, input[j]);
	}*/

	for (j = 0;j<m;j++)
	{
		for (k = 0;k<num;k++)
		{
			mask = (0xff) << (j + 1);
			index1 = ((k << 1)&mask) + (k % (1 << j));
			index2 = index1 + (1 << j);

			tmp1 = input[index1] + input[index2];
			input[index2] = (input[index1] - input[index2]) / 2;
			input[index1] = tmp1 / 2;


			/*			printf("j=%d k=%d, index1 %d=%18.14f index2 %d=%18.4f\n",
			j, k, index1, input[index1], index2,
			input[index2]);*/

		}
	}
	//	printf("IFFT\n");
	for (j = 0;j<Qary;j++)		//just for debug
	{
		//		input[j] = input[j]/4;
		if (input[j]<0)
		{
			printf("There are something wrong with FFT %d =%18.16e \n", j, input[j]);

			input[j] = 0;
			/*			f = fopen("ffttest.fft", "a");	//for test only
			for(k=0;k<max_chk_deg;k++)
			{
			for(l=0;l<16;l++)
			fprintf(f, "%18.14f, ", testLLR[k][l]);
			fprintf(f, "\n");
			}
			fclose(f);

			exit(1);*/

		}
	}

	where = 0;

}

#define MINVALUE2 -1.0e-5
double mylog2;

extern int max_chk_deg;

//for test only 
//extern double testLLR[10][256];



void LGaddminus(double* value1, int* sgn1, double* value2, int* sgn2)
{
	int tsgn1, tsgn2;
	double tvalue1, tvalue2, dtmp, dtmp0;
	double minvalue;

	minvalue = 1.0e-30;

	tsgn1 = (*sgn1);
	tsgn2 = (*sgn2);

	tvalue1 = (*value1);
	tvalue2 = (*value2);

	if (tsgn1 == tsgn2)
	{
		if (tvalue1 == tvalue2)		//Peng 011206
		{
			*sgn2 = 1;
			*value2 = -200;
			*sgn1 = tsgn1;
			*value1 = tvalue1 + mylog2;
		}
		else if (tvalue1>tvalue2)
		{
			*sgn1 = tsgn1;
			*sgn2 = tsgn1;
			dtmp0 = tvalue2 - tvalue1;
			if (dtmp0)
				dtmp = exp(dtmp0);
			//			if(dtmp0>-30)
			{
				*value1 = tvalue1 + log(1.0 + dtmp);
				if (dtmp0>MINVALUE2)	//030706
					dtmp0 = 1.0 - exp(MINVALUE2);
				else
					dtmp0 = 1.0 - dtmp;
				if (dtmp0<minvalue)
				{
					if (where == 1)
						printf("In FFT\n");
					if (where == 2)
						printf("In IFFT\n");

					printf("small value, %16.12f %16.12f %16.12f\n", tvalue1, tvalue2, tvalue1 + log(dtmp0));
					*value2 = -200;
				}
				else
					*value2 = tvalue1 + log(dtmp0);
			}
			/*			else
			{
			*value1 = tvalue1 + log1p(dtmp);
			*value2 = tvalue1 + log1p(-dtmp);

			}*/
		}
		else
		{
			*sgn1 = tsgn1;
			*sgn2 = -tsgn2;
			dtmp0 = tvalue1 - tvalue2;
			dtmp = exp(dtmp0);
			//			if(dtmp0>-30)
			{
				*value1 = tvalue2 + log(1.0 + dtmp);
				if (dtmp0>MINVALUE2)
					dtmp0 = 1.0 - exp(MINVALUE2);
				else
					dtmp0 = 1.0 - dtmp;
				if (dtmp0<minvalue)
				{
					*value2 = -200;

					if (where == 1)
						printf("In FFT\n");
					if (where == 2)
						printf("In IFFT\n");

					printf("small value, %16.12f %16.12f %16.12f\n", tvalue1, tvalue2, tvalue2 + log(dtmp0));
				}
				else
					*value2 = tvalue2 + log(dtmp0);
			}
			/*			else
			{
			*value1 = tvalue2 + log1p(dtmp);
			*value2 = tvalue2 + log1p(-dtmp);

			}*/
		}
	}
	else
	{
		if (tvalue1 == tvalue2)		//Peng 011206
		{
			*sgn1 = 1;
			*value1 = -200;
			*sgn2 = tsgn1;
			*value2 = tvalue1 + mylog2;
		}
		else if (tvalue1>tvalue2)
		{
			*sgn1 = tsgn1;
			*sgn2 = tsgn1;
			dtmp0 = tvalue2 - tvalue1;
			dtmp = exp(dtmp0);


			//			if(dtmp0>-30)
			{
				if (dtmp0>MINVALUE2)
					dtmp0 = 1.0 - exp(MINVALUE2);
				else
					dtmp0 = 1.0 - dtmp;

				if (dtmp0<minvalue)
				{
					if (where == 1)
						printf("In FFT\n");
					if (where == 2)
						printf("In IFFT\n");

					printf("small value, %16.12f %16.12f %16.12f\n", tvalue1, tvalue2, tvalue1 + log(dtmp0));
					*value1 = -200;
				}
				else
					*value1 = tvalue1 + log(dtmp0);
				*value2 = tvalue1 + log(1.0 + dtmp);
			}
			/*			else
			{
			*value1 = tvalue1 + log1p(-dtmp);
			*value2 = tvalue1 + log1p(dtmp);

			}*/
		}
		else
		{
			*sgn1 = tsgn2;
			*sgn2 = tsgn1;
			dtmp0 = tvalue1 - tvalue2;
			dtmp = exp(dtmp0);
			//			if(dtmp0>-30)
			{
				if (dtmp0>MINVALUE2)
					dtmp0 = 1.0 - exp(MINVALUE2);
				else
					dtmp0 = 1.0 - dtmp;
				if (dtmp0<minvalue)
				{
					if (where == 1)
						printf("In FFT\n");
					if (where == 2)
						printf("In IFFT\n");

					printf("small value, %16.12f %16.12f %16.12f\n", tvalue1, tvalue2, tvalue2 + log(dtmp0));
					*value1 = -200;
				}
				else
					*value1 = tvalue2 + log(dtmp0);
				*value2 = tvalue2 + log(1.0 + dtmp);
			}
			/*			else
			{
			*value1 = tvalue2 + log1p(-dtmp);
			*value2 = tvalue2 + log1p(dtmp);

			}*/

		}
	}

}

void LG2mFFT(double* input, int* outsgn, int Qary)
{
	int m, j, k, num;
	int index1, index2;
	unsigned int mask;
	int sgn;
	double tmp1;

	//	printf("In LG fft\n");

	where = 1;

	k = Qary;
	m = 0;
	while (k>1)
	{
		k = k >> 1;
		m++;
	}

	num = Qary / 2;

	for (j = 0;j<Qary;j++)
	{
		outsgn[j] = 1;
		//		printf("input[%d]=%12.8f\n", j, input[j]);
	}

	for (j = 0;j<m;j++)
	{
		for (k = 0;k<num;k++)
		{
			mask = (0xff) << (j + 1);
			index1 = ((k << 1)&mask) + (k % (1 << j));
			index2 = index1 + (1 << j);

			//			printf("FFT j = %d j1 %d k =%d mask %d, index1 %d -index2 %d\n", j, k%(1<<j), k, mask, index1, index2);
			LGaddminus(input + index1, outsgn + index1,
				input + index2, outsgn + index2);

			/*			LGadd(&tmp1, &sgn, input[index1], outsgn[index1],
			input[index2], outsgn[index2]);

			LGminus(input+index2, outsgn+index2, input[index1], outsgn[index1],
			input[index2], outsgn[index2]);
			input[index1] = tmp1;
			outsgn[index1] = sgn;*/

			/*			printf("j=%d k=%d, index1 %d=%18.14f %d  Pr %18.15f index2 %d=%18.14f %d Pr %18.15f\n",
			j, k, index1, input[index1], outsgn[index1], outsgn[index1]*exp(input[index1]), index2,
			input[index2], outsgn[index2],  outsgn[index2]*exp(input[index2]));*/
		}
	}

	where = 0;
}


void LG2mIFFT(double* input, int* insgn, int Qary)
{
	int m, j, k, num;
	int index1, index2;
	int mask;
	int sgn;
	double tmp1;

	int l; 	//test only
	FILE *f;

	//	printf("In LG Ifft\n");

	where = 2;

	k = Qary;
	m = 0;
	while (k>1)
	{
		k = k >> 1;
		m++;
	}

	num = Qary / 2;

	/*	for(j=0;j<Qary;j++)
	{
	if(insgn[j]!=1&&insgn[j]!=-1)
	printf("Wrong input sign in ifft\n");
	printf("input[%d]=%12.8f\n", j, input[j]);
	}*/

	for (j = 0;j<m;j++)
	{
		for (k = 0;k<num;k++)
		{
			mask = (0xff) << (j + 1);
			index1 = ((k << 1)&mask) + (k % (1 << j));
			index2 = index1 + (1 << j);

			LGaddminus(input + index1, insgn + index1,
				input + index2, insgn + index2);
			input[index1] = input[index1] - mylog2;
			input[index2] = input[index2] - mylog2;

			/*			LGadd(&tmp1, &sgn, input[index1], insgn[index1],
			input[index2], insgn[index2]);
			//			printf("IFFT minus %d - %d\n", index1, index2);
			LGminus(input+index2, insgn+index2, input[index1], insgn[index1],
			input[index2], insgn[index2]);
			input[index1] = tmp1-log(2);
			insgn[index1] = sgn;
			input[index2] = input[index2]-log(2);*/

			/*		printf("j=%d k=%d, index1 %d=%18.14f %d  Pr %18.15f index2 %d=%18.14f %d Pr %18.15f\n",
			j, k, index1, input[index1], insgn[index1], insgn[index1]*exp(input[index1]), index2,
			input[index2], insgn[index2],  insgn[index2]*exp(input[index2]));*/

		}
	}
	//	printf("IFFT\n");
	for (j = 0;j<Qary;j++)		//just for debug
	{
		//		input[j] = input[j]/4;
		if (insgn[j]<0)
		{
			printf("There are something wrong with FFT %d =%f \n", j, input[j]);

			input[j] = -100;
			/*			f = fopen("ffttest.fft", "a");	//for test only
			for(k=0;k<max_chk_deg;k++)
			{
			for(l=0;l<16;l++)
			fprintf(f, "%18.14f, ", testLLR[k][l]);
			fprintf(f, "\n");
			}
			fclose(f);

			exit(1);*/

		}
	}
	where = 0;

}





void LGadd(double *outvalue, int *outsgn, double invalue1, int insgn1,
	double invalue2, int insgn2)
{
	int sgn;

	/*	if((insgn1!=1)&&(insgn1!=-1))
	printf("Wrong input sign in add\n");
	if((insgn2!=1)&&(insgn2!=-1))
	printf("Wrong input sign in add\n");*/

	if (insgn1 != insgn2&&invalue1 == invalue2)		// equal to 0
	{
		*outsgn = 1;
		*outvalue = -100;
		//		printf("FFT 0 %f %f \n", invalue1, invalue2 );
	}
	else
	{
		if (((insgn1 == 1) && (insgn2 == 1)) || ((insgn1 == 1) && (insgn2 == -1)
			&& (invalue1>invalue2)) || ((insgn1 == -1) && (insgn2 == 1)
				&& (invalue1<invalue2)))
			sgn = 1;
		else
			sgn = -1;

		if (insgn1 == insgn2)
		{
			if (invalue1>invalue2)
			{
				*outvalue = invalue1 + log(1 + exp(invalue2 - invalue1));
			}
			else
			{
				*outvalue = invalue2 + log(1 + exp(invalue1 - invalue2));
			}
		}
		else
		{
			if (invalue1>invalue2)
			{
				*outvalue = invalue1 + log(1 - exp(invalue2 - invalue1));
			}
			else
			{
				*outvalue = invalue2 + log(1 - exp(invalue1 - invalue2));
			}
		}

		*outsgn = sgn;
	}
}


void LGminus(double *outvalue, int *outsgn, double invalue1, int insgn1,
	double invalue2, int insgn2)
{
	int sgn;

	/*	if((insgn1!=1)&&(insgn1!=-1))
	printf("Wrong input sign in minus\n");
	if((insgn2!=1)&&(insgn2!=-1))
	printf("Wrong input sign in minus\n");*/

	if (insgn1 == insgn2&&invalue1 == invalue2)		// equal to 0
	{
		*outsgn = 1;
		*outvalue = -100;
		//		printf("-FFT 0 %f %f \n", invalue1, invalue2 );
	}
	else
	{
		if (((insgn1 == 1) && (insgn2 == -1)) || ((insgn1 == 1) && (insgn2 == 1)
			&& (invalue1>invalue2)) || ((insgn1 == -1) && (insgn2 == -1)
				&& (invalue1<invalue2)))
			sgn = 1;
		else
			sgn = -1;

		if (insgn1 == insgn2)
		{
			if (invalue1>invalue2)
			{
				*outvalue = invalue1 + log(1 - exp(invalue2 - invalue1));
			}
			else
			{
				*outvalue = invalue2 + log(1 - exp(invalue1 - invalue2));
			}
		}
		else
		{
			if (invalue1>invalue2)
			{
				*outvalue = invalue1 + log(1 + exp(invalue2 - invalue1));
			}
			else
			{
				*outvalue = invalue2 + log(1 + exp(invalue1 - invalue2));
			}
		}

		*outsgn = sgn;
	}
}


void LGmult(double *outvalue, int *outsgn, double invalue1, int insgn1,
	double invalue2, int insgn2)
{
	/*	if((insgn1!=1)&&(insgn1!=-1))
	printf("Wrong input sign in mult\n");
	if((insgn2!=1)&&(insgn2!=-1))
	printf("Wrong input sign in mult\n");*/

	*outvalue = invalue1 + invalue2;
	*outsgn = insgn1*insgn2;
}


void LGdiv(double *outvalue, int *outsgn, double invalue1, int insgn1,
	double invalue2, int insgn2)
{
	/*	if((insgn1!=1)&&(insgn1!=-1))
	printf("Wrong input sign in div\n");
	if((insgn2!=1)&&(insgn2!=-1))
	printf("Wrong input sign in div\n");*/

	*outvalue = invalue1 - invalue2;
	*outsgn = insgn1*insgn2;
}

#if(GFDNUM==2)
int w[2][2] = { 1,1,1,-1 };
#endif

#if(GFDNUM==4)
int w[4][4] = {
1, 1, 1, 1,
1, -1, 1, -1, 
1, 1, -1, -1, 
1, -1, -1, 1};
#endif

#if(GFDNUM==8)
int w[8][8] = {
	1, 1, 1, 1, 1, 1, 1, 1,
	1,-1, 1,-1, 1,-1, 1,-1,
	1, 1,-1,-1, 1, 1,-1,-1,
	1,-1,-1, 1, 1,-1,-1, 1,
	1, 1, 1, 1,-1,-1,-1,-1,
	1,-1, 1,-1,-1, 1,-1, 1,
	1, 1,-1,-1,-1,-1, 1, 1,
	1,-1,-1, 1,-1, 1, 1,-1
};
#endif

#if(GFDNUM==16)
int w[16][16] = {
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
	1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,
	1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,
	1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,
	1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,
	1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,
	1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,
	1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,
	1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,
	1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,
	1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,
	1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,
	1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,
	1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,
	1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1
};
#endif

#if(GFDNUM==32)
int w[32][32] ={
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
	1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
	1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,
	1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,
	1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,
	1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,
	1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,
	1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,
	1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,
	1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,
	1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,
	1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,
	1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,
	1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,
	1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,
	1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,
	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
	1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,
	1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,
	1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,
	1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,
	1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,
	1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,
	1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,
	1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,
	1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,
	1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,
	1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,
	1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,
	1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,
	1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,
	1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1
};
#endif

#if(GFDNUM==64)

int w[64][64] = { 1, 1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,
                  1,-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,
				  1, 1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,
				  1, -1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,
				  1, 1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,
				  1, -1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,
				  1, 1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,
				  1, -1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,
				  1, 1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,
				  1, -1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,
				  1, 1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,
				  1, -1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,
				  1, 1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,
				  1, -1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,
				  1, 1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,
				  1, -1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,
				  1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,
				  1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,
				  1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,
				  1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,
				  1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,
				  1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,
				  1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,
				  1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,
				  1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,
				  1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,
				  1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,
				  1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,
				  1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,
				  1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,
				  1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,
				  1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,
				  1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,
				  1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,
				  1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,
				  1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,
				  1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,
				  1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,
				  1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,
				  1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,
				  1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,
				  1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,
				  1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,
				  1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,
				  1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,
				  1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,
				  1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,
				  1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,
				  1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,
				  1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,
				  1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,
				  1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,
				  1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,
				  1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,
				  1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,
				  1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,
				  1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,
				  1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,
				  1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,
				  1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,
				  1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,
				  1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,
				  1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,	-1,	-1,	1,	1,	1,	1,	-1,	-1,	1,	1,	-1,	-1,	-1,	-1,	1,	1,
				  1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	1,	-1,	1,	-1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	1,	-1,
					};

					#endif

#if(GFDNUM==256)
int w[256][256];
#endif

int nbLDPC::destroy_crossList()
{
	
	struct OLNode *p1, *p2;
	for (int i = 0; i != ROWS; ++i)
	{
		if (M.rhead[i] == NULL)
			continue;
		p1 = M.rhead[i]->right;
		//M.rhead[i] = NULL;
		while (p1 != NULL)
		{
			p2 = p1;
			p1 = p1->right;
			delete []p2->Qmn;
			delete []p2->qmn;
			delete []p2->rmn;
			delete []p2->QmnTemp;
			delete []p2->LLR;
			delete p2;
		}
	}
	

	for (int i = 0; i != ROWS; ++i)
	{
		if (M.rhead[i] == NULL)
			continue;
		if (M.rhead[i]->Status)
		{
			delete[]M.rhead[i]->Qmn;
			delete[]M.rhead[i]->qmn;
			delete[]M.rhead[i]->rmn;
			delete[]M.rhead[i]->QmnTemp;
			delete[]M.rhead[i]->LLR;
		}
	}

	cout << "##############" << endl;
	


	
	cout << "%%%%%%%%%%%%%%%%%" << endl;

	delete[]M.rhead;
	delete[]M.chead;

	return 1;
}

nbLDPC::nbLDPC(int Hrows, int Hcols, int q)
{
	codeLen = Hcols;
	ROWS = Hrows;
	cols = Hcols;
	now_rows = ROWS;
	infoLen = Hcols - Hrows;
	codeLen = Hcols;
	qnumber = q;//q-array
	
#if(LINEAR != 1)
	rearrange = new unsigned char[cols];
	H = new unsigned char[ROWS*cols];
	G = new unsigned char[(cols - ROWS)*ROWS];
	for (int j = 0; j<cols; j++)
		rearrange[j] = 0;

	for (int i = 0; i<ROWS; i++)
		for (int j = 0; j<cols; j++)
			H[i*cols + j] = 0;

	for (int i = 0; i<cols - ROWS; i++)
		for (int j = 0; j<ROWS; j++)
			G[i*ROWS + j] = 0;
#endif

	
}
nbLDPC::~nbLDPC()
{
#if(LINEAR != 1)
	delete[]H;
	delete[]G;
	delete[]rearrange;
#endif
	if (destroy_crossList())
	{
		cout << "succssfully delete the crossList !" << endl;
	}
}
bool nbLDPC::HasNonZeroInTheRow(int i)
{
	for (int j = 0; j != cols; ++j)
	{
		if (H[i*cols + j] != 0)
		{
			return true;
		}
	}
	return false;
}

void nbLDPC::rearrangeXS()
{
	cout << "8888888888" << endl;
	cout << ROWS << endl;
	
	cout << "7777777777" << endl;
	

	// This is my programming 
	int i, j;
	int changed_row = ROWS - 1;
	for (i = 0; i != ROWS;)
	{
		cout << i << endl;
		int k;
		if (H[i*cols + i] == 0)
		{
			k = i + 1;
			if (k<cols - 1)
			{
				while (H[i*cols + k] == 0)
				{
					k = k + 1;
					if (k == cols - 1)
						break;
				}
				for (j = 0; j < ROWS; j++)	//交换列元素(把第i列和第k列交换)
				{
					swap(H[j*cols + i], H[j*cols + k]);
				}
				rearrange[i] = k;  //record the rearrange column number
				col_permutation cp;
				cp.col_1 = i;
				cp.col_2 = k;
				permutation_nodes.push_back(cp);
			}
		}
		else
			rearrange[i] = 0;

		

		//if i-th row has all zero, we need row permutations
		//changed_row = rows-1;
		if (!HasNonZeroInTheRow(i))
		{
			//if (changed_row == i) // below i-th row, all has zero
			if(changed_row <= i) // I think this could solve a problem remaining under cover
			{
				cout << "below " << i << "-th row, all has zero" << endl;
				break;
			}
			cout << i << "-th row has all zero, we need row permutations" << endl;
			for (int k = 0; k != cols; ++k)
			{
				swap(H[changed_row*cols + k], H[i*cols + k]);
			}
			changed_row--;
			cout << changed_row << " ";
			while (!HasNonZeroInTheRow(i) && changed_row != i)
			{
				cout << changed_row << " ";
				for (int k = 0; k != cols; ++k)
				{
					swap(H[changed_row*cols + k], H[i*cols + k]);
				}
				changed_row--;
			}
		}

		

		k = i;
		if (k<cols - 1)
		{
			while (H[i*cols + k] == 0)
			{
				k = k + 1;
				if (k == cols - 1)
					break;
			}
			if (k != i)
			{
				for (j = 0; j < ROWS; j++)	//交换列元素(把第i列和第k列交换)
				{
					swap(H[j*cols + i], H[j*cols + k]);
				}
				rearrange[i] = k;  //record the rearrange column number
				col_permutation cp;
				cp.col_1 = i;
				cp.col_2 = k;
				permutation_nodes.push_back(cp);
			}
		}

		
		if (H[i*cols + i] != 0) // H may be not full-rank.
		{

			gfe ele = GF[GFindex[H[i*cols + i]]] ^ (gfdnum[elementBits] - 2);

			//cout << "$$$$" << endl;

			for (int k = 0; k != cols; ++k)
			{
				//cout << (int)H[i * cols + k] << " ";
				H[i*cols + k] = (GF[GFindex[H[i*cols + k]]] * ele).getValue();
			}
			//cout << endl;

			//cout << "####" << endl;

			// make the other elements in the i cols become 0 except A(i,i)
			for (j = 0; j < ROWS; j++)
			{
				if (j != i)
				{  
					// to make sure the column will be zero except the major element.
					if (H[j*cols + i] != 0)
					{
						//int m = H[j*cols + i];//m(i,:)*m(j,i)+m(j,:)---->m(j,:)
						gfe ele2 = GF[GFindex[H[j*cols + i]]];
						for (int k = 0; k <cols; k++)
						{
							//H[j*cols+k]=add[H[j*cols+k]][mul[H[i*cols+k]][m]];
							//H[j*cols + k] = H[i*cols + k] ^ H[j*cols + k];
							H[j*cols + k] = ((GF[GFindex[H[i*cols + k]]]*ele2) + GF[GFindex[H[j*cols + k]]]).getValue();
						}
					}
				}
			}
		}

		//cout << "%%%%" << endl;

		i++;
	}

	//if there are some redundant rows, we need eliminate them
	now_rows = ROWS;
	while (!HasNonZeroInTheRow(now_rows - 1))
		now_rows--;
	update();



	int num = 0;
	for (int i = 0; i != now_rows; ++i)
	{
		if (H[i*cols + i] == 1)
			num++;
	}
	cout << "the ONE number " << endl;
	cout << num << endl;

	for (int i = 0; i != permutation_nodes.size(); ++i)
	{
		cout << "(" << permutation_nodes[i].col_1 << " " << permutation_nodes[i].col_2 << ") ";
	}
	cout << endl;
	/* 输出列交换的信息
	cout<<"交换:"<<endl;
	for(int ii=0;ii<cols;ii++)
	cout<<rearrange[ii]<<" ";
	cout<<endl<<endl;
	//  rhs=m;// m是[I,D]的形式。D是A逆与B的乘积
	*/
}
void nbLDPC::gen_G()
{
	int i, j;

    char **D = new char *[now_rows];
	for (i = 0; i<now_rows; i++)
		D[i] = new char[cols - now_rows];
	for (i = 0; i<now_rows; i++)
		for (int j = 0; j<cols - now_rows; j++)
			D[i][j] = H[i*cols + j + now_rows];
	//G[j*rows+i]=H[i*cols+j+rows];


	//G[(j-rows)*rows+i]=H[i*cols+j+rows]; //这里，D是---A的逆与B的乘积,而G是D的转置
	const char* G_FILE_name = "G_FILE.txt";
	FILE* G_FILE = fopen(G_FILE_name, "w");
	for (i = 0; i<cols - now_rows; i++)
	{
		for (j = 0; j<now_rows; j++)
		{
			G[i*now_rows + j] = D[j][i];
			if(G[i*now_rows+j] >= qnumber)
				fprintf(G_FILE,"%2d ",G[i*now_rows+j]);	
		}
		fprintf(G_FILE,"\n");

		//fprintf(G_FILE,"\n");
	}
	fclose(G_FILE);
	for (i = 0; i<now_rows; i++)
	{
		delete[]D[i];
		D[i] = NULL;
	}
	delete[]D;
	D = NULL;
}
void nbLDPC::Encoder(int * Mesg, int *CodeWord)
{
	int *Check = new int[codeLen - infoLen];
	int i;
	int CheckLen = now_rows;
	for (i = 0; i < CheckLen; i++)
		Check[i] = 0;

//	const char* MESG_FILE_name_2 = "MESG_FILE_2.txt";
//	FILE* MESG_FILE_2 = fopen(MESG_FILE_name_2, "w");
//	fprintf(MESG_FILE_2, "%2d\n", CheckLen);
//	fprintf(MESG_FILE_2, "%2d\n", codeLen);
//	for (i = 0; i < codeLen - CheckLen; i++)
//		fprintf(MESG_FILE_2, "%2d ", Mesg[i]);
//	fclose(MESG_FILE_2);

//	const char* FILE_name = "FILE_FLAG.txt";
//	FILE* FILE_FLAG = fopen(FILE_name, "w");
//	fprintf(FILE_FLAG,"#############\n");
	
	
//	fprintf(FILE_FLAG,"%%%%%%%%%%%%\n");
	ArrayMultiply(Check, Mesg, G, cols - now_rows, now_rows);
//	fprintf(FILE_FLAG,"^^^^^^^^^^^^\n");
//	fclose(FILE_FLAG);
//	const char* check_FILE_name = "check_FILE.txt";
//	FILE* check_FILE = fopen(check_FILE_name, "w");
//	for (i = 0; i < CheckLen; i++)
//		fprintf(check_FILE, "%2d ", Check[i]);
//	fclose(check_FILE);

	for (i = 0; i < CheckLen; i++)
		CodeWord[i] = Check[i];
	for (i = CheckLen; i < codeLen; i++)
		CodeWord[i] = Mesg[i - CheckLen];


//	const char* MESG_FILE_name_1 = "MESG_FILE_1.txt";
//	FILE* MESG_FILE_1 = fopen(MESG_FILE_name_1, "w");
//	fprintf(MESG_FILE_1, "%2d\n", CheckLen);
//	fprintf(MESG_FILE_1, "%2d\n", codeLen);
//	for (i = 0; i < codeLen-CheckLen; i++)
//		fprintf(MESG_FILE_1, "%2d ", Mesg[i]);
//	fclose(MESG_FILE_1);

//	const char* CODE_FILE_name = "CODE_FILE.txt";
//	FILE* CODE_FILE = fopen(CODE_FILE_name, "w");
//	for (i = 0; i < codeLen; i++)
//		fprintf(CODE_FILE, "%2d ", CodeWord[i]);
//	fclose(CODE_FILE);

	//ReorderSymbol(CodeWord);
	//Reorder_bits(CodeWord, codeLen);
	Reorder_bits(CodeWord);
	//CheckEncoder(CodeWord);
	delete[]Check;
	Check = NULL;

}
void nbLDPC::Reorder_bits(int *u)
{
	for (int i = permutation_nodes.size() - 1; i != -1; --i)
	{
		int temp = u[permutation_nodes[i].col_1];
		u[permutation_nodes[i].col_1] = u[permutation_nodes[i].col_2];
		u[permutation_nodes[i].col_2] = temp;
	}
}
void nbLDPC::order_bits(int *ref)
{
	for (int i = 0; i != permutation_nodes.size(); ++i)
	{
		int temp = ref[permutation_nodes[i].col_1];
		ref[permutation_nodes[i].col_1] = ref[permutation_nodes[i].col_2];
		ref[permutation_nodes[i].col_2] = temp;
	}
}
void nbLDPC::Reorder_bits(int *u, int start)
{
	for (int i = permutation_nodes.size() - 1; i != -1; --i)
	{
		int temp = u[start + permutation_nodes[i].col_1];
		u[start + permutation_nodes[i].col_1] = u[start + permutation_nodes[i].col_2];
		u[start +permutation_nodes[i].col_2] = temp;
	}
}
void nbLDPC::order_bits(int *ref, int start)
{
	for (int i = 0; i != permutation_nodes.size(); ++i)
	{
		int temp = ref[start + permutation_nodes[i].col_1];
		ref[start + permutation_nodes[i].col_1] = ref[start + permutation_nodes[i].col_2];
		ref[start + permutation_nodes[i].col_2] = temp;
	}
}
void nbLDPC::update()
{
	//if there are some redundant rows, we need update
	//the row and cols and rate after eliminating these rows
	infoLen = cols - now_rows;
	delete[]G;
	G = new unsigned char[infoLen*now_rows];
	for (int i = 0; i<infoLen; i++)
		for (int j = 0; j<now_rows; j++)
			G[i*now_rows + j] = 0;
}
void nbLDPC::extract_mesg(int *res, int *ref)
{
	for (int i = 0; i != permutation_nodes.size(); ++i)
	{
		int temp = ref[permutation_nodes[i].col_1];
		ref[permutation_nodes[i].col_1] = ref[permutation_nodes[i].col_2];
		ref[permutation_nodes[i].col_2] = temp;
	}
	for (int i = 0; i != cols - now_rows; ++i)
	{
		res[i] = ref[now_rows + i];
	}
}
void nbLDPC::extract_mesg(double **res, double **ref)
{
	for (int i = 0; i != permutation_nodes.size(); ++i)
	{
		for (int j = 0; j < qnumber; ++j)
		{
			int temp = ref[j][permutation_nodes[i].col_1];
			ref[j][permutation_nodes[i].col_1] = ref[j][permutation_nodes[i].col_2];
			ref[j][permutation_nodes[i].col_2] = temp;
		}
	}
	for (int i = 0; i != cols - now_rows; ++i)
	{
		for (int j = 0; j < qnumber; ++j)
		{
			res[j][i] = ref[j][now_rows + i];
		}
	}
}
int nbLDPC::CreatMatrix_OL_nonbinary(char *s)
{
	int numCOL, numROW, RowWeight, numELEMENT;

	fstream in;
	in.open(s);
	in >> numCOL;
	in >> numROW;
	in >> RowWeight;
	in >> numELEMENT;
	numofnonzeroElem = numELEMENT;
	ROWS = numROW;
	cols = numCOL;



	/*** construct M chainList for decoding  *****/
	M.rhead = new OLNode *[ROWS];
	M.chead = new OLNode *[cols];
	//Initialize the head pointer vector
	for (int i = 0; i != ROWS; ++i)
		M.rhead[i] = NULL;
	for (int i = 0; i != cols; ++i)
		M.chead[i] = NULL;

	struct OLNode *p1, *p2, *q;
	p1 = p2 = q = NULL;
	int temp = 0;
	int value = -1;

	cout << "*****************************************" << endl;

	for (int i = 0; i != ROWS; ++i)
	{
		in >> temp;
		in >> value;
		//cout << temp << " " << value << " ";
		while (temp != numCOL)
		{
#if(LINEAR!=1)
			H[i*cols + temp] = value;
#endif
			p1 = new OLNode;
			if (p1 == NULL)
			{
				cout << "Function CreateMatrix_OL(): Memory exhausted !" << endl;
				exit(1);
			}
			//initialize nodes
			//initialize nodes
			p1->row_num = i;
			p1->col_num = temp;
			p1->ele = GF[GFindex[value]];
			

			p1->qmn = new double[qnumber];
			p1->Qmn = new double[qnumber];
			p1->QmnTemp = new double[qnumber];
			p1->rmn = new double[qnumber];
			p1->LLR = new double[qnumber];
			p1->Status = true;
			

			for (int idx = 0; idx != qnumber; ++idx)
			{
				p1->qmn[idx] = 0.0;
				p1->Qmn[idx] = 0.0;
				p1->QmnTemp[idx] = 0.0;
				p1->rmn[idx] = 0.0;
			}

			//insert at the direction of row
			if (M.rhead[i] == NULL)
			{
				M.rhead[i] = p1;
				p1->right = NULL;
			}
			else
			{
				p2->right = p1;
				p1->right = NULL;
			}
			p2 = p1;

			//insert at the direction of column
			if (M.chead[temp] == NULL)
			{
				M.chead[temp] = p1;
				p1->down = NULL;
			}
			else
			{
				for (q = M.chead[temp]; q->down != NULL; q = q->down);
				q->down = p1;
				p1->down = NULL;
			}
			in >> temp;
			in >> value;
			//cout << temp << " " << value << " ";
		}
		//cout << endl;
	}
	
	cout << ROWS << " " << cols << endl;
	return 1;

}
int nbLDPC::CreatMatrix_OL_nonbinary(string &s)
{
	int numCOL, numROW, RowWeight, numELEMENT;

	fstream in;
	in.open(s);
	in >> numCOL;
	in >> numROW;
	in >> RowWeight;
	in >> numELEMENT;
	numofnonzeroElem = numELEMENT;
	ROWS = numROW;
	cols = numCOL;



	/*** construct M chainList for decoding  *****/
	M.rhead = new OLNode *[ROWS];
	M.chead = new OLNode *[cols];
	//Initialize the head pointer vector
	for (int i = 0; i != ROWS; ++i)
		M.rhead[i] = NULL;
	for (int i = 0; i != cols; ++i)
		M.chead[i] = NULL;

	struct OLNode *p1, *p2, *q;
	p1 = p2 = q = NULL;
	int temp = 0;
	int value = -1;

	cout << "*****************************************" << endl;

	for (int i = 0; i != ROWS; ++i)
	{
		in >> temp;
		in >> value;
		//cout << temp << " " << value << " ";
		while (temp != numCOL)
		{
#if(LINEAR!=1)
			H[i*cols + temp] = value;
#endif
			p1 = new OLNode;
			if (p1 == NULL)
			{
				cout << "Function CreateMatrix_OL(): Memory exhausted !" << endl;
				exit(1);
			}
			//initialize nodes
			//initialize nodes
			p1->row_num = i;
			p1->col_num = temp;
			p1->ele = GF[GFindex[value]];


			p1->qmn = new double[qnumber];
			p1->Qmn = new double[qnumber];
			p1->QmnTemp = new double[qnumber];
			p1->rmn = new double[qnumber];
			p1->LLR = new double[qnumber];
			p1->Status = true;


			for (int idx = 0; idx != qnumber; ++idx)
			{
				p1->qmn[idx] = 0.0;
				p1->Qmn[idx] = 0.0;
				p1->QmnTemp[idx] = 0.0;
				p1->rmn[idx] = 0.0;
			}

			//insert at the direction of row
			if (M.rhead[i] == NULL)
			{
				M.rhead[i] = p1;
				p1->right = NULL;
			}
			else
			{
				p2->right = p1;
				p1->right = NULL;
			}
			p2 = p1;

			//insert at the direction of column
			if (M.chead[temp] == NULL)
			{
				M.chead[temp] = p1;
				p1->down = NULL;
			}
			else
			{
				for (q = M.chead[temp]; q->down != NULL; q = q->down);
				q->down = p1;
				p1->down = NULL;
			}
			in >> temp;
			in >> value;
			//cout << temp << " " << value << " ";
		}
		//cout << endl;
	}

	cout << ROWS << " " << cols << endl;
	return 1;

}
int nbLDPC::CreatMatrix_OL_nonbinary_alist(char *s)
{
	int numCOL, numROW, numQ, maxColWeight, maxRowWeight, numELEMENT;

	fstream in;
	in.open(s);
	in >> numCOL;
	in >> numROW;
	in >> numQ;
	in >> maxColWeight;
	in >> maxRowWeight;
	ROWS = numROW;
	cols = numCOL;

	int *deg_col = new int[numCOL];
	int *deg_row = new int[numROW];
	for (int i = 0; i < numCOL; ++i)
		in >> deg_col[i];
	for (int i = 0; i < numROW; ++i)
		in >> deg_row[i];

	int temp = 0;
	int value = -1;
	for (int j = 0; j < numCOL; ++j)
	{
		for (int i = 0; i < maxColWeight; ++i)
		{
			in >> temp;
			in >> value;
		}
	}


	/*** construct M chainList for decoding  *****/
	M.rhead = new OLNode *[ROWS];
	M.chead = new OLNode *[cols];
	//Initialize the head pointer vector
	for (int i = 0; i != ROWS; ++i)
		M.rhead[i] = NULL;
	for (int i = 0; i != cols; ++i)
		M.chead[i] = NULL;

	struct OLNode *p1, *p2, *q;
	p1 = p2 = q = NULL;
	

	for (int i = 0; i != ROWS; ++i)
	{
		for(int j = 0; j < maxRowWeight; ++j)
		{ 
			in >> temp;
			in >> value;
			if (temp == 0 && value == 0)
				continue;
			temp = temp - 1;
			value = GF[value].getValue();
			cout << temp << " " << value << " ";
			H[i*cols + temp] = value;
			p1 = new OLNode;
			if (p1 == NULL)
			{
				cout << "Function CreateMatrix_OL(): Memory exhausted !" << endl;
				exit(1);
			}
			//initialize nodes
			//initialize nodes
			p1->row_num = i;
			p1->col_num = temp;
			p1->ele = GF[GFindex[value]];


			p1->qmn = new double[qnumber];
			p1->Qmn = new double[qnumber];
			p1->QmnTemp = new double[qnumber];
			p1->rmn = new double[qnumber];
			p1->LLR = new double[qnumber];
			p1->Status = true;


			for (int idx = 0; idx != qnumber; ++idx)
			{
				p1->qmn[idx] = 0.0;
				p1->Qmn[idx] = 0.0;
				p1->QmnTemp[idx] = 0.0;
				p1->rmn[idx] = 0.0;
			}

			//insert at the direction of row
			if (M.rhead[i] == NULL)
			{
				M.rhead[i] = p1;
				p1->right = NULL;
			}
			else
			{
				p2->right = p1;
				p1->right = NULL;
			}
			p2 = p1;

			//insert at the direction of column
			if (M.chead[temp] == NULL)
			{
				M.chead[temp] = p1;
				p1->down = NULL;
			}
			else
			{
				for (q = M.chead[temp]; q->down != NULL; q = q->down);
				q->down = p1;
				p1->down = NULL;
			}
			//cout << temp << " " << value << " ";
		}
		cout << endl;
	}

	delete[]deg_col;
	delete[]deg_row;

	cout << ROWS << " " << cols << endl;
	return 1;

}
int nbLDPC::CreatMatrix_OL_nonbinary_alist_v2(char *s)
{
	int numCOL, numROW, numQ, maxColWeight, maxRowWeight, numELEMENT;

	fstream in;
	in.open(s);
	in >> numCOL;
	in >> numROW;
	in >> numQ;
	in >> maxColWeight;
	in >> maxRowWeight;
	ROWS = numROW;
	cols = numCOL;

	int *deg_col = new int[numCOL];
	int *deg_row = new int[numROW];
	for (int i = 0; i < numCOL; ++i)
		in >> deg_col[i];
	for (int i = 0; i < numROW; ++i)
		in >> deg_row[i];

	int temp = 0;
	int value = -1;


	/*** construct M chainList for decoding  *****/
	M.rhead = new OLNode *[ROWS];
	M.chead = new OLNode *[cols];
	//Initialize the head pointer vector
	for (int i = 0; i != ROWS; ++i)
		M.rhead[i] = NULL;
	for (int i = 0; i != cols; ++i)
		M.chead[i] = NULL;

	struct OLNode *p1, *p2, *q;
	p1 = p2 = q = NULL;


	for (int i = 0; i != ROWS; ++i)
	{
		for (int j = 0; j < maxRowWeight; ++j)
		{
			in >> temp;
			in >> value;
			if (temp == 0 && value == 0)
				continue;
			temp = temp - 1;
			value = GF[value].getValue();
			cout << temp << " " << value << " ";
			H[i*cols + temp] = value;
			p1 = new OLNode;
			if (p1 == NULL)
			{
				cout << "Function CreateMatrix_OL(): Memory exhausted !" << endl;
				exit(1);
			}
			//initialize nodes
			//initialize nodes
			p1->row_num = i;
			p1->col_num = temp;
			p1->ele = GF[GFindex[value]];


			p1->qmn = new double[qnumber];
			p1->Qmn = new double[qnumber];
			p1->QmnTemp = new double[qnumber];
			p1->rmn = new double[qnumber];
			p1->LLR = new double[qnumber];
			p1->Status = true;


			for (int idx = 0; idx != qnumber; ++idx)
			{
				p1->qmn[idx] = 0.0;
				p1->Qmn[idx] = 0.0;
				p1->QmnTemp[idx] = 0.0;
				p1->rmn[idx] = 0.0;
			}

			//insert at the direction of row
			if (M.rhead[i] == NULL)
			{
				M.rhead[i] = p1;
				p1->right = NULL;
			}
			else
			{
				p2->right = p1;
				p1->right = NULL;
			}
			p2 = p1;

			//insert at the direction of column
			if (M.chead[temp] == NULL)
			{
				M.chead[temp] = p1;
				p1->down = NULL;
			}
			else
			{
				for (q = M.chead[temp]; q->down != NULL; q = q->down);
				q->down = p1;
				p1->down = NULL;
			}
			//cout << temp << " " << value << " ";
		}
		cout << endl;
	}

	delete[]deg_col;
	delete[]deg_row;

	cout << ROWS << " " << cols << endl;
	return 1;

}
int nbLDPC::CreatMatrix_OL_nonbinary_Neal(char *s)
{
	int numCOL, numROW, RowWeight, numELEMENT;

	fstream in;
	in.open(s);
	if (in.is_open())
		cout << "open" << endl;
	else
		cout << "no open" << endl;

	cout << s << endl;
	in >> numCOL;
	in >> numROW;
	ROWS = numROW;
	cols = numCOL;

	cout << ROWS << " " << cols << endl;

	/*** construct M chainList for decoding  *****/
	M.rhead = new OLNode *[ROWS];
	M.chead = new OLNode *[cols];
	//Initialize the head pointer vector
	for (int i = 0; i != ROWS; ++i)
		M.rhead[i] = NULL;
	for (int i = 0; i != cols; ++i)
		M.chead[i] = NULL;

	struct OLNode *p1, *p2, *q;
	p1 = p2 = q = NULL;

	int value = -1;
	int row = -1;
	int b = 0;
	int v;
	int col;

	while(1)
	{
		in >> v;
		cout << v << " ";

		if (v == 0) { return 1; }
		else if (v<0)
		{
			row = -v - 1;
			if (row >= ROWS) break;
		}
		else
		{		
			//Peng 122505
			if (b)
				value = v;
			else
				col = v - 1;

			if (col >= cols) break;
			if (row == -1) break;
			if (b)
			{
				H[row*cols + col] = value;
				p1 = new OLNode;
				if (p1 == NULL)
				{
					cout << "Function CreateMatrix_OL(): Memory exhausted !" << endl;
					exit(1);
				}
				//initialize nodes
				//initialize nodes
				p1->row_num = row;
				p1->col_num = col;
				p1->ele = GF[GFindex[value]];


				p1->qmn = new double[qnumber];
				p1->Qmn = new double[qnumber];
				p1->QmnTemp = new double[qnumber];
				p1->rmn = new double[qnumber];
				p1->LLR = new double[qnumber];
				p1->Status = true;


				for (int idx = 0; idx != qnumber; ++idx)
				{
					p1->qmn[idx] = 0.0;
					p1->Qmn[idx] = 0.0;
					p1->QmnTemp[idx] = 0.0;
					p1->rmn[idx] = 0.0;
				}

				//insert at the direction of row
				if (M.rhead[row] == NULL)
				{
					M.rhead[row] = p1;
					p1->right = NULL;
				}
				else
				{
					p2->right = p1;
					p1->right = NULL;
				}
				p2 = p1;

				//insert at the direction of column
				if (M.chead[col] == NULL)
				{
					M.chead[col] = p1;
					p1->down = NULL;
				}
				else
				{
					for (q = M.chead[col]; q->down != NULL; q = q->down);
					q->down = p1;
					p1->down = NULL;
				}
			}
			b ^= 1;
		}
	}

	return 1;

}
int nbLDPC::CreatMatrix_OL_nonbinary_ldspec(char *s)
{
	int numCOL, numROW, EXTENSION, RowWeight, ColWeight;
	char type;
	cout << "****************" << endl;
	fstream in;
	in.open(s);
	in >> numCOL;
	in >> numROW;
	in >> EXTENSION;
	cout << "nC = " << numCOL << endl;
	cout << "nR = " << numROW << endl;
	in >> type;
	in >> RowWeight;
	in >> ColWeight;
	cout << "CW = " << ColWeight << endl;
	cout << "RW = " << RowWeight << endl;
	ROWS = numROW;
	cols = numCOL;



	/*** construct M chainList for decoding  *****/
	M.rhead = new OLNode *[ROWS];
	M.chead = new OLNode *[cols];
	//Initialize the head pointer vector
	for (int i = 0; i != ROWS; ++i)
		M.rhead[i] = NULL;
	for (int i = 0; i != cols; ++i)
		M.chead[i] = NULL;

	struct OLNode *p1, *p2, *q;
	p1 = p2 = q = NULL;
	int temp = 0;
	int value = -1;

	for (int i = 0; i < ROWS; ++i)
	{
		for(int j = 0; j < RowWeight; ++j)
		{ 
			in >> temp;
			in >> value;
			cout << temp << " " << value << " ";
			H[i*cols + temp] = value;
			p1 = new OLNode;
			if (p1 == NULL)
			{
				cout << "Function CreateMatrix_OL(): Memory exhausted !" << endl;
				exit(1);
			}
			//initialize nodes
			//initialize nodes
			p1->row_num = i;
			p1->col_num = temp;
			p1->ele = GF[GFindex[value]];


			p1->qmn = new double[qnumber];
			p1->Qmn = new double[qnumber];
			p1->QmnTemp = new double[qnumber];
			p1->rmn = new double[qnumber];
			p1->Status = true;


			for (int idx = 0; idx != qnumber; ++idx)
			{
				p1->qmn[idx] = 0.0;
				p1->Qmn[idx] = 0.0;
				p1->QmnTemp[idx] = 0.0;
				p1->rmn[idx] = 0.0;
			}

			//insert at the direction of row
			if (M.rhead[i] == NULL)
			{
				M.rhead[i] = p1;
				p1->right = NULL;
			}
			else
			{
				p2->right = p1;
				p1->right = NULL;
			}
			p2 = p1;

			//insert at the direction of column
			if (M.chead[temp] == NULL)
			{
				M.chead[temp] = p1;
				p1->down = NULL;
			}
			else
			{
				for (q = M.chead[temp]; q->down != NULL; q = q->down);
				q->down = p1;
				p1->down = NULL;
			}
		}
		cout << endl;
	}

	cout << ROWS << " " << cols << endl;
	return 1;

}
int nbLDPC::CreatMatrix_OL_nonbinary_FromDrLi(char *s)
{
	int numCOL, numROW, MaxRowWeight;

	fstream in;
	in.open(s);
	in >> numCOL;
	in >> numROW;
	in >> MaxRowWeight;
	ROWS = numROW;
	cols = numCOL;



	/*** construct M chainList for decoding  *****/
	M.rhead = new OLNode *[ROWS];
	M.chead = new OLNode *[cols];
	//Initialize the head pointer vector
	for (int i = 0; i != ROWS; ++i)
		M.rhead[i] = NULL;
	for (int i = 0; i != cols; ++i)
		M.chead[i] = NULL;

	struct OLNode *p1, *p2, *q;
	p1 = p2 = q = NULL;
	int temp = 0;
	int value = -1;

	for (int i = 0; i != ROWS; ++i)
	{
		for(int j = 0; j < MaxRowWeight; ++j)
		{ 
			in >> temp;
			if (temp != 0)
			{
				temp--;
				in >> value;
				cout << temp << " " << value << " ";
				H[i*cols + temp] = value;
				p1 = new OLNode;
				if (p1 == NULL)
				{
					cout << "Function CreateMatrix_OL(): Memory exhausted !" << endl;
					exit(1);
				}
				//initialize nodes
				//initialize nodes
				p1->row_num = i;
				p1->col_num = temp;
				p1->ele = GF[GFindex[value]];


				p1->qmn = new double[qnumber];
				p1->Qmn = new double[qnumber];
				p1->QmnTemp = new double[qnumber];
				p1->rmn = new double[qnumber];
				p1->Status = true;


				for (int idx = 0; idx != qnumber; ++idx)
				{
					p1->qmn[idx] = 0.0;
					p1->Qmn[idx] = 0.0;
					p1->QmnTemp[idx] = 0.0;
					p1->rmn[idx] = 0.0;
				}

				//insert at the direction of row
				if (M.rhead[i] == NULL)
				{
					M.rhead[i] = p1;
					p1->right = NULL;
				}
				else
				{
					p2->right = p1;
					p1->right = NULL;
				}
				p2 = p1;

				//insert at the direction of column
				if (M.chead[temp] == NULL)
				{
					M.chead[temp] = p1;
					p1->down = NULL;
				}
				else
				{
					for (q = M.chead[temp]; q->down != NULL; q = q->down);
					q->down = p1;
					p1->down = NULL;
				}
			
			}
		}
		cout << endl;
	}

	cout << ROWS << " " << cols << endl;
	return 1;

}

void nbLDPC::fft(double* v, double *V, int qnumber)
{
	for (int j = 0; j < qnumber; ++j)
	{
		double value = 0;
		for (int i = 0; i < qnumber; ++i)
		{
			value += v[i] * w[i][j];
		}
		V[j] = value;
	}
}
void nbLDPC::FFT(double* res, double* ref, int qnumber)
{
	int i, j, k;
	int m = elementBits;
	for (i = 0; i < (1 << m); i++) // copy ref to res
		res[i] = ref[i];

	int **Flag = new int*[m];
	for (j = 0; j < m; j++)
	{
		Flag[j] = new int[1 << j];
		int StepSize = 1 << (m - j);
		/***  找到所有分组的界线  ***/
		for (k = 0; k < (1 << j); k++)
		{
			int len = 1 << (m - 1 - j);
			Flag[j][k] = len - 1 + k*StepSize;
			for (i = 0; i < len; i++)
			{
				double temp0 = res[Flag[j][k] - len + 1 + i];
				double temp1 = res[Flag[j][k] + 1 + i];
				res[Flag[j][k] - len + 1 + i] = temp0 + temp1;
				res[Flag[j][k] + 1 + i] = temp0 - temp1;
			}
		}
	}

	for (i = 0; i < m; i++)
		delete[]Flag[i];
	delete[]Flag;
}
void nbLDPC::IFFT(double* res, double* ref, int qnumber)
{
	FFT(res, ref, qnumber);
	for (int i = 0; i < qnumber; i++)
	{
		res[i] = res[i] / qnumber;
		if (res[i] < MINPROB)
			res[i] = MINPROB;
	}
}

void nbLDPC::sort(double *v, gfe ele, int qnumber)
{
	/*double *v_temp = new double[qnumber];
	cout << ele.getValue() << ": ";
	for (int i = 0; i != qnumber; ++i)
	{
		gfe A(i);
		int idx = (A*ele).getValue();
		v_temp[idx] = v[i];
		cout << i << "->" << idx << " ";
	}
	cout << endl;
	for (int i = 0; i != qnumber; ++i)
	{
		v[i] = v_temp[i];
	}
	delete[]v_temp;*/

	double *v_temp = new double[qnumber];
	//cout << ele.getValue() << ": ";
	for (int i = 0; i != qnumber; ++i)
	{
		v_temp[i] = v[i];
	}
	//cout << endl;
	for (int i = 0; i != qnumber; ++i)
	{
		gfe A(i);
		int idx = (A*ele).getValue();
		v[i] = v_temp[idx];
	}
	delete[]v_temp;
}
void nbLDPC::permute(double *input, double *buffer, gfe ele, int qnumber)
{
	for (int q = 0; q < qnumber; ++q)
	{
		buffer[q] = input[q];
	}

	for (int q = 0; q < qnumber; ++q)
	{
		gfe A(q);
		int idx = (A*ele).getValue();
		input[q] = buffer[idx];
	}
	
}
void nbLDPC::permute(double *buffer)
{
	cout << "CCCCCCCCCCCCC" << endl;
}

void nbLDPC::ifft(double *V, double *v, int qnumber)
{
	for (int j = 0; j < qnumber; ++j)
	{
		double value = 0;
		for (int i = 0; i < qnumber; ++i)
		{
			value += V[i] * w[i][j];
		}
		v[j] = value / qnumber;
		/*if (v[j] < 0.0)
			v[j] = 0.1;*/
		if (v[j] < MINPROB)
			v[j] = MINPROB;
	}
	

	//double sum = 0.0;
	//for (int j = 0; j < qnumber; ++j)
	//{
	//	sum += v[j];
	//}
	//for (int j = 0; j < qnumber; ++j)
	//{
	//	v[j] /= sum;
	//}
	
}
bool nbLDPC::judgeZero(int *vhat)
{
	bool isValid = true;
	OLNode *ps;
	for (int i = 0; i != ROWS; ++i)
	{
		gfe value(0);
		ps = M.rhead[i];
		while (ps != NULL)
		{
			
			value = value + (ps->ele * GF[GFindex[vhat[ps->col_num]]]);
			ps = ps->right;
		}
		if (value.getValue() != 0)
		{
			isValid = false;
			break;
		}
	}
	

	return isValid;
}

int nbLDPC::FFT_QSPA(double **f, double **Qn, int *vhat, int* Codeword, int MAXITRNUM)
{

	OLNode *ps, *qs;
	for (int j = 0; j < cols; ++j)
	{
		ps = M.chead[j];
		while (ps != NULL)
		{
			for (int idx = 0; idx < qnumber; ++idx)
			{
				ps->qmn[idx] = f[idx][j];
			}
			ps = ps->down;
		}
	}

	for (int itr_num = 1; itr_num <= MAXITRNUM; ++itr_num)
	{
		/*for (int i = 0; i < ROWS; ++i)
		{
			ps = M.rhead[i];
			while (ps != NULL)
			{
				fft(ps->qmn, ps->Qmn, qnumber);
				gfe H = ps->ele;
				H = H ^ (gfdnum[elementBits]-2);
				sort(ps->Qmn, H, qnumber);
				ps = ps->right;
			}
		}*/

		for (int i = 0; i < ROWS; ++i)
		{
			ps = M.rhead[i];
			while (ps != NULL)
			{
				gfe H = ps->ele;
				H = H ^ (gfdnum[elementBits] - 2);
				//cout << "ele = alpah^" << GFindex[ps->ele.getValue()] << " " << ps->ele.getValue() << " H = alpha^" << GFindex[H.getValue()] << " " << H.getValue() << " ";
				sort(ps->qmn, H, qnumber);
				//fft(ps->qmn, ps->Qmn, qnumber);
				FFT(ps->Qmn, ps->qmn, qnumber);
				ps = ps->right;
			}
		}

		//for (int i = 0; i < 1; ++i)
		//{
		//	ps = M.rhead[i];
		//	while (ps != NULL)
		//	{
		//		//cout << "( ";
		//		for (int k = 0; k < qnumber; ++k)
		//		{
		//			cout << ps->Qmn[k] << " ";
		//		}
		//		cout << endl;
		//		//cout << " )";
		//		ps = ps->right;
		//	}
		//}

		//for (int i = 0; i < 1; ++i)
		//{
		//	qs = M.rhead[i];
		//	while (qs != NULL)
		//	{
		//		cout << "ele=alpha^" << GFindex[qs->ele.getValue()] << " " << qs->ele.getValue() << endl;
		//		for (int idx = 0; idx != qnumber; ++idx)
		//		{
		//			gfe A(idx);
		//			int toIdx = (A*qs->ele).getValue();
		//			cout << idx << "->" << toIdx << " ";
		//		}
		//		cout << endl;
		//		for (int idx = 0; idx < qnumber; ++idx)
		//		{
		//			cout << qs->qmn[idx] << " ";
		//		}
		//		cout << endl;
		//		qs = qs->right;
		//	}
		//}

		

	/*	for (int j = 0; j != cols; ++j)
		{
			cout << f[0][j] << " ";
		}
		cout << endl;*/

		/*for (int i = 0; i != ROWS; ++i)
		{
			ps = M.rhead[i];
			while (ps != NULL)
			{
				for (int idx = 0; idx != qnumber; ++idx)
				{
					cout << ps->Qmn[idx] << " ";
				}
				ps = ps->right;
			}
		}*/
		
		bool isB = false;

		//computing the rmn
		for (int i = 0; i < ROWS; ++i)
		{
			ps = M.rhead[i];

			while (ps != NULL)
			{
				for (int idx = 0; idx < qnumber; ++idx)
				{
					ps->QmnTemp[idx] = 1;
				}
				qs = M.rhead[i];
				while (qs != NULL)
				{
					if (qs->col_num != ps->col_num)
					{
						for (int idx = 0; idx < qnumber; ++idx)
						{
							ps->QmnTemp[idx] *= qs->Qmn[idx];
						}
					}
					qs = qs->right;
				}
				//ifft(ps->QmnTemp, ps->rmn, qnumber);
				IFFT(ps->rmn, ps->QmnTemp, qnumber);
				gfe H = ps->ele;
				//cout << "actual, H = alpha^" << GFindex[H.getValue()] << " " << H.getValue() << " ";
				sort(ps->rmn, H, qnumber);

				ps = ps->right;
			}
		}

		
		

		
	    
		
		

		if (isB)
			cout << "B" << endl;
		


		/*for (int i = 0; i != ROWS; ++i)
		{
			ps = M.rhead[i];

			while (ps != NULL)
			{
				for (int idx = 0; idx != qnumber; ++idx)
				{
					ps->QmnTemp[idx] = 1;
				}
				qs = M.rhead[i];
				while (qs != NULL)
				{
					if (qs->col_num != ps->col_num)
					{
						for (int idx = 0; idx != qnumber; ++idx)
						{
							ps->QmnTemp[idx] *= qs->Qmn[idx];
						}
					}
					qs = qs->right;
				}
				ifft(ps->QmnTemp, ps->rmn, qnumber);

				ps = ps->right;
			}
		}*/

		bool isA = false;

		//computing the qmn
		for (int j = 0; j < cols; ++j)
		{
			ps = M.chead[j];
			while (ps != NULL)
			{
				for (int idx = 0; idx < qnumber; ++idx)
				{
					ps->qmn[idx] = f[idx][j];
				}
				qs = M.chead[j];
				while (qs != NULL)
				{
					if (qs->row_num != ps->row_num)
					{
						for (int idx = 0; idx < qnumber; ++idx)
						{
							ps->qmn[idx] *= qs->rmn[idx];
						}
					}
					qs = qs->down;
				}
				 
				double sum = 0.0;
				for (int idx = 0; idx < qnumber; ++idx)
				{
					sum += ps->qmn[idx];
				}

				for (int idx = 0; idx < qnumber; ++idx)
				{
					ps->qmn[idx] /= sum;
					if (ps->qmn[idx] < 0)
					{
						isA = true;
					}
					if (ps->qmn[idx] < MINPROB)
						ps->qmn[idx] = MINPROB;
				}
				ps = ps->down;
			}
		}

		if (isA)
			cout << "A" << endl;

		//computing the Qn
		for (int j = 0; j < cols; ++j)
		{
			for (int idx = 0; idx != qnumber; ++idx)
			{
				Qn[idx][j] = f[idx][j];
			}
			ps = M.chead[j];
			while (ps != NULL)
			{
				/*if (j == 10)
					cout << ps->rmn[0] << " " << ps->rmn[1] << " ";*/
				for (int idx = 0; idx < qnumber; ++idx)
				{
					Qn[idx][j] *= ps->rmn[idx];
				}
				/*if (j == 10)
					cout << Qn[0][j] << " " << Qn[1][j] << " ";*/
				ps = ps->down;
			}
			/*
			cout << j << ": ";
			for (int idx = 0; idx != qnumber; ++idx)
			{
				cout << Qn[idx][j] << " ";
			}*/
			

			double sum = 0.0;
			for (int idx = 0; idx < qnumber; ++idx)
			{
				sum += Qn[idx][j];
			}
			for (int idx = 0; idx < qnumber; ++idx)
			{
				Qn[idx][j] /= sum;
			}

			/*for (int idx = 0; idx != qnumber; ++idx)
			{
				cout << Qn[idx][j] << " ";
			}
			cout << endl;*/
		}
		//cout << endl;

		//cout << "itr_num = " << itr_num << endl;
		/*for (int j = 0; j != cols; ++j)
		{
			cout << Qn[0][j] << " ";
		}
		cout << endl;*/
		//for (int j = 0; j != cols; ++j)
		//{
		//	cout << Qn[1][j] << " ";
		//}
		//cout << endl;

	    
		/*for (int j = 0; j < cols; ++j)
		{
			if(vhat[j] != Codeword[j])
				cout << j << " ";
		}
		cout << endl;*/

		//cout << "Dec = " << vhat[0] << ", Code = " << Codeword[0] << endl;
	/*	for (int j = 0; j < qnumber; ++j)
		{
			cout << Qn[j][1] << " ";
		}
		cout << endl;*/

		//decision
		for (int j = 0; j < cols; ++j)
		{
			int maxIdx = -1;
			double maxPoss = -1;
			for (int idx = 0; idx < qnumber; ++idx)
			{
				if (maxPoss < Qn[idx][j])
				{
					maxPoss = Qn[idx][j];
					maxIdx = idx;
				}
			}
			if (maxPoss<0)
			{
				while (1)
					cout << "wrong";
			}
			vhat[j] = maxIdx;
		}

	

		
		if (judgeZero(vhat))
		{
			break;
		}
	}
	
		
	


	return 0;
}

int nbLDPC::FFT_QSPA_log(double **inLLR, double **outLLR, int *vhat, int MAXITRNUM)
{

	bool validCode = 0;

	double maxLLR = -10000;
	double max;

	OLNode *ps, *qs;
	for (int j = 0; j < cols; ++j)
	{
		ps = M.chead[j];
		while (ps != NULL)
		{
			for (int idx = 0; idx < qnumber; ++idx)
			{
				ps->LLR[idx] = 0;
			}
			ps = ps->down;
		}
	}

	//cout << "1st iteration" << endl;
	

	double *EdgeLLR = new double[qnumber];
	double *buffer = new double[qnumber];
	double MINLLR = -35;
	int max_chk_deg = 0;
	for (int i = 0; i < ROWS; ++i)
	{
		ps = M.rhead[i];
		int chk_deg = 0;
		while (ps != NULL)
		{
			chk_deg++;
			ps = ps->right;
		}
		if (max_chk_deg < chk_deg)
			max_chk_deg = chk_deg;
	}

	int **LGsgn1 = new int*[max_chk_deg];
	for (int i = 0; i < max_chk_deg; ++i)
	{
		LGsgn1[i] = new int[qnumber];
	}
	int *LGsgn2 = new int[qnumber];

	
	out.open("intermedia.txt");

	for (int itr_num = 1; itr_num <= MAXITRNUM; ++itr_num)
	{
		//Update Message sent from varibable node to check node
		for (int j = 0; j < cols; ++j)
		{
			for (int q = 0; q < qnumber; ++q)
			{
				EdgeLLR[q] = inLLR[q][j];
				if (shouldput)
					osError << "{ " << j << " " << inLLR[q][j] << " ";
			}
			ps = M.chead[j];
			while (ps != NULL)
			{
				for (int q = 0; q < qnumber; ++q)
				{
					EdgeLLR[q] = EdgeLLR[q] + ps->LLR[q];
					
				}
				ps = ps->down;
			}
			ps = M.chead[j];
			while (ps != NULL)
			{
				for (int q = 0; q < qnumber; ++q)
				{
					if (EdgeLLR[q] > maxLLR0)
						maxLLR0 = EdgeLLR[q];
					if (EdgeLLR[q] < minLLR0)
						minLLR0 = EdgeLLR[q];
					ps->LLR[q] = EdgeLLR[q] - ps->LLR[q];
				}
				//for (int q = 0; q < qnumber; ++q)
				for (int q = 1; q < qnumber; ++q)
				{
					ps->LLR[q] = ps->LLR[q] - ps->LLR[0];
				}
				ps->LLR[0] = 0;
				ps = ps->down;
			}

		}

		/*
		for (int i = 0; i < cols; ++i)
		{

			ps = M.chead[i];
			int chk_deg = 0;
			while (ps != NULL)
			{
				for (int q = 0; q < qnumber; ++q)
				{
					out << ps->LLR[q] << " ";
				}
				out << endl;
				ps = ps->down;
			}
		}
		*/
		
		//printf("Update Message sent from check node to variable node\n");

		//Update Message sent from check node to variable node
		for (int i = 0; i < ROWS; ++i)
		{
			for (int q = 0; q < qnumber; ++q)
			{
				EdgeLLR[q] = 0;
				LGsgn2[q] = 1;
			}
			ps = M.rhead[i];
			int chk_deg = 0;
			while (ps != NULL)
			{
				int tmp1 = GFq_inv(ps->ele.getValue(), qnumber);
				gfe ele = tmp1;
				
				//out << "(" << ps->ele.getValue() << " " << tmp1 << " &&&&&&&&&& ";

				////////////////////////////////////////////////
				////////////////////////////////////////////////
				////////////////////////////////////////////////
				//for (int q = 0; q < qnumber; ++q)
				//{
				//	out << ps->LLR[q] << " ";
				//}

				//out << " ############ ";
				
				permute(ps->LLR, buffer, ele, qnumber);

				//out << " $$$$$$$$$$$$ ";
				//for (int q = 0; q < qnumber; ++q)
				//{
				//	out << ps->LLR[q] << " ";
				//}
				//out << endl;
				

				maxLLR = -10000;
				
				for (int q = 0; q < qnumber; ++q)
				{
					if (maxLLR < ps->LLR[q])
						maxLLR = ps->LLR[q];
				}
				for (int q = 0; q < qnumber; ++q)
				{
					double dtmp1 = ps->LLR[q] - maxLLR;
					if (dtmp1 < MINLLR) dtmp1 = MINLLR;
					ps->LLR[q] = exp(dtmp1);
					
				}

				

				mFFT(ps->LLR, qnumber);

				

				for (int q = 0; q < qnumber; ++q)
				{
					
					if (ps->LLR[q] == 0.0)
					{
						LGsgn1[chk_deg][q] = 1;
						ps->LLR[q] = -200;
					}
					else
					{
						if (ps->LLR[q] > 0)
						{
							LGsgn1[chk_deg][q] = 1;
							ps->LLR[q] = log(ps->LLR[q]);
						}
						else
						{
							LGsgn1[chk_deg][q] = -1;
							ps->LLR[q] = log(-ps->LLR[q]);
						}
					}
					
					LGmult(EdgeLLR + q, LGsgn2 + q, EdgeLLR[q], LGsgn2[q], ps->LLR[q], LGsgn1[chk_deg][q]);
				}
				chk_deg++;
				ps = ps->right;
			}

			ps = M.rhead[i];
			chk_deg = 0;
			while (ps != NULL)
			{
				int tmp1 = ps->ele.getValue();
				gfe ele = tmp1;
				for (int q = 0; q < qnumber; ++q)
				{
					LGdiv(ps->LLR + q, &LGsgn1[chk_deg][q], EdgeLLR[q], LGsgn2[q], ps->LLR[q], LGsgn1[chk_deg][q]);
				}
				maxLLR = -10000;
				for (int q = 0; q < qnumber; ++q)
				{
					if (maxLLR < ps->LLR[q])
						maxLLR = ps->LLR[q];
				}
				
				for (int q = 0; q < qnumber; ++q)
				{
					double dtmp1 = ps->LLR[q] - maxLLR;
					if (dtmp1 < MINLLR) dtmp1 = MINLLR;
					ps->LLR[q] = LGsgn1[chk_deg][q] * exp(dtmp1);
				}
	
				mIFFT(ps->LLR, qnumber);
				for (int q = 0; q < qnumber; ++q)
				{
					if (ps->LLR[q] < 0)
					{
						cout << "something wrong FFT" << endl;
					}
					if (ps->LLR[q] == 0)
					{
						ps->LLR[q] = MINLLR;
					}
					else
					{
						ps->LLR[q] = log(ps->LLR[q]);
					}
				}

				permute(ps->LLR, buffer, ele, qnumber);
				for (int q = 1; q < qnumber; ++q)
				{
					ps->LLR[q] = ps->LLR[q] - ps->LLR[0];
				}
				ps->LLR[0] = 0;

				chk_deg++;
				ps = ps->right;
			}
		}

		/*
		for (int i = 0; i < ROWS; ++i)
		{
			
			ps = M.rhead[i];
			int chk_deg = 0;
			while (ps != NULL)
			{

				for (int q = 0; q < qnumber; ++q)
				{
					out << ps->LLR[q] << " ";
				}
				out << endl;
				ps = ps->right;
			}
		}
		*/

		ofstream os;
		os.open("LLR_iter.txt");
		os << "****************************************" << endl;
		for (int q = 0; q < qnumber; ++q)
		{
			os << EdgeLLR[q] << " ";
		}
		os << endl;
		os << "****************************************" << endl;

		for (int j = 0; j < cols; ++j)
		{
			for (int q = 1; q < qnumber; ++q)
			{
				EdgeLLR[q] = inLLR[q][j];
			}
			ps = M.chead[j];
			while (ps != NULL)
			{
				for (int q = 1; q < qnumber; ++q)
				{
					EdgeLLR[q] = EdgeLLR[q] + ps->LLR[q];
				}
				ps = ps->down;
			}

			max = 0;
			vhat[j] = 0;

			for (int q = 1; q < qnumber; ++q)
			{
				if (EdgeLLR[q] > max)
				{
					vhat[j] = q;
					max = EdgeLLR[q];
				}
			}

			for (int q = 0; q < qnumber; ++q)
			{
				os << EdgeLLR[q] << " ";
			}
			os << endl;

		}

		os.close();

		if (judgeZero(vhat))
		{
			validCode = 1;
			break;
		}
	}

	out.close();

	
	for (int j = 0; j < cols; ++j)
	{
		for (int q = 0; q < qnumber; ++q)
			EdgeLLR[q] = 0;

		ps = M.chead[j];
		while (ps != NULL)
		{
			for (int q = 1; q < qnumber; ++q)
			{
				EdgeLLR[q] = EdgeLLR[q] + ps->LLR[q];
			}
			ps = ps->down;
		}
		maxLLR = -10000;
		for (int q = 0; q < qnumber; ++q)
		{
			outLLR[q][j] = EdgeLLR[q];
			if (outLLR[q][j] > maxLLR)
				maxLLR = outLLR[q][j];
		}

		for (int q = 0; q < qnumber; ++q)
		{
			outLLR[q][j] = outLLR[q][j] - maxLLR;
			if (outLLR[q][j] < MINLLR)
				outLLR[q][j] = MINLLR;
		}

		for (int q = 1; q < qnumber; ++q)
		{
			outLLR[q][j] = outLLR[q][j] - outLLR[0][j];
		}
		outLLR[0][j] = 0;



		for (int q = 0; q < qnumber; ++q)
		{
			outLLR[q][j] += inLLR[q][j];
		}
	}
	

	/*for (int j = 0; j < cols; ++j)
	{
		for (int q = 1; q < qnumber; ++q)
		{
			EdgeLLR[q] = inLLR[q][j];
		}
		ps = M.chead[j];
		while (ps != NULL)
		{
			for (int q = 1; q < qnumber; ++q)
			{
				EdgeLLR[q] = EdgeLLR[q] + ps->LLR[q];
			}
			ps = ps->down;
		}

		for (int q = 0; q < qnumber; ++q)
		{
			outLLR[q][j] = EdgeLLR[q];
		}
	}*/

	delete[]buffer;
	delete[]EdgeLLR;
	delete[]LGsgn2;
	for (int i = 0; i < max_chk_deg; ++i)
	{
		delete[]LGsgn1[i];
	}
	delete[]LGsgn1;
	return validCode;
}