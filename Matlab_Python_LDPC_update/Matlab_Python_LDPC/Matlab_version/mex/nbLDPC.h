#ifndef H_NBLDPC
#define H_NBLDPC

#include "CrossList.h"
#include "gf.h"
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

#define MAXLLRTEMP 25.0
#define MINPROB 1e-80

struct col_permutation
{
	int col_1;
	int col_2;
};

class nbLDPC {
public:
	// input: H matrix row number
	//        H matrix col number
	//        CPM size and q-ary
	nbLDPC();
	nbLDPC(int, int, int);
	~nbLDPC();
	void set(int, int, int);
	void input_infoLen(int infoLength) { infoLen = infoLength; }
	bool HasNonZeroInTheRow(int i);
	void rearrangeXS();
	void gen_G();
	void Encoder(int * Mesg, int *CodeWord);
	void Reorder_bits(int *u);
	void order_bits(int *ref);
	void Reorder_bits(int *u, int start);
	void order_bits(int *ref, int start);
	void update();
	void extract_mesg(int *res, int *ref);
	void extract_mesg(double **res, double **ref);
	int CreatMatrix_OL_nonbinary(char *s);
	int CreatMatrix_OL_nonbinary(string &s);
	int CreatMatrix_OL_nonbinary_alist(char *s);
	int CreatMatrix_OL_nonbinary_alist_v2(char *s);
	int CreatMatrix_OL_nonbinary_Neal(char *s);
	int CreatMatrix_OL_nonbinary_ldspec(char *s);
	int CreatMatrix_OL_nonbinary_FromDrLi(char *s);
	bool judgeZero(int *vhat);
	void sort(double *v, gfe ele, int qnumber);
	void permute(double *input, double *buffer, gfe ele, int qnumber);
	void permute(double *buffer);
	void fft(double* v, double *V, int qnumber);
	void ifft(double*V, double *v, int qnumber);
	void FFT(double* res, double *ref, int qnumber);
	void IFFT(double *res, double *ref, int qnumber);
	int FFT_QSPA(double **f, double **Qn, int *vhat, int *CodeWord, int MAXITRNUM);
	int FFT_QSPA_log(double **inLLR, double **outLLR, int *vhat, int MAXITRNUM);
	CrossList  M;
	int destroy_crossList();
public:
	// the message length and the code length
	// they are the final version after eliminating
	// the redundant rows
	int infoLen, codeLen;

	// the original H matrix' rows and columns
	int ROWS, cols;
	// the rows of H matrix after eliminating redundant rows
	int now_rows;

	int numofnonzeroElem;

	// q-ary
	int qnumber;
	unsigned char *G;
	unsigned char *H;
	unsigned char *rearrange;

	vector<col_permutation> permutation_nodes;
};



#endif