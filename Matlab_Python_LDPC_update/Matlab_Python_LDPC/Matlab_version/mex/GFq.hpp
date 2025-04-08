#ifndef GFQ_H
#define GFQ_H

extern const int log4[4]; /* stores i at address \alpha^i */
extern const int log8[8];
extern const int log16[16];
extern const int log32[32];
extern const int log64[64];
extern const int log128[128];
extern const int log256[256];

extern const int exp4[3]; /* stores \alpha^i at address i */
extern const int exp8[7];
extern const int exp16[15];
extern const int exp32[31];
extern const int exp64[63];
extern const int exp128[127];
extern const int exp256[255];


int GFq_m(int a, int b, int q);

int GFq_inv(int a, int q);

int GFq_a(int a, int b);

void permute(double* input, double* buffer, int shift, int Qary);

void permute2(double* input, double* buffer, int shift, int Qary);

void permute_add(double* input, double* buffer, int shift, int Qary);

void permute_add2(double* input, double* output, int shift, int Qary);

void GFq_conv(double *dest,  double *src1, double *src2, int Qary);

#endif

