#ifndef GF_H
#define GF_H

#include <vector>
using namespace std;
#include "gfe.hpp"

extern vector<gfe> GF;
extern vector<int> GFindex;
extern vector<int> alpha_to;
extern vector<int> index_of;

void Generate_GF(int order);
void Gen_poly(int *gen_poly, int codeLen, int infoLen);
void Encode_RS(int *CodeWord, int *Mesg, int *gen_poly, int codeLen, int infoLen);

void genGF(vector<gfe> &A);
void genGFindex(vector<int> &B);

gfe operator+(const gfe &a, const gfe &b);
gfe operator+(const gfe &a, const int &b);
gfe operator+(const int &b, const gfe &a);
int operator&(const gfe &a, const int &b);
gfe operator*(const gfe &a, const gfe &b);
gfe operator^(const gfe &a, const int &b);
gfe operator<<(const gfe &a, const int &b);
bool operator!=(const gfe &a, int b);
bool operator!=(const gfe &a, const gfe &b);
bool operator==(const gfe &a, int b);
bool operator==(const gfe &a, const gfe &b);



#endif
