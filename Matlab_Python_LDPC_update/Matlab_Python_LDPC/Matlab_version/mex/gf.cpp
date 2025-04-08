#include "gf.hpp"
#include <iostream>
using namespace std;

void Generate_GF(int order)
/* generate GF(2**order) from the irreducible polynomial p(X) in pp[0]..pp[mm]
lookup tables:  index->polynomial form   alpha_to[] contains j=alpha**i;
polynomial form -> index form  index_of[j=alpha**i] = i
alpha=2 is the primitive element of GF(2**order)
*/
{
	for (int k = 0; k != gfdnum[elementBits]; ++k)
	{
		alpha_to.push_back(0);
		index_of.push_back(0);
	}
	int i, mask;
	const int reducetable[] = { 3, 3, 3, 5, 3, 9, 29, 17, 9, 5, 83, 27, 43 };
	int qvalue = 1 << order;
	for (i = 0; i < qvalue; i++)
	{
		alpha_to[i] = 0;
		index_of[i] = 0;
	}
	if (order == 1)
	{
		alpha_to[0] = 1;
		index_of[0] = -1;
		index_of[1] = 0;
	}
	else
	{
		mask = 1;
		int reduce = reducetable[order - 2];
		alpha_to[order] = 0;
		for (i = 0; i < order; i++)
		{
			alpha_to[i] = mask;
			index_of[alpha_to[i]] = i;
			if ((reduce & (1 << i)) != 0)
				alpha_to[order] ^= mask;
			mask <<= 1;
		}
		index_of[alpha_to[order]] = order;
		mask >>= 1;
		for (i = order + 1; i < qvalue - 1; i++)
		{
			if (alpha_to[i - 1] >= mask)
				alpha_to[i] = alpha_to[order] ^ ((alpha_to[i - 1] ^ mask) << 1);
			else
				alpha_to[i] = alpha_to[i - 1] << 1;
			index_of[alpha_to[i]] = i;
		}
		index_of[0] = -1;
	}
}
void Gen_poly(int *gen_poly, int codeLen, int infoLen)
/* Obtain the generator polynomial of the tt-error correcting, length
codeLen = (2**order -1) Reed Solomon code  from the product of (X+alpha**i), i=1..2*tt
*/
{
	int i, j;
	gen_poly[0] = 2;    /* primitive element alpha = 2  for GF(2**order)  */
	gen_poly[1] = 1;    /* g(x) = (X+alpha) initially */
	for (i = 2; i <= codeLen - infoLen; i++)
	{
		gen_poly[i] = 1;
		for (j = i - 1; j > 0; j--)
			if (gen_poly[j] != 0)
				gen_poly[j] = gen_poly[j - 1] ^ alpha_to[(index_of[gen_poly[j]] + i) % codeLen];//取模是对alpha**codeLen进行的
			else
				gen_poly[j] = gen_poly[j - 1];
		gen_poly[0] = alpha_to[(index_of[gen_poly[0]] + i) % codeLen];     /* gg[0] can never be zero */
	}
	/* convert gg[] to index form for quicker encoding */
	for (i = 0; i <= codeLen - infoLen; i++)
		gen_poly[i] = index_of[gen_poly[i]];
}
void Encode_RS(int *CodeWord, int *Mesg, int *gen_poly, int codeLen, int infoLen)
/* take the string of symbols in Mesg[i], i=0..(k-1) and encode systematically
to produce 2*tt parity symbols in CodeWord[0]..CodeWord[2*tt-1]
Mesg[] is input and CodeWord[] is output in polynomial form.
Encoding is done by using a feedback shift register with appropriate
connections specified by the elements of gen_poly[], which was generated above.
Codeword is   c(X) = Mesg(X)*X**(codeLen-infoLen)+ CodeWord(X)          */
{
	int i, j;
	int feedback;
	for (i = 0; i < codeLen - infoLen; i++)
		CodeWord[i] = 0;
	for (i = 0; i < infoLen; i++)
		CodeWord[i + codeLen - infoLen] = Mesg[i];
	for (i = infoLen - 1; i >= 0; i--)
	{
		feedback = index_of[Mesg[i] ^ CodeWord[codeLen - infoLen - 1]];//当data[i]^bb[nn-kk-1＝0时，feedback=-1；
		if (feedback != -1)//there exists feedback
		{
			for (j = codeLen - infoLen - 1; j > 0; j--)
				if (gen_poly[j] != -1)//gg[j] = -1, means that the coefficient of x**j is 0;
					CodeWord[j] = CodeWord[j - 1] ^ alpha_to[(gen_poly[j] + feedback) % codeLen];
				else
					CodeWord[j] = CodeWord[j - 1];
			CodeWord[0] = alpha_to[(gen_poly[0] + feedback) % codeLen];
		}
		else//no feedback, just shift
		{
			for (j = codeLen - infoLen - 1; j > 0; j--)
				CodeWord[j] = CodeWord[j - 1];
			CodeWord[0] = 0;
		}
	}
}
void genGF(vector<gfe> &A)
{
	for (size_t k = 0; k != gfdnum[elementBits]; ++k)
		A.push_back(0);
	A[0].memory = 1;
	A[gfdnum[elementBits] - 1].memory = 0;
	for (int i = 1; i != gfdnum[elementBits] - 1; i++)
	{
		if (A[i - 1].memory & (1 << (elementBits - 1)))
			A[i] = (A[i - 1] << 1) + p[elementBits];
		else
			A[i] = A[i - 1] << 1;
	}
}
void genGFindex(vector<int> &B)
{
	B.clear();
	for (size_t i = 0; i != gfdnum[elementBits]; ++i)
		B.push_back(0);
	for (size_t i = 0; i != gfdnum[elementBits]; ++i)
		B[GF[i].memory] = i;
}
gfe operator<<(const gfe &a, const int &b)
{
	/*
	int temp = a.memory << b;
	bitsetint c;
	for( size_t j = 0; j != elementBits; ++ j )
	{
	if( ( temp & ( 1<< j) ) != 0 )
	c.memory = c.memory + ( 1<< j );
	}
	return c;
	*/
	int temp = a.memory << b;
	gfe c;
	c = temp & (gfdnum[elementBits] - 1);
	return c;
}
gfe operator+(const gfe &a, const gfe &b)
{
	gfe c;
	c.memory = a.memory ^ b.memory;
	return c;
}
gfe operator+(const gfe &a, const int &b)
{
	int temp = b & (gfdnum[elementBits] - 1);
	gfe c;
	c.memory = a.memory ^ temp;
	return c;
}
gfe operator+(const int &b, const gfe &a)
{
	int temp = b & (gfdnum[elementBits] - 1);
	gfe c;
	c.memory = a.memory ^ b;// Before I made a change, it had been " c.memory = a.memory + b "
	return c;
}
int operator&(const gfe &a, const int &b)
{
	return (a.memory & b);
}
gfe operator*(const gfe &a, const gfe &b)
{
	/*
	//the idea is from the book 《Error Control Coding》，in page 146
	gfe c;
	for( int i = elementBits - 1; i != 0; i -- )
	{
	if( b.memory & (1<<i) )
	c = c + a;
	if( c.memory & ( 1<<(elementBits-1) ) )
	c = (c<<1)+p[elementBits];
	else
	c = c<<1;
	}
	if( b.memory & (1<<0) )
	c = c + a;
	return c;
	*/
	/*
	bitsetint c( 0 );
	if( getValue(a) == 0 || getValue(b) == 0 )
	return c;
	else
	{
	c = GFD[( GFindex[getValue( a )] + GFindex[getValue( b )] ) % ( gfdnum[elementBits] - 1 )];
	return c;
	}
	*/
	gfe c(0);
	if (a.getValue() == 0 || b.getValue() == 0)
		return c;
	else
	{
		c = GF[(GFindex[a.getValue()] + GFindex[b.getValue()]) % (gfdnum[elementBits] - 1)];
		return c;
	}

}
gfe operator^(const gfe &a, const int &b)
{
	/*
	//the idea is elaborated below;
	//a^7 in dec = a^(111) in binary
	gfe temp = a;
	gfe ans = 1;
	int c = 1;
	while ( c <= b )
	{
	if( (b & c) != 0 )
	{
	ans = temp * ans;
	}
	temp = temp * temp;
	c = c << 1;
	}
	return ans;
	*/
	/*
	bitsetint c;
	if( getValue(a) == 0 )
	return c;
	else
	{
	int cindex = ( GFindex[getValue(a)] * b ) % ( gfdnum[elementBits] - 1 );
	c = GFD[cindex];
	return c;
	}
	*/
	gfe c(0);
	if (a.getValue() == 0)
		return c;
	else
	{
		int cindex = (GFindex[a.getValue()] * b) % (gfdnum[elementBits] - 1);
		c = GF[cindex];
		return c;
	}
}
bool operator!=(const gfe &a, int b)
{
	if (a.memory != b)
		return true;
	else
		return false;
}
bool operator!=(const gfe &a, const gfe &b)
{
	if (a.memory != b.memory)
		return true;
	else
		return false;
}
bool operator==(const gfe &a, int b)
{
	return !(a != b);
}
bool operator==(const gfe &a, const gfe &b)
{
	return !(a != b);
}
