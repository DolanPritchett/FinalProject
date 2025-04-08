#include "function.hpp"
#include <iostream>
using namespace std;

void ArrayMultiply(int *res, const int *a, const int* b, int n, int l)
/**
a: m x n
b: n x l
res: m x l
*/
{

	//static int add[4][4]={{0, 1, 2, 3},{1, 0, 3, 2},{2, 3, 0, 1},{3, 2, 1, 0}};
	//static int mul[4][4]={{0, 0, 0, 0},{0, 1, 2, 3},{0, 2, 3, 1},{0, 3, 1, 2}};
	int j, k;

	//for (i=0; i<m; i++)
	for (j = 0; j<l; j++)
	{
		gfe s = 0;
		for (k = 0; k<n; k++)
			s = s + (GF[GFindex[a[k]]] * GF[GFindex[b[k*l + j]]]); // should be checked whether it is correct

		res[j] = s.getValue();
	} // for j, i

}

void ArrayMultiply(int *res, const int *a, const unsigned char* b, int n, int l)
/**
a: m x n
b: n x l
res: m x l
*/
{

	//static int add[4][4]={{0, 1, 2, 3},{1, 0, 3, 2},{2, 3, 0, 1},{3, 2, 1, 0}};
	//static int mul[4][4]={{0, 0, 0, 0},{0, 1, 2, 3},{0, 2, 3, 1},{0, 3, 1, 2}};
	int j, k;

	//for (i=0; i<m; i++)
	for (j = 0; j<l; j++)
	{
		gfe s = 0;
		for (k = 0; k < n; k++)
		{
			s = s + (GF[GFindex[a[k]]] * GF[GFindex[(int)b[k * l + j]]]); // should be checked whether it is correct
		}

		res[j] = s.getValue();
	} // for j, i

}
