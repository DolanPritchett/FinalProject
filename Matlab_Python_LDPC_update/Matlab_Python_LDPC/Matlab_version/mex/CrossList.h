#ifndef H_CROSSLIST
#define H_CROSSLIST
#include "gf.hpp"

struct OLNode {
	int row_num, col_num;  //the subscript of row and col of non_zero elements
	double *qmn;
	double *Qmn;
	double *QmnTemp;
	double *rmn;
	double *LLR;
	struct OLNode *right, *down;
	struct OLNode *left, *up;
	gfe ele;
	bool Status;
	int pattern;
};

typedef struct CrossList {
	struct OLNode **rhead, **chead;
	struct OLNode **rtail, **ctail;
}CrossList;

#endif
