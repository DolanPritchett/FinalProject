#ifndef GFE_H
#define GFE_H

#include "parameter.hpp"

const int gfdnum[12] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };
const int p[12] = { 1, 1, 3, 3, 3, 5, 3, 9, 29, 17, 9, 5 };

class gfe
{
public:
	gfe() :memory(0) {}
	gfe(int value) :memory(value) {}
	~gfe() {}
	int getValue() const
	{
		return memory;
	}
	gfe& operator=(const gfe &a)
	{
		memory = a.memory;
		return *this;
	}
	gfe& operator=(const int &b)
	{
		memory = b;
		return *this;
	}

	//friend int getValue( const gfe &a )
	//{ return a.memory; }


public:
	int memory;
};

#endif
