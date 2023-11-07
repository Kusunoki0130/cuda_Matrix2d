#pragma once
#include <iostream>
#include <time.h>
#ifndef __MATRIX2D_CPU_H__
#define __MATRIX2D_CPU_H__
namespace Matrix2d_cpu {
	template<typename T> void init(T**, int);
	template<typename T> void assign(T*, T*, int);
	template<typename T> void add(T*, T*, T*, int);
	template<typename T> void add_num(T*, T*, T, int);
	template<typename T> void mult(T*, T*, T*, int, int, int );
	template<typename T> void mult_num(T*, T*, T, int);
	template<typename T> T sum(T*, int);
	template<typename T> void transpose(T*, T*, int, int);
	template<typename T> void cov(T*, T*, T*, int, int, int, int);
	template<typename T> void LU(T*, int, T* L=NULL, T* U=NULL);
	template<typename T> T det(T*, int);
	template<typename T> void inverse(T*, T*, T*, int);
}
#endif


