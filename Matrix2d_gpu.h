#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#ifndef __MATRIX2D_GPU_H__
#define __MATRIX2D_GPU_H__
namespace Matrix2d_gpu {
	template<typename T> void init(T**, int);
	template<typename T> void to_gpu(T**, int);
	template<typename T> void to_cpu(T**, int);
	template<typename T> void free(T**);
	template<typename T> void add(T*, T*, T*, int, int);
	template<typename T> void add_num(T*, T*, T, int, int);
	template<typename T> void assign(T*, T*, int);
	template<typename T> void mult(T*, T*, T*, int, int, int);
	template<typename T> void mult_num(T*, T*, T, int, int);
	template<typename T> T sum(T*, int);
	template<typename T> void transpose(T*, T*, int, int);
	template<typename T> void cov(T*, T*, T*, int, int, int, int);
	template<typename T> void LU(T*, int, T* L=NULL, T* U=NULL);
	template<typename T> T det(T*, int);
	template<typename T> void inverse(T*, T*, T*, int);
}
#endif // !__MATRIX2D_GPU_H__