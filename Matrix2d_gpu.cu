#pragma once
//#include "Matrix2d_gpu.cuh"
#include "host_defines.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>
//#include "Matrix2d_gpu.h"

#define Td 32

namespace Matrix2d_gpu {
	template<typename T> void init(T** res, int len) {
		if (*res != NULL) {
			cudaFree(*res);
		}
		cudaMalloc(res, sizeof(T) * len);
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("init : %s\n", cudaGetErrorString(error));
		}
	}

	template<typename T> void to_gpu(T** res, int len) {
		T* temp = NULL;
		cudaMalloc((void**)&temp, sizeof(T) * len);
		cudaMemcpy(temp, *res, sizeof(T) * len, cudaMemcpyHostToDevice);
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("to_gpu : %s\n", cudaGetErrorString(error));
		}
		*res = temp;
	}

	template<typename T> void to_cpu(T** res, int len) {
		T* temp = (T*)std::malloc(sizeof(T) * len);
		cudaMemcpy(temp, *res, sizeof(T) * len, cudaMemcpyDeviceToHost);
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("to_cpu : %s\n", cudaGetErrorString(error));
		}
		*res = temp;
	}

	template<typename T> __global__ void add_kernel(T* res, T* a, T* b, int h, int w) {
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockDim.y * blockIdx.y + threadIdx.y;
		if (i < h && j < w) {
			res[i * w +j] = a[i * w + j] + b[i * w + j];
		}
	}
	template<typename T> void add(T* res, T* a, T* b, int h, int w) {
		cudaEvent_t startCuda, stopCuda;
		dim3 threadcc(Td, Td);
		dim3 blockcc((h - 1) / Td + 1, (w - 1) / Td + 1);
		time_t st = clock();
		add_kernel << <blockcc, threadcc>> > (res, a, b, h, w);
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("add : %s\n", cudaGetErrorString(error));
		}
		time_t ed = clock();
		std::cout << "Matrix add Matrix(GPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
	}

	template<typename T> __global__ void add_num_kernel(T* res, T* a, T num, int h, int w) {
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockDim.y * blockIdx.y + threadIdx.y;
		if (i < h && j < w) {
			res[i * w + j] = a[i * w + j] + num;
		}
	}
	template<typename T> void add_num(T* res, T* a, T num, int h, int w) {
		dim3 threadcc(Td, Td);
		dim3 blockcc((h - 1) / Td + 1, (w - 1) / Td + 1);
		time_t st = clock();
		add_num_kernel << <blockcc, threadcc >> > (res, a, num, h, w);
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("add_num : %s\n", cudaGetErrorString(error));
		}
		time_t ed = clock();
		std::cout << "Matrix add Num(GPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
	}

	template<typename T> __global__  void assign_kernel(T* res, T* a, int len) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < len) {
			res[i] = a[i];
		}
	}
	template<typename T> void assign(T* res, T* a, int len) {
		dim3 threadcc(Td * Td, 1);
		dim3 blockcc((len - 1) / (Td * Td) + 1, 1);
		assign_kernel << <blockcc, threadcc >> > (res, a, len);
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("assign : %s\n", cudaGetErrorString(error));
		}
	}

	template<typename T> __global__ void mult_kernel(T* res, T* a, T* b, int n, int k, int m) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i < n && j < m) {
			res[i * m + j] = 0;
			for (int p = 0; p < k; ++p) {
				res[i * m + j] += a[i * k + p] * b[p * m + j];
			}
		}
	}
	template<typename T> void mult(T* res, T* a, T* b, int h, int k, int w) {
		dim3 threadcc(Td, Td);
		dim3 blockcc((h - 1) / Td + 1, (w - 1) / Td + 1);
		time_t st = clock();
		mult_kernel << <blockcc, threadcc>> > (res, a, b, h, k, w);
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("mult : %s\n", cudaGetErrorString(error));
		}
		time_t ed = clock();
		std::cout << "Matrix mult Matrix(GPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
	}

	template<typename T> __global__ void mult_num_kernel(T* res, T* a, T num, int h, int w) {
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		int j = blockDim.y * blockIdx.y + threadIdx.y;
		if (i < h && j < w) {
			res[i * w + j] = a[i * w + j] * num;
		}
	}
	template<typename T> void mult_num(T* res, T* a, T num, int h, int w) {
		dim3 threadcc(Td, Td);
		dim3 blockcc((h - 1) / Td + 1, (w - 1) / Td + 1);
		time_t st = clock();
		mult_num_kernel << <blockcc, threadcc >> > (res, a, num, h, w);
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("mult_num : %s\n", cudaGetErrorString(error));
		}
		time_t ed = clock();
		std::cout << "Matrix mult Num(GPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
	}

	template<typename T> __global__ void sum_kernel(T* res, int len) {
		unsigned int t = threadIdx.x;
		__shared__ double partialSum[Td * Td];
		if (blockIdx.x * blockDim.x + t < len)
			partialSum[t] = res[blockIdx.x * blockDim.x + t];
		else
			partialSum[t] = 0;
		__syncthreads();  
		for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
			if (t % (stride<<1) == 0)
				partialSum[t] += partialSum[t + stride];
			__syncthreads();
		}
		if (t == 0)
			res[blockIdx.x * blockDim.x + t] = partialSum[t];
	}
	template<typename T> T sum(T* a, int len) {
		T* temp;
		cudaMallocManaged((void**)&temp, sizeof(T) * len);
		cudaMemcpy(temp, a, sizeof(T) * len, cudaMemcpyDeviceToDevice);
		dim3 threadcc(Td * Td, 1);
		dim3 blockcc((len - 1) / (Td * Td) + 1, 1);
		time_t st = clock();
		sum_kernel << <(len - 1) / (Td * Td) + 1 , Td * Td >> > (temp, len);
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("sum : %s\n", cudaGetErrorString(error));
		}
		time_t ed = clock();
		std::cout << "Matrix sum(GPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
		//T* res = (T*)std::malloc(sizeof(T));
		//cudaMemcpy(res, temp, sizeof(T), cudaMemcpyDeviceToHost);
		T res = T(0);
		int blc = (len - 1) / (Td * Td) + 1;
		for (int i = 0; i < blc; ++i) {
			res += temp[i * Td * Td];
		}
		cudaFree(temp);
		return res;
	}

	template<typename T> __global__ void transpose_kernel(T* res, T* a, int h, int w) {
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		int j = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < h && j < w) {
			res[i * w + j] = a[j * h + i];
		}
	}
	template<typename T> void transpose(T* res, T* a, int h, int w) {
		dim3 threadcc(Td, Td);
		dim3 blockcc((h - 1) / Td + 1, (w - 1) / Td + 1);
		time_t st = clock();
		transpose_kernel << <blockcc, threadcc >> > (res, a, h, w);
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("transpose : %s\n", cudaGetErrorString(error));
		}
		time_t ed = clock();
		std::cout << "Matrix transpose(GPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
	}

	template<typename T> __global__ void cov_kernel(T* res, T* a, T* b, int h1, int w1, int h2, int w2) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		if (row < h1 - h2 + 1 && col < w1 - w2 + 1) {
			float temp = 0;
			int startcol = col;
			int startrow = row;
			for (int i = 0; i < h2; ++i) {
				for (int j = 0; j < w2; ++j) {
					int currow = startrow + i;
					int curcol = startcol + j;
					if (currow > -1 && currow < h1 && curcol> -1 && curcol < w1) {
						temp += b[i * w2 + j] * a[currow * w1 + curcol];
					}
				}
			}
			res[row * (w1 - w2 + 1) + col] = temp;
		}
	}
	template<typename T> void cov(T* res, T* a, T* b, int h1, int w1, int h2, int w2) {
		dim3 threadcc(Td, Td);
		dim3 blockcc((h1 - h2 + 1) / Td + 1, (w1 - w2 + 1) / Td + 1);
		time_t st = clock();
		cov_kernel << <blockcc, threadcc >> > (res, a, b, h1, w1, h2, w2);
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("cov : %s\n", cudaGetErrorString(error));
		}
		time_t ed = clock();
		std::cout << "Matrix cov(GPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
	}

	template<typename T> __global__ void LU_kernel(T* res, int n, int row) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		if (col < n) {
			T num = T(0);
			int m = col;
			if (row < col) {
				m = row;
			}
			for (int i = 0; i < m; ++i) {
				num += res[row * n + i] * res[i * n + col];
			}
			res[row * n + col] -= num;
			if (row > col) {
				res[row * n + col] /= res[col * n + col];
			}
		}
	}
	template<typename T> __global__ void L_kernel(T* res, T* lu, int n) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < n) {
			for (int j = 0; j < i; ++j) {
				res[i * n + j] = lu[i * n + j];
			}
			for (int j = i + 1; j < n; ++j) {
				res[i * n + j] = T(0);
			}
			res[i * n + i] = T(1);
		}
	}
	template<typename T> __global__ void U_kernel(T* res, T* lu, int n) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < n) {
			for (int j = 0; j < i; ++j) {
				res[i * n + j] = T(0);
			}
			for (int j = i; j < n; ++j) {
				res[i * n + j] = lu[i * n + j];
			}
		}
	}
	template<typename T> void LU(T* res, int n, T* L, T* U) {
		time_t st = clock();
		for (int i = 0; i < n; ++i) {
			LU_kernel << <(n * n - 1) / (Td * Td) + 1, Td * Td >> > (res, n, i);
			cudaDeviceSynchronize();
		}
		cudaError_t error = cudaGetLastError();
		if (error) {
			printf("LU : %s\n", cudaGetErrorString(error));
		}
		time_t ed = clock();
		std::cout << "Matrix LU(GPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
		if (L != NULL) {
			L_kernel << <(n * n - 1) / (Td * Td) + 1, Td * Td >> > (L, res, n);
		}
		if (U != NULL) {
			U_kernel << <(n * n - 1) / (Td * Td) + 1, Td * Td >> > (U, res, n);
		}
	}
	
	template<typename T> __global__ void det_kernel(T* res, T* a, int n) {
		*res = T(1);
		for (int i = 0; i < n; ++i) {
			*res *= a[i * n + i];
		}
	}
	template<typename T> T det(T* a, int n) {
		T* temp = NULL;
		cudaMalloc((T**)&temp, sizeof(T));
		det_kernel << <1, 1 >> > (temp, a, n);
		T* res = (T*)std::malloc(sizeof(T));
		cudaMemcpy(res, temp, sizeof(T), cudaMemcpyDeviceToHost);
		cudaFree(temp);
		return *res; 
	}

	template<typename T> __global__ void inverse_dag_kernel(T* mati, T* mat, int n) {
		for (int i = 0; i < n; ++i) {
			mati[i * n + i] = T(1) / mat[i * n + i];
		}
	}
	template<typename T> __global__ void inverse_U_kernel(T* mati, T* mat, int n) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		for (int j = 0; j < i; ++j) {
			mati[i * n + j] = 0;
		}
		for (int j = i + 1; j < n; ++j) {
			mati[i * n + j] = -mati[j * n + j] * mat[i * n + j];
		}
	}
	template<typename T> __global__ void inverse_L_kernel(T* mati, T* mat, int n) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		for (int j = 0; j < i; ++j) {
			mati[i * n + j] = -mat[i * n + j];
		}
		mati[i * n + i] = T(1);
		for (int j = i + 1; j < n; ++j) {
			mati[i * n + j] = T(0);
		}
	}
	template<typename T> void inverse(T* res, T* L, T* U, int n) {
		T *Li, *Ui;
		cudaMalloc((void**)&Li, sizeof(T) * n * n);
		cudaMalloc((void**)&Ui, sizeof(T) * n * n);
		inverse_dag_kernel << <1, 1 >> > (Ui, U, n);
		inverse_L_kernel << <(n * n - 1) / (Td * Td) + 1, Td* Td >> > (Li, L, n);
		inverse_U_kernel << <(n * n - 1) / (Td * Td) + 1, Td* Td >> > (Ui, U, n);
		dim3 threadblock(n, n);
		mult_kernel << <1, threadblock >> > (res, Ui, Li, n, n, n);
		cudaFree(Li);
		cudaFree(Ui);
	}


	//template void init<float>(float**, int);
	//template void to_gpu<float>(float**, int);
	//template void to_cpu<float>(float**, int);
	//template void add<float>(float*, float*, float*, int, int);
	//template void add_num<float>(float*, float*, float, int, int);
	//template void assign<float>(float*, float*, int);
	//template void mult<float>(float*, float*, float*, int, int, int);
	//template void mult_num<float>(float*, float*, float, int, int);
	//template float sum<float>(float*, int);
	//template void transpose<float>(float*, float*, int, int);
	//template void cov<float>(float*,  float*, float*, int, int, int, int);
	//template void LU<float>(float*, int, float*, float*);
	//template float det<float>(float*, int);
	//template void inverse<float>(float*, float*, float*, int);

	template void init<double>(double**, int);
	template void to_gpu<double>(double**, int);
	template void to_cpu<double>(double**, int);
	template void add<double>(double*, double*, double*, int, int);
	template void add_num<double>(double*, double*, double, int, int);
	template void assign<double>(double*, double*, int);
	template void mult<double>(double*, double*, double*, int, int, int);
	template void mult_num<double>(double*, double*, double, int, int);
	template double sum<double>(double*, int);
	template void transpose<double>(double*, double*, int, int);
	template void cov<double>(double*, double*, double*, int, int, int, int);
	template void LU<double>(double*, int, double*, double*);
	template double det<double>(double*, int);
	template void inverse<double>(double*, double*, double*, int);
}