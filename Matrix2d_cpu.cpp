#pragma
#include "Matrix2d_cpu.h"
#include <iostream>

namespace Matrix2d_cpu {
	template<typename T>
	void init(T** res, int len) {
		if (*res != NULL) {
			free(*res);
		}
		*res = (T*)malloc(sizeof(T) * len);
	}

	template<typename T>
	void add(T* res, T* a, T* b, int len) {
		time_t st = clock();
		for (int i = 0; i < len; ++i) {
			res[i] = a[i] + b[i];
		}
		time_t ed = clock();
		std::cout << "Matrix add Matrix(CPU): " << 1000 * (ed - st)/ CLOCKS_PER_SEC << " ms" << std::endl;
	}

	template<typename T>
	void add_num(T* res, T* a, T num, int len) {
		time_t st = clock();
		for (int i = 0; i < len; ++i) {
			res[i] = a[i] + num;
		}
		time_t ed = clock();
		std::cout << "Matrix add Num(CPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
	}


	template<typename T>
	void assign(T* res, T* a, int len) {
		for (int i = 0; i < len; ++i) {
			res[i] = a[i];
		}
	}

	// O(n^3)
	template<typename T>
	void mult(T* res, T* a, T* b, int n, int k, int m) {
		time_t st = clock();
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				res[i * m + j] = 0;
				for (int p = 0; p < k; ++p) {
					res[i * m + j] += a[i * k + p] * b[p * m + j];
				}
			}
		}
		time_t ed = clock();
		std::cout << "Matrix Mult Matrix(CPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
	}

	template<typename T>
	void mult_num(T* res, T* a, T num, int len) {
		time_t st = clock();
		for (int i = 0; i < len; ++i) {
			res[i] = a[i] * num;
		}
		time_t ed = clock();
		std::cout << "Matrix mult Num(CPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
	}


	template<typename T>
	T sum(T* a, int len) {
		time_t st = clock();
		T res = 0;
		for (int i = 0; i < len; ++i) {
			res += a[i];
		}
		time_t ed = clock();
		std::cout << "Matrix sum(CPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
		return res;
	}

	template<typename T>
	void transpose(T* res, T* a, int h, int w) {
		time_t st = clock();
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				res[i * w + j] = a[j * h + i];
			}
		}
		time_t ed = clock();
		std::cout << "Matrix transpose(CPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
	}

	template<typename T>
	void cov(T* res, T* a, T* b, int h1, int w1, int h2, int w2) {
		time_t st = clock();
		for (int i = 0; i < h1 - h2 + 1; ++i) {
			for (int j = 0; j < w1 - w2 + 1; ++j) {
				res[i * (w1 - w2 + 1) + j] = 0;
				for (int p = 0; p < h2; ++p) {
					for (int q = 0; q < w2; ++q) {
						res[i * (w1 - w2 + 1) + j] += a[(i + p) * w1 + j + q] * b[p * w2 + q];
					}
				}
			}
		}
		time_t ed = clock();
		std::cout << "Matrix cov(CPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
	}

	template<typename T>
	void LU(T* a, int n, T* L, T* U) {
		time_t st = clock();
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				int m = i < j ? i : j;
				for (int k = 0; k < m; ++k) {
					a[i * n + j] -= a[k * n + j] * a[i * n + k];
				}
				if (i > j) {
					a[i * n + j] /= a[j * n + j];
				}
			}
		}
		time_t ed = clock();
		std::cout << "Matrix LU(CPU): " << 1000 * (ed - st) / CLOCKS_PER_SEC << " ms" << std::endl;
		if (L != NULL) {
			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < i; ++j) {
					L[i * n + j] = a[i * n + j];
				}
				for (int j = i + 1; j < n; ++j) {
					L[i * n + j] = T(0);
				}
				L[i * n + i] = T(1);
			}
		}
		if (U != NULL) {
			for (int i = 0; i < n; ++i) {
				for (int j = 0; j < i; ++j) {
					U[i * n + j] = T(0);
				}
				for (int j = i; j < n; ++j) {
					U[i * n + j] = a[i * n + j];
				}
			}
		}
	}

	template<typename T>
	T det(T* a, int n) {
		// LU<T>(a, n);
		T res(1);
		for (int i = 0; i < n; ++i) {
			res *= a[i * n + i];
		}
		return res;
	}

	template<typename T>
	void inverse(T* res, T* L, T* U, int n) {
		T* Li = (T*)malloc(sizeof(T) * n * n);
		T* Ui = (T*)malloc(sizeof(T) * n * n);
		for (int i = 0; i < n; ++i) {
			Ui[i * n + i] = T(1) / U[i * n + i];
			for (int j = 0; j < i; ++j) {
				Li[i * n + j] = -L[i * n + j];
			}
			Li[i * n + i] = 1;
			for (int j = i + 1; j < n; ++j) {
				Li[i * n + j] = 0;
			}
		}
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				Ui[i * n + j] = 0;
			}
			for (int j = i + 1; j < n; ++j) {
				Ui[i * n + j] = -Ui[j * n + j] * U[i * n + j];
			}
		}
		mult(res, Ui, Li, n, n, n);
		free(Li);
		free(Ui);
	}

	//template void init<float>(float**, int);
	//template void add<float>(float*, float*, float*, int);
	//template void assign<float>(float*, float*, int);
	//template void add_num<float>(float*, float*, float, int);
	//template void mult<float>(float*, float*, float*, int, int, int);
	//template void mult_num<float>(float*, float*, float, int);
	//template float sum<float>(float*, int);
	//template void transpose<float>(float*, float*, int, int);
	//template void cov<float>(float*, float*, float*, int, int, int, int);
	//template void LU<float>(float*, int, float*, float*);
	//template float det<float>(float*, int);
	//template void inverse<float>(float*, float*, float*, int);

	template void init<double>(double**, int);
	template void add<double>(double*, double*, double*, int);
	template void assign<double>(double*, double*, int);
	template void add_num<double>(double*, double*, double, int);
	template void mult<double>(double*, double*, double*, int, int, int);
	template void mult_num<double>(double*, double*, double, int);
	template double sum<double>(double*, int);
	template void transpose<double>(double*, double*, int, int);
	template void cov<double>(double*, double*, double*, int, int, int, int);
	template void LU<double>(double*, int, double*, double*);
	template double det<double>(double*, int);
	template void inverse<double>(double*, double*, double*, int);
}