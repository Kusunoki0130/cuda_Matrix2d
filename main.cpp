#include <iostream>
#include "Matrix2d.h"

int n = 5000;
int k = 20;
const int N = 1e8 + 8;
double d1[N] = { 0 };
double d2[N] = { 0 };
extern "C" void test() {
	for (int i = 0; i < n; ++i) {
		d1[i * n] = 1 + i * (double)10 / n;
		d2[i * n] = 2 - i * (double)10 / n;
		for (int j = 1; j < n; ++j) {
			d1[i * n + j] = d1[i * n + j - 1] + (double)10 / n;
			d2[i * n + j] = d2[i * n + j - 1] - (double)10 / n;
		}
	}
	//double d1[16] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	//double d2[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
	//double d3[100] = { 0 };
	//for (int i = 0; i < 10; ++i) {
	//	for (int j = 0; j < 10; ++j) {
	//		d3[i * 10 + j] = 1;
	//	}
	//}
	//for (int i = 0; i < 10; ++i) {
	//	std::cout << i + 1 << " ";
	//	Matrix2d<double> mat3(n, n, d1, false);
	//	// Matrix2d<double> mat4(k, k, d2, false);
	//	mat3.LU();
	//}
	//std::cout << std::endl;
	for (int i = 0; i < 10;  ++i) {
		std::cout << i + 1 << " ";
		Matrix2d<double> mat1(n, n, d1, true);
		// Matrix2d<double> mat2(k, k, d2, true);
		mat1.LU();
	}

	// 初试化参数：矩阵高，矩阵宽，一维数组，CPU or GPU
	Matrix2d<double> A(10, 7, d1, true);
	Matrix2d<double> B(7, 8, d2, true);
	Matrix2d<double> C = A * B;
	std::cout << C << std::endl;
	
	//mat3.LU();
}

int main() {
	test();
	return 0;
}