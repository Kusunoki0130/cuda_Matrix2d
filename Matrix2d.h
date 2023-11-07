#pragma once
#include <iomanip>
#include <utility>
#include <iostream>
#include "Matrix2d_cpu.h"
#include "Matrix2d_gpu.h"

template<typename T>
class Matrix2d {
private:
	int h;
	int w;
	T* data;
	bool dev;
	T* L;
	T* U;

public:
	Matrix2d() {};
	Matrix2d(int, int, T*, bool);
	Matrix2d(int, int, bool);
	Matrix2d(Matrix2d<T>&);
	~Matrix2d();

	std::pair<int, int> size() {
		return std::make_pair(this->h, this->w);
	}
	T* get() {
		return data;
	}
	T& get(int i, int j) {
		return data[i * this->w + j];
	}
	bool device() {
		return this->dev;
	}
	void ones();
	T sum();
	T det();
	Matrix2d<T> LU(Matrix2d<T>* L = NULL, Matrix2d<T>* U = NULL);
	Matrix2d<T> cov(Matrix2d<T>& mat);
	Matrix2d<T> i();
	Matrix2d<T> inverse();
	Matrix2d<T> t();
	Matrix2d<T> transpose();

	Matrix2d<T> operator+(Matrix2d<T>& mat);
	Matrix2d<T> operator+(T num);
	Matrix2d<T> operator-(Matrix2d<T>& mat);
	Matrix2d<T> operator-(T num);
	Matrix2d<T> operator-();
	Matrix2d<T> operator*(Matrix2d<T>& mat);
	Matrix2d<T> operator*(T num);
	Matrix2d<T> operator^(int num);


	void operator=(Matrix2d<T>& mat);
	void operator+=(Matrix2d<T>& mat);
	void operator+=(T num);
	void operator-=(Matrix2d<T>& mat);
	void operator-=(T num);
	void operator*=(Matrix2d<T>& mat);
	void operator*=(T num);
	void operator^=(int num);

	friend std::ostream& operator<<(std::ostream& out, const Matrix2d<T>&);
};

template<typename T>
Matrix2d<T>::Matrix2d(int h, int w, T* input, bool device) {
	this->h = h;
	this->w = w;
	this->dev = device;
	this->data = NULL;
	this->L = NULL;
	this->U = NULL;
	if (!this->dev) {
		Matrix2d_cpu::init(&(this->data), this->h * this->w);
		Matrix2d_cpu::assign(this->data, input, this->h * this->w);
	}
	else {
		Matrix2d_gpu::init(&(this->data), this->h * this->w);
		Matrix2d_gpu::to_gpu(&input, this->h * this->w);
		Matrix2d_gpu::assign(this->data, input, this->h * this->w);
		cudaFree(input);
	}
}

template<typename T>
Matrix2d<T>::Matrix2d(Matrix2d<T>& mat) {
	std::pair<int, int> mat_size = mat.size();
	this->h = mat_size.first;
	this->w = mat_size.second;
	this->data = NULL;
	this->L = NULL;
	this->U = NULL;
	this->dev = mat.device();
	if (!this->dev) {
		Matrix2d_cpu::init(&(this->data), this->h * this->w);
		Matrix2d_cpu::assign(this->data, mat.get(), this->h * this->w);
	}
	else {
		Matrix2d_gpu::init(&(this->data), this->h * this->w);
		Matrix2d_gpu::assign(this->data, mat.get(), this->h * this->w);
	}
}

template<typename T>
Matrix2d<T>::Matrix2d(int h, int w, bool device) {
	this->h = h;
	this->w = w;
	this->dev = device;
	this->data = NULL;
	this->L = NULL;
	this->U = NULL;
	if (!this->dev) {
		Matrix2d_cpu::init(&(this->data), this->h * this->w);
	}
	else {
		Matrix2d_gpu::init(&(this->data), this->h * this->w);
	}
}

template<typename T>
Matrix2d<T>::~Matrix2d() {
	if (!this->dev) {
		free(this->data);
	}
	else {
		cudaFree(this->data);
	}
}


template<typename T>
void Matrix2d<T>::ones() {
	if (this->h != this->w) {
		throw;
	}
	T* temp = NULL;
	Matrix2d_cpu::init(&temp, this->h * this->w);
	for (int i = 0; i < this->h; ++i) {
		for (int j = 0; j < this->w; ++j) {
			temp[i * this->w + j] = T(0);
		}
		temp[i * this->w + i] = T(1);
	}
	if (!this->dev) {
		Matrix2d_cpu::assign(this->data, temp, this->h * this->w);
	}
	else {
		Matrix2d_gpu::to_gpu(&temp, this->h * this->w);
		Matrix2d_gpu::assign(this->data, temp, this->h * this->w);
	}
}


template<typename T>
Matrix2d<T> Matrix2d<T>::operator+(Matrix2d<T>& mat) {
	std::pair<int, int> mat_size = mat.size();
	int mat_h = mat_size.first;
	int mat_w = mat_size.second;
	if (this->h != mat_h || this->w != mat_w) {
		throw;
	}
	T* temp = NULL;
	if (!this->dev) {
		Matrix2d_cpu::init(&temp, this->h * this->w);
		Matrix2d_cpu::add(temp, this->data, mat.get(), this->h * this->w);
	}
	else {
		Matrix2d_gpu::init(&temp, this->h * this->w);
		Matrix2d_gpu::add(temp, this->data, mat.get(), this->h, this->w);
	}
	Matrix2d<T> ret(this->h, this->w, temp, this->dev);
	return ret;
}

template<typename T>
Matrix2d<T> Matrix2d<T>::operator+(T num) {
	int mat_h = this->h;
	int mat_w = this->w;
	T* temp = NULL;
	if (!this->device()) {
		Matrix2d_cpu::init(&temp, mat_h * mat_w);
		Matrix2d_cpu::add_num(temp, this->get(), num, mat_h * mat_w);
	}
	else {
		Matrix2d_gpu::init(&temp, mat_h * mat_w);
		Matrix2d_gpu::add_num(temp, this->get(), num, mat_h, mat_w);
	}
	Matrix2d<T> ret(mat_h, mat_w, temp, this->device());
	return ret;
}

template<typename T>
Matrix2d<T> Matrix2d<T>::operator*(Matrix2d<T>& mat) {
	std::pair<int, int> mat_size = mat.size();
	int mat_h = mat_size.first;
	int mat_w = mat_size.second;
	if (this->w != mat_h) {
		throw;
	}
	T* temp = NULL;
	if (!this->dev) {
		Matrix2d_cpu::init(&temp, this->h * mat_w);
		Matrix2d_cpu::mult(temp, this->data, mat.get(), this->h, this->w, mat_w);
	}
	else {
		Matrix2d_gpu::init(&temp, this->h * mat_w);
		Matrix2d_gpu::mult(temp, this->data, mat.get(), this->h, this->w, mat_w);
	}
	Matrix2d<T> ret(this->h, mat_w, temp, this->dev);
	return ret;
}



template<typename T>
Matrix2d<T> Matrix2d<T>::operator*(T num) {
	T* temp = NULL;
	if (!this->dev) {
		Matrix2d_cpu::init(&temp, this->h * this->w);
		Matrix2d_cpu::mult_num(temp, this->data, num, this->h * this->w);
	}
	else {
		Matrix2d_gpu::init(&temp, this->h * this->w);
		Matrix2d_gpu::mult_num(temp, this->data, num, this->h, this->w);
	}
	Matrix2d<T> ret(this->h, this->w, temp, this->dev);
	return ret;
}


template<typename T>
Matrix2d<T> Matrix2d<T>::operator-() {
	return Matrix2d<T>::operator*((T)-1);
}
template<typename T>
Matrix2d<T> Matrix2d<T>::operator-(Matrix2d<T>& mat) {
	return Matrix2d<T>::operator+(-mat);
}
template<typename T>
Matrix2d<T> Matrix2d<T>::operator-(T num) {
	return Matrix2d<T>::operator+(-num);
}

template<typename T>
void Matrix2d<T>::operator=(Matrix2d<T>& mat) {
	std::pair<int, int> mat_size = mat.size();
	int mat_h = mat_size.first;
	int mat_w = mat_size.second;
	this->h = mat_h;
	this->w = mat_w;
	if (!mat.device()) {
		Matrix2d_cpu::init(&this->data, this->h * this->w);
		Matrix2d_cpu::assign(this->data, mat.get(), this->h * this->w);
	}
	else {
		Matrix2d_gpu::init(&this->data, this->h * this->w);
		Matrix2d_gpu::assign(this->data, mat.get(), this->h * this->w);
	}
}


template<typename T>
void Matrix2d<T>::operator+=(Matrix2d<T>& mat) {
	Matrix2d<T> temp = Matrix2d<T>::operator+(mat);
	Matrix2d<T>::operator=(temp);
}
template<typename T>
void Matrix2d<T>::operator-=(Matrix2d<T>& mat) {
	Matrix2d<T> temp = Matrix2d<T>::operator-(mat);
	Matrix2d<T>::operator=(temp);
}
template<typename T>
void Matrix2d<T>::operator*=(Matrix2d<T>& mat) {
	Matrix2d<T> temp = Matrix2d<T>::operator*(mat);
	Matrix2d<T>::operator=(temp);
}

template<typename T>
void Matrix2d<T>::operator+=(T num) {
	Matrix2d<T> temp = Matrix2d<T>::operator+(num);
	Matrix2d<T>::operator=(temp);
}
template<typename T>
void Matrix2d<T>::operator-=(T num) {
	Matrix2d<T> temp = Matrix2d<T>::operator-(num);
	Matrix2d<T>::operator=(temp);
}
template<typename T>
void Matrix2d<T>::operator*=(T num) {
	Matrix2d<T> temp = Matrix2d<T>::operator*(num);
	Matrix2d<T>::operator=(temp);
}


template<typename T>
Matrix2d<T> Matrix2d<T>::operator^(int num) {
	if (this->h != this->w) {
		throw;
	}
	Matrix2d<T> temp(*this);
	Matrix2d<T> res(this->h, this->w, this->dev);
	res.ones();
	while (num) {
		if (num&1) {
			res.operator*=(temp);
		}
		temp.operator*=(temp);
		num >>= 1;
	}
	return res;
}

template<typename T>
void Matrix2d<T>::operator^=(int num) {
	Matrix2d<T> temp = operator^(num);
	Matrix2d<T>::operator=(temp);
}

template<typename T>
T Matrix2d<T>::sum() {
	if (!this->dev) {
		return Matrix2d_cpu::sum(this->data, this->h * this->w);
	}
	else {
		return Matrix2d_gpu::sum(this->data, this->h * this->w);
	}
}

template<typename T>
Matrix2d<T> Matrix2d<T>::transpose() {
	T* temp = NULL;
	if (!this->dev) {
		Matrix2d_cpu::init(&temp, this->h * this->w);
		Matrix2d_cpu::transpose(temp, this->data, this->w, this->h);
	}
	else {
		Matrix2d_gpu::init(&temp, this->h * this->w);
		Matrix2d_gpu::transpose(temp, this->data, this->w, this->h);
	}
	Matrix2d<T> res(this->w, this->h, temp, this->dev);
	return res;
}
template<typename T>
Matrix2d<T> Matrix2d<T>::t() {
	return transpose();
}


template<typename T>
Matrix2d<T> Matrix2d<T>::cov(Matrix2d<T>& mat) {
	std::pair<int, int> mat_size = mat.size();
	int mat_h = mat_size.first;
	int mat_w = mat_size.second;
	if (this->dev != mat.device() || this->h < mat_h || this->w < mat_w) {
		throw;
	}
	T* temp = NULL;
	if (!this->dev) {
		Matrix2d_cpu::init(&temp, (this->h - mat_h + 1) * (this->w - mat_w + 1));
		Matrix2d_cpu::cov(temp, this->data, mat.get(), this->h, this->w, mat_h, mat_w);
	}
	else {
		Matrix2d_gpu::init(&temp, (this->h - mat_h + 1) * (this->w - mat_w + 1));
		Matrix2d_gpu::cov(temp, this->data, mat.get(), this->h, this->w, mat_h, mat_w);
	}
	Matrix2d<T> res((this->h - mat_h + 1), (this->w - mat_w + 1), temp, this->dev);
	return res;
}


template<typename T>
Matrix2d<T> Matrix2d<T>::LU(Matrix2d<T>* L, Matrix2d<T>* U) {
	if (this->h != this->w) {
		throw;
	}
	T* temp = NULL;
	if (!this->dev) {
		Matrix2d_cpu::init(&temp, this->h * this->w);
		Matrix2d_cpu::assign(temp, this->data, this->h * this->w);
		if (this->L == NULL) {
			Matrix2d_cpu::init(&(this->L), this->h * this->w);
		}
		if (this->U == NULL) {
			Matrix2d_cpu::init(&(this->U), this->h * this->w);
		}
		Matrix2d_cpu::LU(temp, this->h, this->L, this->U);
	}
	else {
		if (this->L == NULL) {
			Matrix2d_gpu::init(&(this->L), this->h * this->w);
		}
		if (this->U == NULL) {
			Matrix2d_gpu::init(&(this->U), this->h * this->w);
		}
		Matrix2d_gpu::init(&temp, this->h * this->w);
		Matrix2d_gpu::assign(temp, this->data, this->h * this->w);
		Matrix2d_gpu::LU(temp, this->h, this->L, this->U);
	}
	Matrix2d<T> res(this->h, this->w, temp, this->dev);
	if (L != NULL) {
		*L = Matrix2d<T>(this->h, this->w, this->L, this->dev);
	}
	if (U != NULL) {
		*U = Matrix2d<T>(this->h, this->w, this->U, this->dev);
	}
	return res;
}


template<typename T>
T Matrix2d<T>::det() {
	if (this->h != this->w) {
		throw;
	}
	T res(0);
	if (this->U == NULL) {
		Matrix2d<T>::LU();
	}
	if (!this->dev) {
		res = Matrix2d_cpu::det(this->U, this->h);
	}
	else {
		res = Matrix2d_gpu::det(this->U, this->h);
	}
	return res;
}


template<typename T>
Matrix2d<T> Matrix2d<T>::inverse() {
	if (this->U == NULL) {
		Matrix2d<T>::LU();
	}
	if (Matrix2d<T>::det() == 0) {
		throw;
	}
	T* temp = NULL;
	if (!this->dev) {
		Matrix2d_cpu::init(&temp, this->h * this->w);
		Matrix2d_cpu::inverse(temp, this->L, this->U, this->h);
	}
	else {
		Matrix2d_gpu::init(&temp, this->h * this->w);
		Matrix2d_gpu::inverse(temp, this->L, this->U, this->h);
	}
	Matrix2d<T> res(this->h, this->w, temp, this->dev);
	return res;
}

template<typename T>
Matrix2d<T> Matrix2d<T>::i() {
	return Matrix2d<T>::inverse();
}

template<typename T>
std::ostream& operator<<(std::ostream& out, Matrix2d<T>& mat) {
	std::pair<int, int> mat_size = mat.size();
	int mat_h = mat_size.first;
	int mat_w = mat_size.second;
	T* temp = mat.get();
	if (mat.device()) {
		Matrix2d_gpu::to_cpu(&temp, mat_h * mat_w);
	}
	for (int i = 0; i < mat_h; ++i) {
		for (int j = 0; j < mat_w; ++j) {
			out << std::setw(5) << temp[i * mat_w + j] << " ";
		}
		out << std::endl;
	}
	return out;
}





