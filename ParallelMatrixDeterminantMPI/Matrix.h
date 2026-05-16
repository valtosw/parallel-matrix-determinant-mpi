#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>

class Matrix
{
private:
    int _rows;
    int _cols;
    std::vector<double> _data;

public:
    Matrix();
    Matrix(int rows, int cols);

    int Rows() const;
    int Cols() const;

    double& operator()(int row, int col);
    double operator()(int row, int col) const;

    const double* Data() const;
    double* Data();

    void Print() const;
    double Determinant() const;
};