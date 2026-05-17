#include "Matrix.h"

Matrix::Matrix() : _rows(0), _cols(0) {}

Matrix::Matrix(int rows, int cols) : _rows(rows), _cols(cols), _data(rows* cols, 0.0) {}

int Matrix::Rows() const { return _rows; }

int Matrix::Cols() const { return _cols; }

double& Matrix::operator()(int row, int col)
{
    return _data[static_cast<size_t>(row) * _cols + col];
}

double Matrix::operator()(int row, int col) const
{
    return _data[static_cast<size_t>(row) * _cols + col];
}

const double* Matrix::Data() const
{
    return _data.empty() ? nullptr : _data.data();
}

double* Matrix::Data()
{
    return _data.empty() ? nullptr : _data.data();
}

void Matrix::Print() const
{
    for (int i = 0; i < _rows; ++i)
    {
        for (int j = 0; j < _cols; ++j)
        {
            std::cout << (*this)(i, j) << " ";
        }

        std::cout << '\n';
    }
}

double Matrix::Determinant() const
{
    if (_rows != _cols || _rows == 0)
    {
        throw std::runtime_error("Matrix must be square and non-empty to compute a determinant.");
    }

    int n = _rows;
    double determinant = 1.0;
    Matrix temp = *this;

    for (int i = 0; i < n; ++i)
    {
        int pivotRow = i;

        for (int row = i + 1; row < n; ++row)
        {
            if (std::abs(temp(row, i)) > std::abs(temp(pivotRow, i)))
            {
                pivotRow = row;
            }
        }

        if (std::abs(temp(pivotRow, i)) < 1e-9)
        {
            return 0.0;
        }

        if (pivotRow != i)
        {
            for (int col = i; col < n; ++col)
            {
                std::swap(temp(i, col), temp(pivotRow, col));
            }
            determinant *= -1.0;
        }

        determinant *= temp(i, i);

        for (int row = i + 1; row < n; ++row)
        {
            double factor = temp(row, i) / temp(i, i);

            for (int col = i + 1; col < n; ++col)
            {
                temp(row, col) -= factor * temp(i, col);
            }
        }
    }

    return determinant;
}