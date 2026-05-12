#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

class Matrix
{
private:
    int _rows;
    int _cols;
    std::vector<double> _data;

public:
    Matrix(int rows, int cols) : _rows(rows), _cols(cols), _data(rows* cols, 0.0) {}

    int Rows() const
    {
        return _rows;
    }

    int Cols() const
    {
        return _cols;
    }

    double& operator()(int row, int col)
    {
        return _data[static_cast<std::vector<double, std::allocator<double>>::size_type>(row) * _cols + col];
    }

    double operator()(int row, int col) const
    {
        return _data[static_cast<std::vector<double, std::allocator<double>>::size_type>(row) * _cols + col];
    }

    void Print() const
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

    double Determinant() const
    {
        if (_rows != _cols)
        {
            throw std::runtime_error("Matrix must be square to compute a determinant.");
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
};