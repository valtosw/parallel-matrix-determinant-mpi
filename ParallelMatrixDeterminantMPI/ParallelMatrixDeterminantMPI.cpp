//#include <mpi.h>
#include "Matrix.cpp"

#include <iostream>


int main()
{
    Matrix matrix(3, 3);

    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(0, 2) = 3;

    matrix(1, 0) = 0;
    matrix(1, 1) = 4;
    matrix(1, 2) = 5;

    matrix(2, 0) = 1;
    matrix(2, 1) = 0;
    matrix(2, 2) = 6;

    std::cout << "Matrix:" << std::endl;
    matrix.Print();

    double determinant = matrix.Determinant();

    std::cout << "\nDeterminant: "
        << determinant
        << std::endl;

    return 0;
}