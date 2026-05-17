#include <mpi.h>
#include <omp.h>
#include "Matrix.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <string>

static int LocalRows(int rank, int size, int N)
{
    int base = N / size;
    return base + (rank < N % size ? 1 : 0);
}

static int RowOffset(int rank, int size, int N)
{
    int offset = 0;
    for (int r = 0; r < rank; ++r)
    {
        offset += LocalRows(r, size, N);
    }
    return offset;
}

static double ParallelDeterminantMPI(const Matrix& matrix, int N, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int localRows = LocalRows(rank, size, N);
    int rowOffset = RowOffset(rank, size, N);

    std::vector<double> localData(static_cast<size_t>(localRows) * N);
    std::vector<int> sendCounts(size);
    std::vector<int> displs(size);

    if (rank == 0)
    {
        int offset = 0;
        for (int r = 0; r < size; ++r)
        {
            sendCounts[r] = LocalRows(r, size, N) * N;
            displs[r] = offset;
            offset += sendCounts[r];
        }
    }

    MPI_Scatterv(rank == 0 ? matrix.Data() : nullptr, sendCounts.data(), displs.data(), MPI_DOUBLE, localData.data(), localRows * N, MPI_DOUBLE, 0, comm);

    double detSign = 1.0;
    std::vector<double> pivotRow(N);

    for (int k = 0; k < N; ++k)
    {
        double localMax = 0.0;
        int localPivot = -1;

        for (int i = 0; i < localRows; ++i)
        {
            int globalRow = rowOffset + i;
            if (globalRow < k) continue;

            double value = std::abs(localData[static_cast<size_t>(i) * N + k]);
            if (value > localMax)
            {
                localMax = value;
                localPivot = globalRow;
            }
        }

        struct {
            double value;
            int index;
        } localPair, globalPair;

        localPair.value = localMax;
        localPair.index = localPivot;

        MPI_Allreduce(&localPair, &globalPair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);

        int pivotGlobalRow = globalPair.index;

        if (std::abs(globalPair.value) < 1e-12) return 0.0;

        int ownerPivot = 0;
        int ownerK = 0;

        for (int r = 0; r < size; ++r)
        {
            int start = RowOffset(r, size, N);
            int end = start + LocalRows(r, size, N);

            if (pivotGlobalRow >= start && pivotGlobalRow < end) ownerPivot = r;
            if (k >= start && k < end) ownerK = r;
        }

        if (pivotGlobalRow != k)
        {
            detSign *= -1.0;

            if (ownerPivot == ownerK)
            {
                if (rank == ownerPivot)
                {
                    int localPivotRow = pivotGlobalRow - rowOffset;
                    int localKRow = k - rowOffset;
                    for (int col = 0; col < N; ++col)
                    {
                        std::swap(localData[static_cast<size_t>(localPivotRow) * N + col], localData[static_cast<size_t>(localKRow) * N + col]);
                    }
                }
            }
            else
            {
                if (rank == ownerPivot)
                {
                    int localPivotRow = pivotGlobalRow - rowOffset;
                    MPI_Sendrecv_replace(&localData[static_cast<size_t>(localPivotRow) * N], N, MPI_DOUBLE, ownerK, 0, ownerK, 0, comm, MPI_STATUS_IGNORE);
                }

                if (rank == ownerK)
                {
                    int localKRow = k - rowOffset;
                    MPI_Sendrecv_replace(&localData[static_cast<size_t>(localKRow) * N], N, MPI_DOUBLE, ownerPivot, 0, ownerPivot, 0, comm, MPI_STATUS_IGNORE);
                }
            }
        }

        if (rank == ownerK)
        {
            int localKRow = k - rowOffset;
            std::copy(&localData[static_cast<size_t>(localKRow) * N], &localData[static_cast<size_t>(localKRow) * N + N], pivotRow.begin());
        }

        MPI_Bcast(pivotRow.data(), N, MPI_DOUBLE, ownerK, comm);

        double pivot = pivotRow[k];

        for (int i = 0; i < localRows; ++i)
        {
            int globalRow = rowOffset + i;
            if (globalRow <= k) 
            {
                continue;
            }

            double factor = localData[static_cast<size_t>(i) * N + k] / pivot;
            localData[static_cast<size_t>(i) * N + k] = 0.0;

            for (int col = k + 1; col < N; ++col)
            {
                localData[static_cast<size_t>(i) * N + col] -= factor * pivotRow[col];
            }
        }
    }

    double localDet = 1.0;
    for (int i = 0; i < localRows; ++i)
    {
        int globalRow = rowOffset + i;
        localDet *= localData[static_cast<size_t>(i) * N + globalRow];
    }

    double globalDet = 1.0;
    MPI_Reduce(&localDet, &globalDet, 1, MPI_DOUBLE, MPI_PROD, 0, comm);

    return globalDet * detSign;
}


static double ParallelDeterminantOpenMP(const Matrix& matrix, int& outNumThreads)
{
    int n = matrix.Rows();
    double determinant = 1.0;
    Matrix temp = matrix;

    int numThreads = 1;
    bool isSingular = false;

#pragma omp parallel
    {
#pragma omp single
        numThreads = omp_get_num_threads();

        for (int i = 0; i < n; ++i)
        {
#pragma omp single
            {
                int pivotRow = i;
                double maxVal = std::abs(temp(i, i));

                for (int row = i + 1; row < n; ++row)
                {
                    if (std::abs(temp(row, i)) > maxVal)
                    {
                        maxVal = std::abs(temp(row, i));
                        pivotRow = row;
                    }
                }

                if (maxVal < 1e-12)
                {
                    isSingular = true;
                }
                else
                {
                    if (pivotRow != i)
                    {
                        for (int col = i; col < n; ++col)
                        {
                            std::swap(temp(i, col), temp(pivotRow, col));
                        }
                        determinant *= -1.0;
                    }

                    determinant *= temp(i, i);
                }
            }

            if (isSingular) 
            {
                break;
            }

#pragma omp for schedule(static)
            for (int row = i + 1; row < n; ++row)
            {
                double factor = temp(row, i) / temp(i, i);
                for (int col = i + 1; col < n; ++col)
                {
                    temp(row, col) -= factor * temp(i, col);
                }
            }
        }
    }

    if (isSingular) 
    {
        return 0.0;
    }

    outNumThreads = numThreads;
    return determinant;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1000;
    Matrix matrix;

    if (rank == 0)
    {
        matrix = Matrix(N, N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(-2.0, 2.0);

        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                matrix(i, j) = (i == j) ? dist(gen) * 10.0 : dist(gen);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double startMPI = MPI_Wtime();
    double detMPI = ParallelDeterminantMPI(matrix, N, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double timeMPI = MPI_Wtime() - startMPI;

    if (rank == 0)
    {
        int ompThreads = 0;
        double startOMP = omp_get_wtime();
        double detOMP = ParallelDeterminantOpenMP(matrix, ompThreads);
        double timeOMP = omp_get_wtime() - startOMP;

        double startSeq = MPI_Wtime();
        double detSeq = matrix.Determinant();
        double timeSeq = MPI_Wtime() - startSeq;

        double speedupMPI = timeSeq / timeMPI;
        double efficiencyMPI = (speedupMPI / size) * 100.0;

        double speedupOMP = timeSeq / timeOMP;
        double efficiencyOMP = (speedupOMP / ompThreads) * 100.0;

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "\n=== EXECUTION ENVIRONMENT ===\n";
        std::cout << "Matrix size (N) : " << N << "\n";
        std::cout << "MPI processes   : " << size << "\n";
        std::cout << "OpenMP threads  : " << ompThreads << "\n";

        std::cout << "\n=== COMPUTED DETERMINANTS ===\n";
        std::cout << "Sequential      : " << detSeq << "\n";
        std::cout << "MPI             : " << detMPI << "\n";
        std::cout << "OpenMP          : " << detOMP << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n=== PERFORMANCE METRICS ===\n";
        std::cout << std::left << std::setw(15) << "Method"
            << std::setw(15) << "Time (sec)"
            << std::setw(15) << "Speedup (S)"
            << std::setw(15) << "Efficiency (E)" << "\n";
        std::cout << std::string(60, '-') << "\n";

        std::cout << std::left << std::setw(15) << "Sequential"
            << std::setw(15) << timeSeq
            << std::setw(15) << "1.000000x"
            << std::setw(15) << "100.00%" << "\n";

        std::cout << std::left << std::setw(15) << "MPI"
            << std::setw(15) << timeMPI
            << std::to_string(speedupMPI) + "x\t"
            << std::to_string(efficiencyMPI) + "%" << "\n";

        std::cout << std::left << std::setw(15) << "OpenMP"
            << std::setw(15) << timeOMP
            << std::to_string(speedupOMP) + "x\t"
            << std::to_string(efficiencyOMP) + "%" << "\n";

        std::cout << "\n";
    }

    MPI_Finalize();
    return 0;
}