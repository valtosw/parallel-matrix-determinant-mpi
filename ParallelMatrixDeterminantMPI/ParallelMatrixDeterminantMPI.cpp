#include <mpi.h>
#include "Matrix.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>

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

    MPI_Scatterv(rank == 0 ? matrix.Data() : nullptr, sendCounts.data(), displs.data(), MPI_DOUBLE, localData.data(), localRows * N, MPI_DOUBLE, 0,comm);

    double detSign = 1.0;
    std::vector<double> pivotRow(N);

    for (int k = 0; k < N; ++k)
    {
        double localMax = 0.0;
        int localPivot = -1;

        for (int i = 0; i < localRows; ++i)
        {
            int globalRow = rowOffset + i;

            if (globalRow < k)
            {
                continue;
            }

            double value = std::abs(localData[static_cast<size_t>(i) * N + k]);

            if (value > localMax)
            {
                localMax = value;
                localPivot = globalRow;
            }
        }

        struct
        {
            double value;
            int index;
        } localPair, globalPair;

        localPair.value = localMax;
        localPair.index = localPivot;

        MPI_Allreduce(&localPair, &globalPair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);

        int pivotGlobalRow = globalPair.index;

        if (std::abs(globalPair.value) < 1e-12)
        {
            return 0.0;
        }

        int ownerPivot = 0;
        int ownerK = 0;

        for (int r = 0; r < size; ++r)
        {
            int start = RowOffset(r, size, N);
            int end = start + LocalRows(r, size, N);

            if (pivotGlobalRow >= start && pivotGlobalRow < end)
            {
                ownerPivot = r;
            }

            if (k >= start && k < end)
            {
                ownerK = r;
            }
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

            double factor = localData[ static_cast<size_t>(i) * N + k] / pivot;

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

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 500;

    Matrix matrix;

    if (rank == 0)
    {
        matrix = Matrix(N, N);

        std::mt19937 gen(42);

        std::uniform_real_distribution<double>dist(-2.0, 2.0);

        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                matrix(i, j) = (i == j) ? dist(gen) * 10.0 : dist(gen);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double startParallel = MPI_Wtime();

    double detParallel = ParallelDeterminantMPI(matrix, N, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    double parallelTime = MPI_Wtime() - startParallel;

    if (rank == 0)
    {
        double startSequential = MPI_Wtime();
        double detSequential = matrix.Determinant();
        double sequentialTime = MPI_Wtime() - startSequential;
        double speedup = sequentialTime / parallelTime;
        double efficiency = speedup / size;

        std::cout << std::scientific << std::setprecision(6);

        std::cout << "\n=== RESULTS ===\n";

        std::cout
            << "Matrix size (N): "
            << N << "\n";

        std::cout
            << "MPI processes  : "
            << size << "\n\n";

        std::cout
            << "Sequential determinant : "
            << detSequential << "\n";

        std::cout
            << "Parallel determinant   : "
            << detParallel << "\n\n";

        std::cout
            << std::fixed;

        std::cout
            << "Sequential time : "
            << sequentialTime
            << " sec\n";

        std::cout
            << "Parallel time   : "
            << parallelTime
            << " sec\n";

        std::cout
            << "Speedup (S)     : "
            << speedup
            << "x\n";

        std::cout
            << "Efficiency (E)  : "
            << efficiency * 100.0
            << "%\n";
    }

    MPI_Finalize();

    return 0;
}
