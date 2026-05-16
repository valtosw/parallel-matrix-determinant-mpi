#include <mpi.h>
#include "Matrix.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

static int get_owner(int global_row, int P) {
    return global_row % P;
}

static int get_local_row(int global_row, int P) {
    return global_row / P;
}

static int get_global_row(int local_row, int rank, int P) {
    return local_row * P + rank;
}

struct DoubleInt {
    double val;
    int rank_or_row;
};

static double ParallelDeterminantMPI(const Matrix& global_matrix, int N, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int local_rows = (N / size) + (rank < (N % size) ? 1 : 0);
    std::vector<double> local_data(local_rows * N, 0.0);

    std::vector<int> sendcounts(size, 0);
    std::vector<int> displs(size, 0);
    std::vector<double> send_buffer;

    if (rank == 0) {
        send_buffer.resize(static_cast<std::vector<double, std::allocator<double>>::size_type>(N) * N);
        int offset = 0;
        for (int p = 0; p < size; ++p) {
            displs[p] = offset;
            int rows_for_p = (N / size) + (p < (N % size) ? 1 : 0);
            sendcounts[p] = rows_for_p * N;

            for (int lr = 0; lr < rows_for_p; ++lr) {
                int global_row = get_global_row(lr, p, size);

                for (int col = 0; col < N; ++col) {
                    send_buffer[static_cast<std::vector<double, std::allocator<double>>::size_type>(offset) + col] = global_matrix(global_row, col);
                }

                offset += N;
            }
        }
    }

    const double* sendbuf = rank == 0 ? send_buffer.data() : nullptr;
    MPI_Scatterv(sendbuf, sendcounts.data(), displs.data(), MPI_DOUBLE, local_data.data(), local_rows * N, MPI_DOUBLE, 0, comm);

    double determinant_sign = 1.0;
    std::vector<double> pivot_buffer(N, 0.0);

    for (int i = 0; i < N; ++i) {
        double local_max_val = -1.0;
        int local_max_row = -1;

        for (int lr = 0; lr < local_rows; ++lr) {
            int global_row = get_global_row(lr, rank, size);
            if (global_row >= i) {
                double val = std::abs(local_data[static_cast<std::vector<double, std::allocator<double>>::size_type>(lr) * N + i]);
                if (val > local_max_val) {
                    local_max_val = val;
                    local_max_row = global_row;
                }
            }
        }

        DoubleInt local_max{ local_max_val, local_max_row };
        DoubleInt global_max;

        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);

        if (global_max.val < 1e-9) {
            return 0.0;
        }

        int pivot_row = global_max.rank_or_row;
        int owner_i = get_owner(i, size);
        int owner_p = get_owner(pivot_row, size);

        if (i != pivot_row) {
            if (owner_i == owner_p) {
                if (rank == owner_i) {
                    int l_i = get_local_row(i, size);
                    int l_p = get_local_row(pivot_row, size);
                    for (int col = 0; col < N; ++col) {
                        std::swap(local_data[static_cast<std::vector<double, std::allocator<double>>::size_type>(l_i) * N + col], 
                            local_data[static_cast<std::vector<double, std::allocator<double>>::size_type>(l_p) * N + col]);
                    }
                }
            }
            else { 
                if (rank == owner_i) {
                    int l_i = get_local_row(i, size);
                    MPI_Sendrecv_replace(&local_data[static_cast<std::vector<double, std::allocator<double>>::size_type>(l_i) * N], 
                        N, MPI_DOUBLE, owner_p, 0, owner_p, 0, comm, MPI_STATUS_IGNORE);
                }
                else if (rank == owner_p) {
                    int l_p = get_local_row(pivot_row, size);
                    MPI_Sendrecv_replace(&local_data[static_cast<std::vector<double, std::allocator<double>>::size_type>(l_p) * N], 
                        N, MPI_DOUBLE, owner_i, 0, owner_i, 0, comm, MPI_STATUS_IGNORE);
                }
            }
            determinant_sign *= -1.0;
        }

        if (rank == owner_i) {
            int l_i = get_local_row(i, size);
            std::copy(&local_data[static_cast<std::vector<double, std::allocator<double>>::size_type>(l_i) * N], 
                &local_data[static_cast<std::vector<double, std::allocator<double>>::size_type>(l_i) * N] + N, pivot_buffer.begin());
        }
        MPI_Bcast(pivot_buffer.data(), N, MPI_DOUBLE, owner_i, comm);

        for (int lr = 0; lr < local_rows; ++lr) {
            int global_row = get_global_row(lr, rank, size);
            if (global_row > i) {
                double factor = local_data[static_cast<std::vector<double, std::allocator<double>>::size_type>(lr) * N + i] / pivot_buffer[i];
                for (int col = i + 1; col < N; ++col) {
                    local_data[static_cast<std::vector<double, std::allocator<double>>::size_type>(lr) * N + col] -= factor * pivot_buffer[col];
                }
            }
        }
    }

    double local_det = 1.0;
    for (int lr = 0; lr < local_rows; ++lr) {
        int global_row = get_global_row(lr, rank, size);
        local_det *= local_data[static_cast<std::vector<double, std::allocator<double>>::size_type>(lr) * N + global_row];
    }

    double global_det = 1.0;
    MPI_Reduce(&local_det, &global_det, 1, MPI_DOUBLE, MPI_PROD, 0, comm);

    return global_det * determinant_sign;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 2000;

    bool run_sequential = true;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--skip-seq") {
            run_sequential = false;
        }
    }

    Matrix matrix;
    if (rank == 0) {
        matrix = Matrix(N, N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(-2.0, 2.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                matrix(i, j) = (i == j) ? dist(gen) * 5.0 : dist(gen);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_par = MPI_Wtime();
    double det_par = ParallelDeterminantMPI(matrix, N, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double time_par = MPI_Wtime() - start_par;

    if (rank == 0) {
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "\n=== Calculation Results (N = " << N << ", p = " << size << ") ===\n";

        if (run_sequential) {
            double start_seq = MPI_Wtime();
            double det_seq = matrix.Determinant();
            double time_seq = MPI_Wtime() - start_seq;

            std::cout << std::scientific;
            std::cout << "Sequential Determinant : " << det_seq << "\n";
            std::cout << std::fixed;
            std::cout << "Sequential Time (T1)   : " << time_seq << " seconds\n";
            std::cout << "Parallel Time (Tp)     : " << std::fixed << time_par << " seconds\n";
            std::cout << "Parallel Determinant   : " << det_par << "\n";

            double acceleration = time_seq / time_par;
            double efficiency = acceleration / size;
            std::cout << "Acceleration (S)       : " << acceleration << "x\n";
            std::cout << "Efficiency (E)         : " << (efficiency * 100.0) << "%\n";
        }
        else {
            std::cout << "Parallel Time (Tp)     : " << std::fixed << time_par << " seconds\n";
            std::cout << "Parallel Determinant   : " << det_par << "\n";
            std::cout << "S = T1 / " << time_par << "\n";
            std::cout << "E = S / " << size << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}