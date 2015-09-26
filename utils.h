#ifndef UTILS_H
#define UTILS_H
#include <mpi.h>

inline size_t block_decompose(const size_t n, const int p, const int rank)
{
    return n / p + (((size_t)rank < n % p) ? 1 : 0);
}

inline size_t block_decompose(const size_t n, MPI_Comm comm)
{
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    return block_decompose(n, p, rank);
}

#endif
