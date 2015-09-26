#ifndef PARALLEL_SORT_H
#define PARALLEL_SORT_H
#include <mpi.h>
#include <stdlib.h>
#include <vector>

void parallel_sort(int * begin, int* end, MPI_Comm comm);

bool initialize_random( MPI_Comm comm );

int calc_interval_overlap( int first_start, int first_end, int second_start, int second_end );

void setup_alltoall_send( int datanum, int rankstart, int sendstartindex, int sendendindex, std::vector<int> &sendcnts, std::vector<int> &sdispls, int newsize );

void setup_alltoall_recv( std::vector<int> &datanum, int needstartindex, int needendindex, std::vector<int> &recvcnts, std::vector<int> &rdispls );

void select_pivot (int* pivot, int* begin, int* end, MPI_Comm comm);

void print( int *begin, int * end );

#endif
