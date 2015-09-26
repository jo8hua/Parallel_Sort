#include "parallel_sort.h"
#include "utils.h"
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <numeric>

void parallel_sort(int * begin, int* end, MPI_Comm comm) {
    static bool initrand = initialize_random( comm );
    int size, rank, pivot;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    
    if (size == 1) {
        
        std::sort( begin, end );
        return;
    } 

    // Eliminate processes with zero elements
    bool zero = end - begin == 0;
    int nzcolor = static_cast<int>( zero );
    MPI_Comm nzcomm;
    MPI_Comm_split( comm, nzcolor, rank, &nzcomm );

    if ( zero ) 
        return;
        
    // Generate pivot index and broadcast
    select_pivot(&pivot, begin, end, nzcomm);
    if ( pivot == -1 )
        return;
    
    //Partition locally
    std::ptrdiff_t n = end - begin;
    std::vector<int> left;
    std::vector<int> right;
    for(int i = 0 ; i < n ; i++) {
        if (begin[i] <= pivot) {
            left.push_back(begin[i]);
        } else {
            right.push_back(begin[i]);
        }
    }
    
    // Determine overall left and right partition sizes
    int left_size = left.size();
    int right_size = right.size();
    int left_sum, right_sum;
    MPI_Allreduce( &left_size, &left_sum, 1, MPI_INT, MPI_SUM, nzcomm);
    MPI_Allreduce( &right_size, &right_sum, 1, MPI_INT, MPI_SUM, nzcomm);

    // Get individual partition sizes
    std::vector<int> allleft( size );
    std::vector<int> allright( size );
    MPI_Allgather( &left_size, 1, MPI_INT,
                   &allleft[0], 1, MPI_INT, nzcomm );
    MPI_Allgather( &right_size, 1, MPI_INT,
                   &allright[0], 1, MPI_INT, nzcomm );

    // Create new communicators
    MPI_Comm newcomm;
    int rankpivot = size * left_sum / ( left_sum + right_sum );

    // Make sure there is a split
    if ( rankpivot == 0 )
        rankpivot++;
    else if ( rankpivot == size )
        rankpivot--;
    
    bool onright = rank >= rankpivot;
    int color = static_cast<int>( onright );
    MPI_Comm_split( nzcomm, color, rank, &newcomm );

    // New stats
    int newsize, newrank;
    MPI_Comm_size( newcomm, &newsize );
    MPI_Comm_rank( newcomm, &newrank );

    // Global send ranges
    int lsstart = std::accumulate( allleft.begin(), allleft.begin()+rank, 0 );
    int lsend = lsstart + allleft[rank];
    int rsstart = std::accumulate( allright.begin(), allright.begin()+rank, 0 );
    int rsend = rsstart + allright[rank];

    // Global recv ranges
    int lrstart, lrend, rrstart, rrend;
    if ( onright ){
        lrstart = 0;
        lrend = 0;
        rrstart = 0;
        rrend = block_decompose( right_sum, newsize, 0 );
        for ( int i = 1; i <= newrank; i++ ){
            rrstart = rrend;
            rrend += block_decompose( right_sum, newsize, i );
        }
    }
    else{
        rrstart = 0;
        rrend = 0;
        lrstart = 0;
        lrend = block_decompose( left_sum, newsize, 0 );
        for ( int i = 1; i <= newrank; i++ ){
            lrstart = lrend;
            lrend += block_decompose( left_sum, newsize, i );
        }
    }
    
    // Setup AlltoAll
    std::vector<int> lsendcnts( size, 0 ),
                     lsdispls( size, 0 ),
                     lrecvcnts( size, 0 ),
                     lrdispls( size, 0 ),
                     rsendcnts( size, 0 ),
                     rsdispls( size, 0 ),
                     rrecvcnts( size, 0 ),
                     rrdispls( size, 0 );

    if ( rankpivot )
        setup_alltoall_send( left_sum, 0, lsstart, lsend, lsendcnts, lsdispls, rankpivot );
    setup_alltoall_recv( allleft, lrstart, lrend, lrecvcnts, lrdispls );
    if ( size - rankpivot )
        setup_alltoall_send( right_sum, rankpivot, rsstart, rsend, rsendcnts, rsdispls, size - rankpivot );
    setup_alltoall_recv( allright, rrstart, rrend, rrecvcnts, rrdispls );

    // AlltoAll data
    std::vector<int> data( std::max( rrend - rrstart, lrend - lrstart ) );
    MPI_Alltoallv( &left[0], &lsendcnts[0], &lsdispls[0], MPI_INT,
                    &data[0], &lrecvcnts[0], &lrdispls[0], MPI_INT, comm );
    MPI_Alltoallv( &right[0], &rsendcnts[0], &rsdispls[0], MPI_INT,
                    &data[0], &rrecvcnts[0], &rrdispls[0], MPI_INT, comm );

    // Recurse
    parallel_sort( &data[0], &data[0]+data.size(), newcomm );

    // Setup AlltoAll
    std::vector<int> sendcnts( size, 0 ),
                     sdispls( size, 0 ),
                     recvcnts( size, 0 ),
                     rdispls( size, 0 );

    int sum = left_sum+right_sum;

    // Global send range
    int sstart, send;
    if ( onright ){
        sstart = left_sum + rrstart;
        send = left_sum + rrend;
    }
    else{
        sstart = lrstart;
        send = lrend;
    }

    // Global recv range
    int rstart, rend;
            
    rstart = 0;
    rend = block_decompose( sum, size, 0 );
    for ( int i = 1; i <= rank; i++ ){
        rstart = rend;
        rend += block_decompose( sum, size, i );
    }
    
    std::vector<int> olddecompose( size );
    for ( int i = 0; i < size; i++ ){
        if ( i >= rankpivot )
            olddecompose[i] = block_decompose( right_sum, size - rankpivot, i-rankpivot );
        else
            olddecompose[i] = block_decompose( left_sum, rankpivot, i );                
    }

    setup_alltoall_send( sum, 0, sstart, send, sendcnts, sdispls, size );
    setup_alltoall_recv( olddecompose, rstart, rend, recvcnts, rdispls );

    // Write back to input              
    MPI_Alltoallv( &data[0], &sendcnts[0], &sdispls[0], MPI_INT,
                    begin, &recvcnts[0], &rdispls[0], MPI_INT, nzcomm );    
}

// Synchronize seed
bool initialize_random( MPI_Comm comm ){    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    double now;
    if ( rank == 0 ){
        struct timespec t_now;
        clock_gettime(CLOCK_MONOTONIC,  &t_now);
        now = t_now.tv_sec + (double) t_now.tv_nsec * 1e-9;
    }
    MPI_Bcast( &now, 1, MPI_DOUBLE, 0, comm );
    srand( now );       
    return true;
}

// Calculates ovelap amount of two intervals
int calc_interval_overlap( int first_start, int first_end, int second_start, int second_end ){
    return ( std::abs( first_start - second_end ) + std::abs( first_end - second_start ) 
                                - std::abs( first_start - second_start ) - std::abs( first_end - second_end ) ) / 2;
}


void setup_alltoall_send( int datanum, int rankstart, int sendstartindex, int sendendindex,
                            std::vector<int> &sendcnts, std::vector<int> &sdispls, int newsize ){
    int minindex = 0;
    int maxindex = block_decompose( datanum, newsize, 0 );
    if ( sendendindex > minindex && sendstartindex < maxindex ){
        sendcnts[rankstart] = calc_interval_overlap( sendstartindex, sendendindex, minindex, maxindex );
        sdispls[rankstart] = 0;
    }
    for ( int i = 1; i < newsize; i++ ){
        int r = rankstart + i;
        minindex = maxindex;
        maxindex += block_decompose( datanum, newsize, i );
        if ( sendendindex > minindex && sendstartindex < maxindex ){
            sendcnts[r] = calc_interval_overlap( sendstartindex, sendendindex, minindex, maxindex );
            sdispls[r] = sdispls[r-1] + sendcnts[r-1];
        }
    }
}

void setup_alltoall_recv( std::vector<int> &datanum, int needstartindex, int needendindex,
                            std::vector<int> &recvcnts, std::vector<int> &rdispls ){
    int minindex = 0;
    int maxindex = datanum[0];
    if ( needendindex > minindex && needstartindex < maxindex ){
        recvcnts[0] = calc_interval_overlap( needstartindex, needendindex, minindex, maxindex );
        rdispls[0] = 0;
    }
    for ( size_t i = 1; i < datanum.size(); i++ ){
        minindex = maxindex;
        maxindex += datanum[i];
        if ( needendindex > minindex && needstartindex < maxindex ){
            recvcnts[i] = calc_interval_overlap( needstartindex, needendindex, minindex, maxindex );
            rdispls[i] = rdispls[i-1] + recvcnts[i-1];
        }
    }
}

void select_pivot (int * pivot, int * begin, int * end, MPI_Comm comm) {
    
    int rank, size, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    p = rand() % size;
    
    std::ptrdiff_t n = end - begin;
    *pivot = begin[rand() % n];

    MPI_Bcast(pivot, 1, MPI_INT, p, comm);
}

void print( int *begin, int * end ){
    for ( int *it = begin; it != end; it++ ){
        std::cout << *it << ' ';
    }
    std::cout << std::endl;
}
