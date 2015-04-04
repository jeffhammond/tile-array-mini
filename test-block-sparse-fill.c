#include <math.h>

#include "tile-array.h"

int main(int argc, char * argv[])
{
    int requested=MPI_THREAD_SERIALIZED, provided;
    MPI_Init_thread(&argc, &argv, requested, &provided);

    if (provided<requested) MPI_Abort(MPI_COMM_WORLD, provided);

    int np, me;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    MPI_Barrier(MPI_COMM_WORLD);

    size_t count = (argc>1) ? atol(argv[1]) : 4097;

    ta_t g_x;

    /* A block-sparse matrix with the following fill:
     *
     *    +----+
     *    |XX00|
     *    |XX00|
     *    |00X0|
     *    |000X|
     *    +----+
     *              */
    ssize_t block_offset[4][4] = {{ 0, 1,-1,-1},
                                  { 2, 3,-1,-1},
                                  {-1,-1, 4,-1},
                                  {-1,-1,-1, 5}};

    int ntiles = 6;
    ta_create(MPI_COMM_WORLD, ntiles, count, &g_x);

    cntr_t nxtval;
    cntr_create(MPI_COMM_WORLD, &nxtval);
    if (me==0) cntr_zero(nxtval);

    MPI_Barrier(MPI_COMM_WORLD);

    long counter = 0;
    cntr_fadd(nxtval, 1, &counter);
    for (int i=0; i<4; i++) {
      for (int j=0; j<4; j++) {
        if (block_offset[i][j]==counter) {
            printf("rank %d got task (%d,%d)\n", me, i, j); fflush(stdout);
            double * tmp = malloc(count * sizeof(double));
            ta_get_tile(g_x, block_offset[i][j], tmp);
            for (size_t k=0; k<count; k++) tmp[k] = 1.+block_offset[i][j];
            ta_put_tile(g_x, block_offset[i][j], tmp);
            free(tmp);
            cntr_fadd(nxtval, 1, &counter);
        }
      }
    }
    ta_sync_array(g_x);

    if (count<100) ta_print_array(g_x);

    cntr_destroy(&nxtval);

    ta_destroy(&g_x);

    MPI_Finalize();
    return 0;
}
