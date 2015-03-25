#include <unistd.h> /* getpagesize() */
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

    size_t count = (argc>1) ? atol(argv[1]) : 1+2*getpagesize();

    ta_t g_a, g_b, g_c;

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
    ta_create(MPI_COMM_WORLD, ntiles, count, &g_a);
    ta_create(MPI_COMM_WORLD, ntiles, count, &g_b);
    ta_create(MPI_COMM_WORLD, ntiles, count, &g_c);

    ta_memset_array(g_a, 1.0);
    ta_memset_array(g_b, 1.0);
    ta_memset_array(g_c, 0.0);

    cntr_t nxtval;
    cntr_create(MPI_COMM_WORLD, &nxtval);
    if (me==0) cntr_zero(nxtval);

    MPI_Barrier(MPI_COMM_WORLD);

    long counter = 0;
    long taskid  = 0;
    cntr_fadd(nxtval, 1, &counter);
    for (int i=0; i<4; i++) {
      for (int j=0; j<4; j++) {
        for (int k=0; k<4; k++) {
          if (block_offset[i][j] >= 0) {
          if (block_offset[i][k] >= 0) {
          if (block_offset[k][j] >= 0) {
          if (counter==taskid) {
              printf("rank %d got task (%d,%d,%d)\n", me, i, j, k); fflush(stdout);
              printf("block_offset[i][k] = %ld\n", block_offset[i][k]);
              double * t_a = malloc(count * sizeof(double));
              double * t_b = malloc(count * sizeof(double));
              double * t_c = malloc(count * sizeof(double));
              ta_get_tile(g_a, block_offset[i][k], t_a);
              ta_get_tile(g_b, block_offset[k][j], t_b);
              for (size_t p=0; p<count; p++) t_c[p] = t_a[p] * t_b[p];
              ta_sum_tile(g_c, block_offset[i][j], t_c);
              free(t_a);
              free(t_b);
              free(t_c);
              cntr_fadd(nxtval, 1, &counter);
          }
          }
          }
          }
          taskid++;
        }
      }
    }
    ta_sync_array(g_c);

    if (count<100) ta_print_array(g_c);

    cntr_destroy(&nxtval);

    ta_destroy(&g_a);
    ta_destroy(&g_b);
    ta_destroy(&g_c);

    MPI_Finalize();
    return 0;
}
