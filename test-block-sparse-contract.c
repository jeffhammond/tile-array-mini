#include <unistd.h> /* getpagesize() */
#include <math.h>

#include "tile-blas.h"

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

    int tilesize = (argc>1) ? atoi(argv[1]) : 200; 
    size_t count = tilesize*tilesize;

    ta_t g_a, g_b, g_c;

#if PROBLEM_SIZE==4
    /* A block-sparse matrix with the following fill:
     *
     *    +----+
     *    |XX00|
     *    |XX00|
     *    |00X0|
     *    |000X|
     *    +----+
     *              */
    int ntiles = 6;
    int tilesdim = 4;
    ssize_t block_offset[4][4] = {{ 0, 1,-1,-1},
                                  { 2, 3,-1,-1},
                                  {-1,-1, 4,-1},
                                  {-1,-1,-1, 5}};
#else
    int ntiles = 54;
    int tilesdim = 12;
    ssize_t block_offset[12][12] = {{ 0, 1, 2, 3, 4, 5,-1,-1,-1,-1,-1,-1},
                                    { 6, 7, 8, 9,10,11,-1,-1,-1,-1,-1,-1},
                                    {12,13,14,15,16,17,-1,-1,-1,-1,-1,-1},
                                    {18,19,20,21,22,23,-1,-1,-1,-1,-1,-1},
                                    {24,25,26,27,28,29,-1,-1,-1,-1,-1,-1},
                                    {30,31,32,33,34,35,-1,-1,-1,-1,-1,-1},
                                    {-1,-1,-1,-1,-1,-1,36,37,38,-1,-1,-1},
                                    {-1,-1,-1,-1,-1,-1,39,40,41,-1,-1,-1},
                                    {-1,-1,-1,-1,-1,-1,42,43,44,-1,-1,-1},
                                    {-1,-1,-1,-1,-1,-1,-1,-1,-1,45,46,47},
                                    {-1,-1,-1,-1,-1,-1,-1,-1,-1,48,49,50},
                                    {-1,-1,-1,-1,-1,-1,-1,-1,-1,51,52,53}};
#endif

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

    double * t_a = malloc(count * sizeof(double));
    double * t_b = malloc(count * sizeof(double));
    double * t_c = malloc(count * sizeof(double));

    long counter = 0;
    long taskid  = 0;
    cntr_fadd(nxtval, 1, &counter);
    for (int i=0; i<tilesdim; i++) {
      for (int j=0; j<tilesdim; j++) {
        if (block_offset[i][j] >= 0) {
          for (int k=0; k<tilesdim; k++) {
            if ((block_offset[i][k] >= 0) && (block_offset[k][j] >= 0)) {
              if (counter==taskid) {
                printf("rank %d counter %ld taskid %ld (%d,%d,%d)\n", me, counter, taskid, i, j, k); fflush(stdout);
                memset(t_a, 0, count*sizeof(double));
                memset(t_b, 0, count*sizeof(double));
                memset(t_c, 0, count*sizeof(double));
                ta_get_tile(g_a, block_offset[i][k], t_a);
                ta_get_tile(g_b, block_offset[k][j], t_b);

                matmul(tilesize, tilesize, tilesize, t_a, t_b, t_c, false);

                char label[8];
                sprintf(label, "%d", me);
                ta_print_tile(label, count, t_a);
                ta_print_tile(label, count, t_b);
                ta_print_tile(label, count, t_c);

                ta_sum_tile(g_c, block_offset[i][j], t_c);
                cntr_fadd(nxtval, 1, &counter);
              }
              taskid++;
            }
          }
        }
      }
    }
    ta_sync_array(g_c);

    if (tilesize<50) ta_print_array(g_c);

    free(t_a);
    free(t_b);
    free(t_c);

    cntr_destroy(&nxtval);

    ta_destroy(&g_a);
    ta_destroy(&g_b);
    ta_destroy(&g_c);

    MPI_Finalize();
    return 0;
}
