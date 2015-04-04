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
    const int ntiles = 6;
    const int tilesdim = 4;
    ssize_t block_offset[4][4] = {{ 0, 1,-1,-1},
                                  { 2, 3,-1,-1},
                                  {-1,-1, 4,-1},
                                  {-1,-1,-1, 5}};
#else
    const int ntiles = 54;
    const int tilesdim = 12;
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

    double * t_a = malloc(count * sizeof(double)); assert(t_a!=NULL);
    double * t_b = malloc(count * sizeof(double)); assert(t_b!=NULL);
    double * t_c = malloc(count * sizeof(double)); assert(t_c!=NULL);

    long counter = 0;
    long taskid  = 0;
    cntr_fadd(nxtval, 1, &counter);
    for (int i=0; i<tilesdim; i++) {
      for (int j=0; j<tilesdim; j++) {
        if (block_offset[i][j] >= 0) {
          for (int k=0; k<tilesdim; k++) {
            if ((block_offset[i][k] >= 0) && (block_offset[k][j] >= 0)) {
              if (counter==taskid) {
#if DEBUG_LEVEL>1
                printf("rank %d counter %ld taskid %ld (%d,%d,%d)\n", me, counter, taskid, i, j, k); fflush(stdout);
#endif
                memset(t_a, 0, count*sizeof(double));
                memset(t_b, 0, count*sizeof(double));
                memset(t_c, 0, count*sizeof(double));
                ta_get_tile(g_a, block_offset[i][k], t_a);
                ta_get_tile(g_b, block_offset[k][j], t_b);

                matmul(tilesize, tilesize, tilesize, t_a, t_b, t_c, false);

#if DEBUG_LEVEL>2
                char label[8];
                sprintf(label, "%d", me);
                ta_print_tile(label, count, t_a);
                ta_print_tile(label, count, t_b);
                ta_print_tile(label, count, t_c);
#endif

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

#if DEBUG_LEVEL>2
    if (tilesize<50) ta_print_array(g_c);
#endif

    free(t_a);
    free(t_b);
    free(t_c);

    cntr_destroy(&nxtval);

    /* begin correctness checking */

    MPI_Barrier(MPI_COMM_WORLD);

    if (me==0) {
      const int matrixdim = tilesdim * tilesize;
      double * restrict m_a = calloc(matrixdim*matrixdim*sizeof(double),sizeof(double));
      double * restrict m_b = calloc(matrixdim*matrixdim*sizeof(double),sizeof(double));
      double * restrict m_c = calloc(matrixdim*matrixdim*sizeof(double),sizeof(double));
      double * restrict m_d = calloc(matrixdim*matrixdim*sizeof(double),sizeof(double));
      if (m_a==NULL || m_b==NULL || m_c==NULL || m_d==NULL) {
        printf("cannot allocate dense matrices on 1 process - correctness checking skipped\n");
        if (m_a!=NULL) free(m_a);
        if (m_b!=NULL) free(m_b);
        if (m_c!=NULL) free(m_c);
        if (m_d!=NULL) free(m_d);
      } else {
        double * restrict t_a = malloc(count * sizeof(double));
        double * restrict t_b = malloc(count * sizeof(double));
        double * restrict t_c = malloc(count * sizeof(double));
        /* gather tiles into buffer and copy into matrix manually 
         * because we do not have GA-style API for local strided */
        for (int i=0; i<tilesdim; i++) {
          for (int j=0; j<tilesdim; j++) {
            if (block_offset[i][j] >= 0) {
              /* fetch tile */
              ta_get_tile(g_a, block_offset[i][j], t_a);
              ta_get_tile(g_b, block_offset[i][j], t_b);
              ta_get_tile(g_c, block_offset[i][j], t_c);
              /* copy into matrix */
              for (int ii=0; ii<tilesize; ii++) {
                for (int jj=0; jj<tilesize; jj++) {
                  const int row = i*tilesize + ii;
                  const int col = j*tilesize + jj;
                  m_a[row*matrixdim+col] = t_a[ii*tilesize+jj]; 
                  m_b[row*matrixdim+col] = t_b[ii*tilesize+jj]; 
                  m_c[row*matrixdim+col] = t_c[ii*tilesize+jj]; 
                }
              }
            }
          }
        }
        free(t_a);
        free(t_b);
        free(t_c);
        matmul(matrixdim, matrixdim, matrixdim, m_a, m_b, m_d, false);
        size_t errors = 0;
        for (int i=0; i<matrixdim; i++) {
          for (int j=0; j<matrixdim; j++) {
            errors += (m_c[i*matrixdim+j] != m_d[i*matrixdim+j]);
          }
        }
        printf("%zu errors\n", errors);
        free(m_a);
        free(m_b);
        free(m_c);
        free(m_d);
        assert(errors==0);
      }
    }
#if 0
              /* print, including the zeros */
              for (int ii=0; ii<tilesize; ii++) {
                for (int jj=0; jj<tilesize; jj++) {
                    const int row = i*tilesize + ii;
                    const int col = j*tilesize + jj;
                    printf("m_a(%d,%d) = %lf\n", row, col, m_a[row*matrixdim+col]);
                }
              }
              fflush(stdout);
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    /* end correctness checking */

    ta_destroy(&g_a);
    ta_destroy(&g_b);
    ta_destroy(&g_c);

    MPI_Finalize();
    return 0;
}
