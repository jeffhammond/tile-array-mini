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

    size_t count = (argc>1) ? atol(argv[1]) : 16385;

    ta_t g_x, g_y;

    int ntiles = np*6;
    ta_create(MPI_COMM_WORLD, ntiles, count, &g_x);
    ta_create(MPI_COMM_WORLD, ntiles, count, &g_y);

    ta_memset_array(g_x, 10.0);
    ta_memset_array(g_y, 0.3333);

    cntr_t nxtval;
    cntr_create(MPI_COMM_WORLD, &nxtval);
    if (me==0) cntr_zero(nxtval);

    MPI_Barrier(MPI_COMM_WORLD);

    const double alpha = 0.7;

    long counter;
    cntr_fadd(nxtval, 1, &counter);
    for (size_t t=0; t<ntiles; t++) {
        if (t==counter) {
            printf("rank %d got task %ld\n", me, t); fflush(stdout);
            double * tmp = malloc(count * sizeof(double));
            ta_get_tile(g_x, t, tmp);
            for (size_t i=0; i<count; i++) tmp[i] *= alpha;
            ta_sum_tile(g_y, t, tmp);
            free(tmp);
            cntr_fadd(nxtval, 1, &counter);
        }
    }
    ta_sync_array(g_y);

    if (count<100) ta_print_array(g_y);

    cntr_destroy(&nxtval);

    ta_destroy(&g_x);
    ta_destroy(&g_y);

    MPI_Finalize();
    return 0;
}
