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

    ta_create(MPI_COMM_WORLD, 1, count, &g_a);
    ta_memset_array(g_a, (double)(me+1));
    if (count<100) ta_print_array(g_a);
    double * tmp = malloc(count * sizeof(double));
    for (int i=0; i<count; i++) tmp[i] = exp(1.+(double)i/count);
    ta_sum_tile(g_a, 0, tmp);
    ta_sync_array(g_a);
    if (count<100) ta_print_array(g_a);
    ta_destroy(&g_a);

    ta_create(MPI_COMM_WORLD, np, count, &g_b);
    ta_memset_array(g_b, (double)(me+1));
    if (count<100) ta_print_array(g_b);
    ta_destroy(&g_b);

    ta_create(MPI_COMM_WORLD, np*np, count, &g_c);
    ta_memset_array(g_c, (double)(me+1));
    if (count<100) ta_print_array(g_c);
    ta_destroy(&g_c);

    MPI_Finalize();
    return 0;
}
