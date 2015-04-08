#ifdef __INTEL_COMPILER
#define PRAGMA_NOVECTOR _Pragma("novector")
#else
#define PRAGMA_NOVECTOR
#endif

#include "tile-array.h"

int main(int argc, char * argv[])
{
    int requested=MPI_THREAD_SERIALIZED, provided;
    MPI_Init_thread(&argc, &argv, requested, &provided);

    if (provided<requested) MPI_Abort(MPI_COMM_WORLD, provided);

    int np, me;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    /* times to increment counter */
    int reps = (argc>1) ? atoi(argv[1]) : 10000;
    if (me==0) printf("reps = %ld\n", reps);

    MPI_Barrier(MPI_COMM_WORLD);

    cntr_t nxtval;
    cntr_create(MPI_COMM_WORLD, &nxtval);
    if (me==0) cntr_zero(nxtval);
    MPI_Barrier(MPI_COMM_WORLD);

    PRAGMA_NOVECTOR
    for (int i=0; i<reps; i++) {
        long out;
        cntr_fadd(nxtval, 1, &out);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    long total;
    if (me==0) {
        cntr_read(nxtval, &total);
        printf("total = %ld (correct=%d)\n", total, reps*np);
        fflush(stdout);
    }

    cntr_destroy(&nxtval);

    if (me==0) printf("SUCCESS\n");

    MPI_Finalize();
    return 0;
}
