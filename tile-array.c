#include "tile-array.h"

/* INTERNAL */

int tai_create(MPI_Comm comm, MPI_Datatype dt, int ntiles, size_t tilesize, ta_t * tilearray)
{
    int rc;

#ifdef TA_DEBUG
    MPI_Comm_dup(comm, &(tilearray->wincomm));
#endif

    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    int local_ntiles = 0;
    {
        int rem = ntiles % comm_size;
        int tmp = ntiles / comm_size;
        if (comm_rank < rem ) tmp++;
        local_ntiles = tmp;
#ifdef TA_DEBUG
        if (comm_rank==0) {
            printf("nprocs    = %d\n", comm_size);
            printf("ntiles    = %d\n", ntiles);
        }
        printf("%d: local_ntiles = %d\n", comm_rank, local_ntiles);

        int sum_local_ntiles = 0;
        MPI_Allreduce(&local_ntiles, &sum_local_ntiles, 1, MPI_INT, MPI_SUM, comm);
        if (comm_rank==0) {
            printf("SUM of local_ntiles = %d\n", sum_local_ntiles);
        }
#endif
    }

    tilearray->total_ntiles = ntiles;
    tilearray->local_ntiles = local_ntiles;
    tilearray->tilesize     = tilesize;

    int typesize;
    MPI_Type_size(dt, &typesize);

    MPI_Aint winsize = (MPI_Aint)tilesize * local_ntiles * typesize;
#ifdef TA_DEBUG
    printf("%d: winsize = %ld bytes\n", comm_rank, winsize);
#endif

    MPI_Info winfo = MPI_INFO_NULL;
    MPI_Info_create(&winfo);
    MPI_Info_set(winfo, "same_size", "true"); 
    MPI_Info_set(winfo, "accumulate_ordering", ""); 
    MPI_Info_set(winfo, "accumulate_ops", "same_op"); 

    rc = MPI_Win_allocate(winsize, sizeof(double) /* disp */, winfo, 
                          comm, (void*) &(tilearray->baseptr), &(tilearray->win));

    MPI_Win_lock_all(MPI_MODE_NOCHECK, tilearray->win);

    /* Zero array */
    {
        size_t n = tilesize * local_ntiles;
        if (dt==MPI_DOUBLE) {
            double * dptr = tilearray->baseptr;
            for (size_t i=0; i<n; i++) {
                dptr[i] = 0.0;
            }
        } else if (dt==MPI_FLOAT) {
            float * fptr = tilearray->baseptr;
            for (size_t i=0; i<n; i++) {
                fptr[i] = 0.0f;
            }
        } else if (dt==MPI_LONG) {
            long * lptr = tilearray->baseptr;
            for (size_t i=0; i<n; i++) {
                lptr[i] = 0L;
            }
        } else if (dt==MPI_INT) {
            int * iptr = tilearray->baseptr;
            for (size_t i=0; i<n; i++) {
                iptr[i] = 0;
            }
        } else {
            char * cptr = tilearray->baseptr;
            memset(cptr, '\0', winsize);
        }
        MPI_Win_sync(tilearray->win);
        MPI_Barrier(comm);
    }

    MPI_Info_free(&winfo);

    return rc;
}

/* UTILITY */

int ta_get_comm(ta_t tilearray, MPI_Comm old, MPI_Comm * new)
{
    MPI_Group group;
    MPI_Win_get_group(tilearray.win, &group);
    MPI_Comm_create(old, group, new);
    MPI_Group_free(&group);
    return MPI_SUCCESS;
}

int ta_get_ntiles(ta_t tilearray)
{
    return (tilearray.total_ntiles);
}

size_t ta_get_tilesize(ta_t tilearray)
{
    return (tilearray.tilesize);
}

/* DATA PARALLEL OPS */

int ta_print_array(ta_t tilearray)
{
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

#ifdef TA_DEBUG
    MPI_Barrier(tilearray.wincomm);
#endif

    double * ptr = tilearray.baseptr;
    printf("tilesize = %ld \n", tilearray.tilesize);
    printf("local_ntiles = %d \n", tilearray.local_ntiles);
    size_t n = (tilearray.tilesize) * (tilearray.local_ntiles);
    printf("%d: printing %ld\n", comm_rank, n);
    for (size_t i=0; i<n; i++) {
        printf("%d: [%ld]=%lf\n", comm_rank, i, ptr[i]);
    }
    fflush(stdout);

#ifdef TA_DEBUG
    MPI_Barrier(tilearray.wincomm);
#endif

    return MPI_SUCCESS;
}

int ta_memset_array(ta_t tilearray, double value)
{
#ifdef TA_DEBUG
    MPI_Barrier(tilearray.wincomm);
#endif

    MPI_Win_sync(tilearray.win);

    double * ptr = tilearray.baseptr;
    size_t n = (tilearray.tilesize) * (tilearray.local_ntiles);
    for (size_t i=0; i<n; i++) {
        ptr[i] = value;
    }

    MPI_Win_sync(tilearray.win);

#ifdef TA_DEBUG
    MPI_Barrier(tilearray.wincomm);
#endif

    return MPI_SUCCESS;
}

/* SYNCHRONIZATION */

int ta_sync_array(ta_t tilearray)
{
    return MPI_Win_flush_all(tilearray.win);
}

/* DATA MOVEMENT */

int tai_rma_tile(ta_t tilearray, int tile, double * buffer, rma_e rma)
{
    int rc;

    if (tile>(tilearray.total_ntiles)) {
        printf("tai_rma_tile: tile (%d) out-of-range (%d) \n", tile, tilearray.total_ntiles);
    }

    int comm_size;
#ifdef TA_DEBUG
    MPI_Comm_size(tilearray.wincomm, &comm_size);
#else
    comm_size = tilearray.comm_size;
#endif

    int tilesize  = (tilearray.tilesize);

    int      target_proc = tile % comm_size;
    int      tile_offset = tile / comm_size;
    MPI_Aint win_offset  = tilesize * (MPI_Aint)tile_offset;
#ifdef TA_DEBUG
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    printf("%d: RMA trgt=%d, toff=%d, woff=%ld\n", comm_rank, target_proc, tile_offset, win_offset);
#endif
    
    if (rma==GET) {
        rc = MPI_Get(buffer, tilesize, MPI_DOUBLE,
                     target_proc, win_offset, tilesize, MPI_DOUBLE,
                     tilearray.win);
    } else if (rma==PUT) {
        rc = MPI_Put(buffer, tilesize, MPI_DOUBLE,
                     target_proc, win_offset, tilesize, MPI_DOUBLE,
                     tilearray.win);
    } else if (rma==SUM) {
    rc = MPI_Accumulate(buffer, tilesize, MPI_DOUBLE,
                        target_proc, win_offset, tilesize, MPI_DOUBLE,
                        MPI_SUM, tilearray.win);
    } else  {
        printf("Impossible\n");
        MPI_Abort(MPI_COMM_WORLD, rma);
    }

    MPI_Win_flush_local(target_proc, tilearray.win);

    return rc;
}

int ta_get_tile(ta_t tilearray, int tile, double * buffer)
{
    return tai_rma_tile(tilearray, tile, buffer, GET);
}

int ta_put_tile(ta_t tilearray, int tile, const double * buffer)
{
    return tai_rma_tile(tilearray, tile, (double*)buffer, PUT);
}

int ta_sum_tile(ta_t tilearray, int tile, const double * buffer)
{
    return tai_rma_tile(tilearray, tile, (double*)buffer, SUM);
}

/* ALLOCATION and DEALLOCATION */

int ta_create(MPI_Comm comm, int ntiles, size_t tilesize, ta_t * tilearray)
{
    return tai_create(comm, MPI_DOUBLE, ntiles, tilesize, tilearray);
}

int ta_destroy(ta_t * tilearray)
{
#ifdef TA_DEBUG
    MPI_Barrier(tilearray->wincomm);
    MPI_Comm_free(&(tilearray->wincomm));
#endif
    MPI_Win_unlock_all(tilearray->win);
    MPI_Win_free(&(tilearray->win));
    return MPI_SUCCESS;
}

/* NXTVAL type thing */

int cntr_create(MPI_Comm comm, cntr_t * cntr)
{
    return tai_create(comm, MPI_LONG, 1, 1, cntr);
}

int cntr_destroy(cntr_t * cntr)
{
    return ta_destroy(cntr);
}

int cntr_zero(cntr_t tilearray)
{
    long zero = 0, junk;
    MPI_Fetch_and_op(&zero, &junk, MPI_LONG, 0, (MPI_Aint)0, MPI_REPLACE, tilearray.win);
    MPI_Win_flush(0, tilearray.win);
    return MPI_SUCCESS;
}

int cntr_fadd(cntr_t tilearray, long incr, long * result)
{
    MPI_Fetch_and_op(&incr, result, MPI_LONG, 0, (MPI_Aint)0, MPI_SUM, tilearray.win);
    MPI_Win_flush(0, tilearray.win);
    return MPI_SUCCESS;
}

int cntr_read(cntr_t tilearray, long * result)
{
    MPI_Fetch_and_op(NULL, result, MPI_LONG, 0, (MPI_Aint)0, MPI_NO_OP, tilearray.win);
    MPI_Win_flush(0, tilearray.win);
    return MPI_SUCCESS;
}

/* THE END */
