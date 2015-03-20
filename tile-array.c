#include "tile-array.h"

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
    MPI_Barrier(tilearray.wincomm);

    int comm_rank;
    MPI_Comm_rank(tilearray.wincomm, &comm_rank);

    double * ptr = tilearray.baseptr;
    printf("tilesize = %ld \n", tilearray.tilesize);
    printf("local_ntiles = %d \n", tilearray.local_ntiles);
    size_t n = (tilearray.tilesize) * (tilearray.local_ntiles);
    printf("%d: printing %ld\n", comm_rank, n);
    for (size_t i=0; i<n; i++) {
        printf("%d: [%ld]=%lf\n", comm_rank, i, ptr[i]);
    }
    fflush(stdout);

    MPI_Barrier(tilearray.wincomm);

    return MPI_SUCCESS;
}

int ta_memset_array(ta_t tilearray, double value)
{
    MPI_Win_sync(tilearray.win);
    MPI_Barrier(tilearray.wincomm);

    double * ptr = tilearray.baseptr;
    size_t n = (tilearray.tilesize) * (tilearray.local_ntiles);
    for (size_t i=0; i<n; i++) {
        ptr[i] = value;
    }

    MPI_Win_sync(tilearray.win);
    MPI_Barrier(tilearray.wincomm);

    return MPI_SUCCESS;
}

/* SYNCHRONIZATION */

int ta_sync_array(ta_t tilearray)
{
    return MPI_Win_flush_all(tilearray.win);
}

/* DATA MOVEMENT */

int ta_get_tile(ta_t tilearray, int tile, double * buffer)
{
    int rc;

    if (tile>(tilearray.total_ntiles)) {
        printf("tile (%d) out-of-range (%d) \n", tile, tilearray.total_ntiles);
    }

    int comm_size;
    MPI_Comm_size(tilearray.wincomm, &comm_size);

    int tilesize  = (tilearray.tilesize);

    int      target_proc = tile % comm_size;
    int      tile_offset = tile / comm_size;
    MPI_Aint win_offset  = tilesize * (MPI_Aint)tile_offset;
    
    rc = MPI_Get(buffer, tilesize, MPI_DOUBLE,
                 target_proc, win_offset, tilesize, MPI_DOUBLE,
                 tilearray.win);

    MPI_Win_flush_local(target_proc, tilearray.win);

    return rc;
}

int ta_put_tile(ta_t tilearray, int tile, const double * buffer)
{
    int rc;

    if (tile>(tilearray.total_ntiles)) {
        printf("tile (%d) out-of-range (%d) \n", tile, tilearray.total_ntiles);
    }

    int comm_size;
    MPI_Comm_size(tilearray.wincomm, &comm_size);

    int tilesize  = (tilearray.tilesize);

    int      target_proc = tile % comm_size;
    int      tile_offset = tile / comm_size;
    MPI_Aint win_offset  = tilesize * (MPI_Aint)tile_offset;
    
    rc = MPI_Put((double*)buffer, tilesize, MPI_DOUBLE,
                 target_proc, win_offset, tilesize, MPI_DOUBLE,
                 tilearray.win);

    MPI_Win_flush_local(target_proc, tilearray.win);

    return rc;
}

int ta_sum_tile(ta_t tilearray, int tile, const double * buffer)
{
    int rc;

    if (tile>(tilearray.total_ntiles)) {
        printf("tile (%d) out-of-range (%d) \n", tile, tilearray.total_ntiles);
    }

    int comm_size;
    MPI_Comm_size(tilearray.wincomm, &comm_size);

    int tilesize  = (tilearray.tilesize);

    int      target_proc = tile % comm_size;
    int      tile_offset = tile / comm_size;
    MPI_Aint win_offset  = tilesize * (MPI_Aint)tile_offset;
    
    rc = MPI_Accumulate((double*)buffer, tilesize, MPI_DOUBLE,
                        target_proc, win_offset, tilesize, MPI_DOUBLE,
                        MPI_SUM, tilearray.win);

    MPI_Win_flush_local(target_proc, tilearray.win);

    return rc;
}

/* ALLOCATION and DEALLOCATION */

int ta_create(MPI_Comm comm, int ntiles, size_t tilesize, ta_t * tilearray)
{
    int rc;

    //MPI_Comm_dup(comm, &(tilearray->wincomm));
    tilearray->wincomm = comm;

    int comm_rank, comm_size;
    MPI_Comm_rank(tilearray->wincomm, &comm_rank);
    MPI_Comm_size(tilearray->wincomm, &comm_size);

    int local_ntiles = 0;
    {
        int rem = ntiles % comm_size;
        int tmp = ntiles / comm_size;
        if (comm_rank < rem ) tmp++;
        local_ntiles = tmp;
#ifdef TA_DEBUG
        if (comm_rank==0) {
            printf("nprocs    = %d\n", comm_size);
            printf("myrank    = %d\n", comm_rank);
            printf("ntiles    = %d\n", ntiles);
        }
        printf("%d: local_ntiles = %d\n", comm_rank, local_ntiles);

        int sum_local_ntiles = 0;
        MPI_Allreduce(&local_ntiles, &sum_local_ntiles, 1, MPI_INT, MPI_SUM, comm);
        if (comm_rank==0) {
            printf("sum_local_ntiles = %d\n", sum_local_ntiles);
        }
#endif
    }

    tilearray->total_ntiles = ntiles;
    tilearray->local_ntiles = local_ntiles;
    tilearray->tilesize     = tilesize;

    MPI_Aint winsize = (MPI_Aint)tilesize * local_ntiles * sizeof(double);
#ifdef TA_DEBUG
    printf("%d: winsize = %ld\n", comm_rank, winsize);
#endif

    MPI_Info winfo = MPI_INFO_NULL;
    //MPI_Info_create(&winfo);
    //MPI_Info_set(winfo, "same_size", "true"); 
    //MPI_Info_set(winfo, "accumulate_ordering", ""); 
    //MPI_Info_set(winfo, "accumulate_ops", "same_op"); 

    rc = MPI_Win_allocate(winsize, sizeof(double) /* disp */, winfo, 
                          comm, (void*) &(tilearray->baseptr), &(tilearray->win));

    MPI_Win_lock_all(MPI_MODE_NOCHECK, tilearray->win);

    /* Zero array */
    {
        double * ptr = tilearray->baseptr;
        size_t n = tilesize * local_ntiles;
        for (size_t i=0; i<n; i++) {
            //printf("%d: i=%ld\n", comm_rank, i); fflush(stdout);
            ptr[i] = 0.0;
        }
        MPI_Win_sync(tilearray->win);
        MPI_Barrier(tilearray->wincomm);
    }

    //MPI_Info_free(&winfo);

    return rc;
}

int ta_destroy(ta_t * tilearray)
{
    MPI_Barrier(tilearray->wincomm);
    MPI_Win_unlock_all(tilearray->win);
    //MPI_Comm_free(&(tilearray->wincomm));
    MPI_Win_free(&(tilearray->win));
    return MPI_SUCCESS;
}

/* THE END */
