#include "tile-array.h"

int ta_get_comm(ta_t tilearray, MPI_Comm old, MPI_Comm * new)
{
    MPI_Group group;
    MPI_Win_get_group(tilearray->win, &group);
    MPI_Comm_create(old, group, new);
    MPI_Group_free(&group);
    return MPI_SUCCESS;
}

int ta_get_ntiles(ta_t tilearray)
{
    return (tilearray->ntiles);
}

size_t ta_get_tilesize(ta_t tilearray)
{
    return (tilearray->tilesize);
}

int ta_get_tile(ta_t tilearray, int tile, double * buffer)
{
    int rc;

    if (tile>(tilearray->ntiles)) {
        printf("tile (%d) out-of-range (%d) \n", tile, tilearray->ntiles);
    }

    int comm_size = (tilearray->comm_size);
    int tilesize  = (tilearray->tilesize);

    int      target_proc = tile % comm_size;
    int      tile_offset = tile / comm_size;
    MPI_Aint win_offset  = tilesize * (MPI_Aint)tile_offset;
    
    rc = MPI_Get(buffer, tilesize, MPI_DOUBLE,
                 target_proc, win_offset, tilesize, MPI_DOUBLE,
                 tilearray->win);

    return rc;
}

int ta_put_tile(ta_t tilearray, int tile, const double * buffer)
{
    int rc;

    if (tile>(tilearray->ntiles)) {
        printf("tile (%d) out-of-range (%d) \n", tile, tilearray->ntiles);
    }

    int comm_size = (tilearray->comm_size);
    int tilesize  = (tilearray->tilesize);

    int      target_proc = tile % comm_size;
    int      tile_offset = tile / comm_size;
    MPI_Aint win_offset  = tilesize * (MPI_Aint)tile_offset;
    
    rc = MPI_Put(buffer, tilesize, MPI_DOUBLE,
                 target_proc, win_offset, tilesize, MPI_DOUBLE,
                 tilearray->win);

    return rc;
}

int ta_sum_tile(ta_t tilearray, int tile, const double * buffer)
{
    int rc;

    if (tile>(tilearray->ntiles)) {
        printf("tile (%d) out-of-range (%d) \n", tile, tilearray->ntiles);
    }

    int comm_size = (tilearray->comm_size);
    int tilesize  = (tilearray->tilesize);

    int      target_proc = tile % comm_size;
    int      tile_offset = tile / comm_size;
    MPI_Aint win_offset  = tilesize * (MPI_Aint)tile_offset;
    
    rc = MPI_Accumulate(buffer, tilesize, MPI_DOUBLE,
                        target_proc, win_offset, tilesize, MPI_DOUBLE,
                        MPI_SUM, tilearray->win);

    return rc;
}

int ta_create(MPI_Comm comm, int ntiles, size_t tilesize, ta_t tilearray)
{
    int rc;

    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    tilearray->ntiles    = ntiles;
    tilearray->tilesize  = tilesize;
    tilearray->comm_size = comm_size;

    int my_ntiles = 0;
    {
        int rem = ntiles % comm_size;
        int tmp = ntiles / comm_size;
        if (comm_rank < rem ) tmp++;
        my_ntiles = tmp;
#ifdef TA_DEBUG
        if (comm_rank==0) {
            printf("nprocs    = %d\n", comm_size);
            printf("myrank    = %d\n", comm_rank);
            printf("ntiles    = %d\n", ntiles);
        }
        printf("%d: my_ntiles = %d\n", comm_rank, my_ntiles);

        int sum_my_ntiles = 0;
        MPI_Allreduce(&my_ntiles, &sum_my_ntiles, 1, MPI_INT, MPI_SUM, comm);
        if (comm_rank==0) {
            printf("sum_my_ntiles = %d\n", sum_my_ntiles);
        }
#endif
    }

    MPI_Aint winsize = (MPI_Aint)tilesize * my_ntiles;

    MPI_Info winfo;
    MPI_Info_create(&winfo);
    MPI_Info_set(winfo, "same_size", "true"); 
    MPI_Info_set(winfo, "accumulate_ordering", ""); 
    MPI_Info_set(winfo, "accumulate_ops", "same_op"); 

    rc = MPI_Win_allocate(winsize, sizeof(double) /* disp */, winfo, 
                          comm, (void*) &(tilearray->baseptr), &(tilearray->win));

    MPI_Win_lock_all(MPI_MODE_NOCHECK, tilearray->win);

    MPI_Info_free(&winfo);

    return rc;
}

int ta_destroy(ta_t tilearray)
{
    int rc;

    MPI_Win_unlock_all(tilearray->win);

    rc = MPI_Win_free(&(tilearray->win));

    return rc;
}
