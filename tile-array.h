#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define TA_DEBUG

/* BEGIN TYPES */

typedef struct
{
    MPI_Win  win;
    double * baseptr;
    int      total_ntiles;
    int      local_ntiles;
    size_t   tilesize;
    MPI_Comm wincomm;
} ta_t;

/* END TYPES */


/* BEGIN API */

int ta_create(MPI_Comm comm, int ntiles, size_t tilesize, ta_t * tilearray);
int ta_destroy(ta_t * tilearray);

int ta_memset_array(ta_t tilearray, double value);
int ta_sync_array(ta_t tilearray);

int ta_get_tile(ta_t tilearray, int tile, double * buffer);
int ta_put_tile(ta_t tilearray, int tile, const double * buffer);
int ta_sum_tile(ta_t tilearray, int tile, const double * buffer);

int ta_get_comm(ta_t tilearray, MPI_Comm old, MPI_Comm * new);
int ta_get_ntiles(ta_t tilearray);
size_t ta_get_tilesize(ta_t tilearray);

/* END API */

