#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include <mpi.h>

//#define TA_DEBUG
#define TA_COMM_DUP

/* BEGIN TYPES */

typedef enum { GET, PUT, SUM } rma_e;

typedef struct
{
    MPI_Win  win;
    void *   baseptr;
    int      total_ntiles;
    int      local_ntiles;
    size_t   tilesize;
#ifdef TA_COMM_DUP
    MPI_Comm wincomm;
#else
    int      comm_size;
#endif
} ta_t;

typedef ta_t cntr_t;

/* END TYPES */


/* BEGIN API */

int ta_create(MPI_Comm comm, int ntiles, size_t tilesize, ta_t * tilearray);
int ta_destroy(ta_t * tilearray);

void ta_print_tile(char * label, size_t n, const double * restrict ptr);

int ta_memset_array(ta_t tilearray, double value);
int ta_print_array(ta_t tilearray);
int ta_sync_array(ta_t tilearray);

int ta_get_tile(ta_t tilearray, int tile, double * buffer);
int ta_put_tile(ta_t tilearray, int tile, const double * buffer);
int ta_sum_tile(ta_t tilearray, int tile, const double * buffer);

int ta_get_comm(ta_t tilearray, MPI_Comm old, MPI_Comm * new);
int ta_get_ntiles(ta_t tilearray);
size_t ta_get_tilesize(ta_t tilearray);

int cntr_create(MPI_Comm comm, ta_t * cntr);
int cntr_destroy(ta_t * cntr);
int cntr_zero(ta_t tilearray);
int cntr_read(cntr_t tilearray, long * result);
int cntr_fadd(ta_t tilearray, long incr, long * result);

/* END API */

