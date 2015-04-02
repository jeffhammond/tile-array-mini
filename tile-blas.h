#include <stdint.h>

/* dgemm_ is common, so default to it and let the user override if necessary */
#ifndef DGEMM_SYMBOL
#define DGEMM_SYMBOL dgemm_
#endif

#define CONST const

#ifdef BLAS_64BIT_INTEGER
    void DGEMM_SYMBOL(char* , char* ,CONST int64_t* , CONST int64_t* , CONST int64_t* , CONST double* , CONST double* , CONST int64_t* , CONST double* , CONST int64_t* , CONST double* , double* , CONST int64_t* );
#else /* CONST int32 is usually the default */
    void DGEMM_SYMBOL(char* , char* ,CONST int32_t* , CONST int32_t* , CONST int32_t* , CONST double* , CONST double* , CONST int32_t* , CONST double* , CONST int32_t* , CONST double* , double* , CONST int32_t* );
#endif

void matmul(int m, int n, int k, const double * restrict a, const double * restrict b, double * restrict c);
