#include "tile-blas.h"

void matmul(int m, int n, int k, const double * restrict a, const double * restrict b, double * restrict c)
{
    const double alpha = 1.0;
    const double beta  = 1.0;
    const int rowa=m, rowc=m;
    const int colb=n, colc=n;
    const int cola=k, rowb=k;
    DGEMM_SYMBOL("n","n",&rowa,&colb,&cola,&alpha,a,&rowa,b,&rowb,&beta,c,&rowc);
}
