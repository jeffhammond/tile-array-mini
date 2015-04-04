CC     := /opt/mpich/dev/intel/debug/bin/mpicc
CFLAGS := -g3 -Wall -O2 -std=c99 -qopenmp -Wl,-no_pie
#CC     := /opt/mpich/dev/clang/debug/bin/mpicc
#CFLAGS := -g3 -Wall -O2 -std=c99

# DEBUG_LEVEL (inclusive with lower levels)
# 1 = task scheduling
# 2 = matrix elements
# 3 = array elements
CFLAGS += -DDEBUG_LEVEL=2

LIBS=-mkl=sequential
OBJS=tile-array.o tile-blas.o
TESTS=test-basic.x test-cntr.x test-daxpy.x test-block-sparse-fill.x test-block-sparse-contract.x test-block-sparse-contract-omp.x

all: $(TESTS)

%.x: %.o $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) $(LIBS) -o $@

tile-blas.o: tile-blas.c tile-blas.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c tile-array.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)

