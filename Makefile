CC     := /opt/mpich/dev/intel/debug/bin/mpicc
CFLAGS := -g3 -Wall -O2 -std=c99 -Wl,-no_pie
#CC     := /opt/mpich/dev/clang/debug/bin/mpicc
#CFLAGS := -g3 -Wall -O2 -std=c99

LIBS=-mkl=sequential
OBJS=tile-array.o tile-blas.o
TESTS=test-basic.x test-cntr.x test-daxpy.x test-block-sparse-fill.x test-block-sparse-contract.x

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

