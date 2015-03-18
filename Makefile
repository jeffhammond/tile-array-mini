CC     = mpicc
CFLAGS = -g -Wall -O2 -std=c99

LIBS =

TESTS=test-basic.x

all: $(TESTS)

%.x: %.o tile-array.o
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%.o: %.c tile-array.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o
	-rm -f $(TESTS)

