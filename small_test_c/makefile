DEFAULT_GOAL := small_test

small_test: small_test.o
	gcc -L/usr/local/software/spack/spack-0.11.2/opt/spack/linux-rhel7-x86_64/gcc-5.4.0/hdf5-1.10.1-z7bre2kdm5mup3ooctobwo2aojtje7uo/lib/ -o small_test small_test.o -lhdf5
	#gcc -o small_test small_test.o -lhdf5

small_test.o: small_test.c
	gcc -c -Wall small_test.c 


.PHONY: clean

clean:
	rm -f small_test
	rm -f small_test.o

