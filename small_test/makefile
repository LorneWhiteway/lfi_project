DEFAULT_GOAL := small_test

small_test: small_test.o
	g++ -o small_test small_test.o -lhdf5

small_test.o: small_test.cpp
	g++ -c -Wall small_test.cpp 


.PHONY: clean

clean:
	rm -f small_test
	rm -f small_test.o

