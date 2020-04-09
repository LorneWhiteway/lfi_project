DEFAULT_GOAL := avx_test

avx_test: avx_test.cpp
	g++ -march=x86-64 -o avx_test avx_test.cpp 

.PHONY: clean

clean:
	rm -f avx_test
