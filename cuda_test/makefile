DEFAULT_GOAL := cuda_test

cuda_test: cuda_test.cu
	nvcc -o cuda_test cuda_test.cu

.PHONY: clean

clean:
	rm -f cuda_test
	