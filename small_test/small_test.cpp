#include <iostream>
#include <hdf5.h>

int main() {
	
	htri_t h = H5Fis_hdf5("foo");
	(void)h;

    std::cout << "Hello World" << std::endl;
    return 0; 
} 