#include <stdio.h>
#include <hdf5.h>
#include <unistd.h>

int main() {
	
	htri_t h = H5Fis_hdf5("./CMakeLists.txt");
	printf("%d\n", h);

	hid_t dataset = H5Pcreate( H5P_DATASET_CREATE );
	(void)dataset;

    printf("About to sleep\n");
	sleep(1);
	printf("About to exit\n");
	
	
    return 0; 
} 