#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <unistd.h>
#include <limits.h>
#include <string.h>



int main( int argc, char *argv[]) {

	int my_rank;
	int numprocs;


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	//std::cout << "My rank is " << my_rank << std::endl;
	
	
	if (my_rank != 0) {
		char hostname[HOST_NAME_MAX];
		gethostname(hostname, HOST_NAME_MAX);
		char message[200];
		message[0] = 0;
		sprintf(message, "Greetings from process %d on hostname %s!\0", my_rank, hostname);
		int dest = 0;
		int tag = 0;
		MPI_Send(message, strlen(message), MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	}
	else {
		for (int source = 1; source < numprocs; ++source) {
			char message[200];
			memset(message, 0, sizeof(message));
			int tag = 0;
			MPI_Status status;
			MPI_Recv(message, sizeof(message), MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
			std::cout << message << std::endl;
		}
	}
	MPI_Finalize();

}
