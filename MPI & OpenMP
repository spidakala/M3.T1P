#include <iostream>
#include<stdio.h>
#include<time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <mpi.h>
#include <omp.h>


using namespace std;

// maximum size of matrix
int max_size = 4;

int **_A, **_B, **_C;

void init(int** &_A, int rows, int cols, bool initialise) {
    _A = (int **) malloc(sizeof(int*) * rows * cols); // number of rows * size of int* address in the memory
    int* tmp = (int *) malloc(sizeof(int) * cols * rows);

    for (int i = 0 ; i < max_size ; i++) {
        _A[i] = &tmp[i * cols];
    }

    if(!initialise) return;
    for(long i = 0 ; i < rows; i++) { 
        for(long j = 0 ; j < cols; j++) {
            _A[i][j] = rand() % 10; // any number less than 100 }
        } 
    }

void displayMatrices(int** Matrics)
{
    printf("Matix\n");
    for (int i = 0; i < max_size; i++) {
        for (int j = 0; j < max_size; j++) 
            cout << Matrics[i][j] << " ";
        cout << endl;
    }
}

void multi_omp(int num_processes)
{
    int sum = 0;
    #pragma omp parallel
    {

    #pragma omp for 54
    for (int k = 0; k < max_size/num_processes; k++)
    {
        for (int i = 0; i < max_size; i++)
        {
            for (int j = 0; j < max_size; j++)
            {
                sum += _A[k][j] * _B[j][i];
                }
                _C[k][i] = sum;
                
                sum=0;
                }
            }    
        }
    }

void head(int num_processes)
{ 
    init(_A, max_size, max_size, true), init(_B, max_size, max_size, true), init(_C, max_size, max_size, false);
    //displayMatrices(_A);
    //displayMatrices(_B);


    //my plan is to scatter _A based on number of processes and broadcast _B to all nodes
    int num_rows_per_process_from_A = max_size / num_processes;
    int num_elements_to_bcast = (max_size * max_size);
    int num_elements_to_scatter_or_gather = (max_size * max_size) / num_processes;

    MPI_Scatter(&_A[0][0], num_elements_to_scatter_or_gather , MPI_INT , &_A , 0, MPI_INT, 0 , MPI_COMM_WORLD);
    MPI_Bcast(&_B[0][0], num_elements_to_bcast , MPI_INT , 0 , MPI_COMM_WORLD);

    struct timeval timecheck;
    double time_elapsed = 0.0;
    gettimeofday(&timecheck, NULL);
    long timeofday_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

    // Matrix Multiplication
    multi_omp(num_processes);

    gettimeofday(&timecheck, NULL);
    long timeofday_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
   
    time_elapsed = timeofday_end - timeofday_start; 
    printf("time elapsed: %.5f\n", time_elapsed);

    MPI_Gather(MPI_IN_PLACE, num_elements_to_scatter_or_gather , MPI_INT, &_C[0][0] , num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);
    //send the results back to the head node for merging and printing 103

    //displayMatrices(_C);

}
void node(int process_rank, int num_processes)
{

    int num_rows_per_process_from_A = max_size / num_processes;
    int num_elements_to_bcast = (max_size * max_size);
    int num_elements_to_scatter_or_gather = (max_size * max_size) / num_processes;

    //receive my rows of matrix _A, and all _B
    init(_A, num_rows_per_process_from_A , max_size, true), init(_B, max_size, max_size, false), init(_C, num_rows_per_process_from_A, max_size, false);

    MPI_Scatter(NULL, num_elements_to_scatter_or_gather , MPI_INT , &_A[0][0], num_elements_to_scatter_or_gather, MPI_INT, 0 , MPI_COMM_WORLD);
    MPI_Bcast(&_B[0][0], num_elements_to_bcast , MPI_INT , 0 , MPI_COMM_WORLD);


    multi_omp(num_processes);

    MPI_Gather(&_C[0][0], num_elements_to_scatter_or_gather , MPI_INT, NULL, 􏰁→ num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);
}
int main(int argc, char **argv)
{
    if (argc > 1)
        max_size = atoi(argv[1]);

    MPI_Init(NULL, NULL);
    
    int num_processes; 
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    int process_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    if(process_rank == 0)
        head(num_processes);
    else
        node(process_rank, num_processes);

    MPI_Finalize();

}
