#include <iostream>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <mpi.h>
#include <CL/cl.h>

using namespace std; 

#define BILLION 1000000000L;
int SZ=4;
const int TS = 4;
int **_A, **_B, **_C, **D;

void init(int **&_A, int rows, int cols, bool initialise);
void print(int **_A, int rows, int cols);
void head(int num_processes);
void node(int process_rank, int num_processes);

cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;

cl_event event = NULL;
int err;

cl_mem bufA, bufB, bufC; 33
size_t local[2] = {TS, TS};
size_t global[2] = {SZ, SZ}; // number of rows and cols or basically the number of threads with indices i and j where i is the row and j is the col of the matric _C 36
cl_device_id create_device();
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);

void setup_openCL_device_context_queue_kernel();
void setup_kernel_memory(int num_rows_per_process_from_A);
void copy_kernel_args(int num_rows_per_process_from_A);
void free_memory();

int main(int argc, char **argv)
{

//no change -- init MPI, run head or node depending on the process rank
//if I'm process Zero, then call head
//otherwise, call node

    if (argc > 1)
        SZ = atoi(argv[1]);

    MPI_Init(NULL, NULL); 56
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
    
void head(int num_processes)
{
    //MPI
    ////OpenCL
    //MPI
    
    
    init(_A, SZ, SZ, true), init(_B, SZ, SZ, true), init(_C, SZ, SZ, false), init(D, SZ, SZ, false);
    // print(_A, SZ, SZ);
    // print(_B, SZ, SZ);

    int num_rows_per_process_from_A = SZ / num_processes;
    int num_elements_to_bcast = (SZ * SZ);
    int num_elements_to_scatter_or_gather = (SZ * SZ) / num_processes;
    
    MPI_Scatter(&_A[0][0], num_elements_to_scatter_or_gather, MPI_INT, &_A[0][0], 0, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_B[0][0], num_elements_to_bcast, MPI_INT, 0, MPI_COMM_WORLD);

    struct timeval timecheck;
    double time_elapsed = 0.0;
    gettimeofday(&timecheck, NULL);
    long timeofday_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
    
//Start of OpenCL 
    
    local[0] = num_rows_per_process_from_A, local[1] = SZ;
    global[0] = num_rows_per_process_from_A, global[1] = SZ; 

    setup_openCL_device_context_queue_kernel();
    setup_kernel_memory(num_rows_per_process_from_A);
    copy_kernel_args(num_rows_per_process_from_A);
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
    clWaitForEvents(1, &event);

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, num_rows_per_process_from_A * SZ * sizeof(int), &_C[0][0], 0, NULL, NULL);

    //end of OpenCL 
    gettimeofday(&timecheck, NULL);
    long timeofday_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec /1000;
    time_elapsed = timeofday_end - timeofday_start;
    printf("time elapsed: %.5f\n", time_elapsed);

    MPI_Gather(MPI_IN_PLACE, num_elements_to_scatter_or_gather, MPI_INT, &_C[0][0], 􏰁→ num_elements_to_scatter_or_gather, MPI_INT, 0, MPI_COMM_WORLD);

    //print(_C, SZ, SZ);
    free_memory();
}

void node(int process_rank, int num_processes)
{
    //MPI
    ////OpenCL
    //MPI
    int num_rows_per_process_from_A = SZ / num_processes;
    int num_elements_to_bcast = (SZ * SZ);
    int num_elements_to_scatter_or_gather = (SZ * SZ) / num_processes;

    init(_A, num_rows_per_process_from_A, SZ, true), init(_B, SZ, SZ, true), init(_C,  num_rows_per_process_from_A, SZ, false);

    MPI_Scatter(NULL, num_elements_to_scatter_or_gather, MPI_INT, &_A[0][0], num_elements_to_scatter_or_gather, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_B[0][0], num_elements_to_bcast, MPI_INT, 0, MPI_COMM_WORLD); 

    //Start of OpenCL 

    local[0] = num_rows_per_process_from_A, local[1] = SZ;
    global[0] = num_rows_per_process_from_A, global[1] = SZ;

    setup_openCL_device_context_queue_kernel();
    setup_kernel_memory(num_rows_per_process_from_A);
    copy_kernel_args(num_rows_per_process_from_A);
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
    clWaitForEvents(1, &event);

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, num_rows_per_process_from_A * SZ * sizeof(int), &_C[0][0], 0, NULL, NULL);

//End of OpenCL

    MPI_Gather(&_C[0][0], num_elements_to_scatter_or_gather, MPI_INT, NULL, num_elements_to_scatter_or_gather, MPI_INT, 0, MPI_COMM_WORLD); 160 free_memory();
} 
void free_memory()
{
    clReleaseKernel(kernel);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
}

void copy_kernel_args(int num_rows_per_process_from_A)
{
    //NOTE that we modified the first parameter (Rows of _A)
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&num_rows_per_process_from_A);
    clSetKernelArg(kernel, 1, sizeof(int), (void *)&SZ);
    clSetKernelArg(kernel, 2, sizeof(int), (void *)&SZ);

    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufA);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&bufB);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&bufC);
    if(err<0)
    {
        perror("Couldn't create _A kernel argument");
        printf("error = %d", err);
        exit(1);
        }
}

void setup_kernel_memory(int rows)
{

//NOTE that we modified the bufA to only cover rows of _A and _C
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, rows * SZ * sizeof(int), NULL, NULL);
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, SZ * SZ * sizeof(int), NULL, NULL);
    bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, rows * SZ * sizeof(int), NULL, NULL);

// Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, rows * SZ * sizeof(int), &_A[0][0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, SZ * SZ * sizeof(int), &_B[0][0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, rows * SZ * sizeof(int), &_C[0][0], 0, NULL, NULL); 205 }

void init(int **&_A, int rows, int cols, bool initialise)
{
    _A = (int **)malloc(sizeof(int *) * rows * cols); // number of rows * size of int* address in the memory

int *tmp = (int *)malloc(sizeof(int) * rows * cols); 

    for(int = 0 ; i < SZ; i++)
    {
        _A[i] = &tmp[i * cols];
    }

    if (!initialise)
        return;

    for (long i = 0; i < rows; i++)
    {
        for (long j = 0; j < cols; j++)
        {
            _A[i][j] = rand() % 6; // any number less than 100
        }
    }
}


void print(int **_A, int rows, int cols)
{
    for(long i = 0; i < rows; i++)
    {
        for(long j = 0; j < cols; j++)
        {
            printf("%d ", _A[i][j]);
        }
        printf("\n");
    }
    printf("--------------------------------------\n");
{ 

void setup_openCL_device_context_queue_kernel()
{
    device_id = create_device();
    cl_int err;
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if(err<0)
    {
        perror("Couldn't create _A context");
        exit(1);
    }

    program = build_program(context, device_id, "./matrix_mul.cl"); 255
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if(err<0)
    {
        perror("Couldn't create _A command queue");
        exit(1);
    };

    kernel = clCreateKernel(program, "multiply_matrices", &err);
    if(err<0)
    {
        perror("Couldn't create _A kernel");
        printf("error =%d", err);
        exit(1);
    };
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log; 
    size_t program_size, log_size;
    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r"); if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1); 
    }
        fseek(program_handle, 0, SEEK_END);
        program_size = ftell(program_handle);
        rewind(program_handle);
        program_buffer = (char *)malloc(program_size + 1); 
        program_buffer[program_size] = '\0';
        fread(program_buffer, sizeof(char), program_size, program_handle); 
        fclose(program_handle);

        /* Create program from file

        Creates _A program from the source code in the add_numbers.cl file.
        Specifically, the code reads the file's content into _A char array
        called program_buffer, and then calls clCreateProgramWithSource.
        */
    program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer, &program_size, &err);
    if(err<0)
    {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);
    /* Build program 
        The fourth parameter accepts options that configure the compilation.
        These are similar to the flags used by gcc. For example, you can
        define _A macro with the option -DMACRO=VALUE and turn off optimization
        with -cl-opt-disable.*/
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err<0)
    {

    /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
        log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

    cl_device_id create_device()
    {
        cl_platform_id platform; 
        cl_device_id dev;
        int err;
/* Identify _A platform */
    err = clGetPlatformIDs(1, &platform, NULL); 
    if(err<0)
{
    perror("Couldn't identify _A platform");
    exit(1); }
    // Access _A device
    // GPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND)
    {
    // CPU
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if(err<0)
        {
            perror("Couldn't access any devices");
            exit(1); 
        }
    return dev; 
}
