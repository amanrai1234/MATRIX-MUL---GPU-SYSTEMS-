## CUDA-matrix-Multiplication
Here I used CUDA to compute matrix multiplication and evaluated other frameworks such as OPENMP, MPI and Pthreads






## MATRIX MULTIPLICATION USING CUDA:

There are multiple ways to parallelize the code while doing matrix multiplication. We can 
use the method of doing parallelization using shared limited access to the global memory 
that can be useful (not for small matrix sizes). I have done the basic version of matrix 
parallelization so that I can compare the openmp and mpi versions which were doing 
normal matrix multiplication:

## Steps for my appraoch:

1] I initialized the row and column of the matrix using built in x and y dimensions inside the 
cuda (for blockIdx and blockDim), here I have not initialized separate row and column
variable (m x n) for a matrix because the program is run on a square matrix of nsize.

2]I have computed the matrix multiplication (entire row X entire column =One element) and 
then used result variable to pass the values to the C array variable.

3]in the main function, I have used cudaMallocHost(used this only to compute the double 
size time, it is currently commented in the code)

4] I have used the float variables for the CPU memory and for the device memory I have 
used cudaMalloc function and initialized three variables and used cudaMemcpy to copy 
variables form ram to GPU (for implementing unified I used Cudamallocmanaged (this is 
currently commented in the code))

5] I have initialized the kernel using dim3 variable and kept the block size as (32,32), I did 
not keep any blocks in the Z direction.

6]I then used cuda synchronize to synchronize the thread, and then used cudaMemcpy to 
copy result from the device to host.

7] To verify all the result I have used the verify function and I was able to get correct values 
for all.

## Problems encountered:

The main problem that got me stuck during Implementation was when I used the wrong 
initialization for rows and columns and I was getting runtime errors and apart from that I 
was using the cudaMallocHost and creating three varibles because I assumed that 
CudaMemcpy may not use the normal malloc. Apart from this I was using (32,32) for grid 
dimensions, but it was better for creating dimGrids depending on nsize.
## Comparison of versions:

## USING Float size:(Main Reference Comparisions)
       128     512      1024     2048

GPU   0.0744   0.0567   0.5670  0.0974

SEQ   0.00168 0.18911 1.3487 47.6565

OPENMP 0.00407 0.03434 0.30945 11.6471



The above is the comparision of CUDA vs OPENMP vs Sequential version of the code and I 
have seen that for small tasks using CUDA is not as beneficial for doing the big tasks (but 
again this can be contradicted as we want to parallelize the task and it is more fruitful than
sequential tasks). S0 initially CUDA was taking more time because it is communicating using 
many threads and this is giving taking time and int this aspect openMP and sequential(this is 
best for very small task – 128 matrix size). 

But as we increased the matrix size the time for CUDA became constant and this was 
because of high level parallelism that it does and we can see that it came to a level where 
openMP is taking almost 12 seconds for task and Cuda is taking around 0.2 seconds this is 
actually insane and beneficial for high level computation. 

## Extras:

Comparison of explicit memory management vs CUDA 

Unified Memory performance:

Using explicit memory management:

          128    512   1024    2048

GPU     0.0744 0.0567 0.5670 0.0974

SEQ     0.00168 0.18911 1.3487 47.6565

OPENMP  0.00407 0.03434 0.30945 11.6471



Using Cuda Unified Memory:

          128     512     1024     2048

GPU    0.0947 0.0667628 0.0665615 0.11327

SEQ    0.00244 0.139677 1.33041 47.3782

OPENMP 0.00807 0.0218039 0.316809 11.6471

The thing about unified memory is that its easy to implement as we are using 
Mallocmanaged Function that takes care of all the copying part(GPU to CPU and vice versa)
but there are better ways that we can limit the access to the global memory or shared 
memory depending upon circumstances and make better parallelism. While comparing the 
explicit vs unified memory we can see that there is not much difference in the runtime (this 
can be different if we can compare it using complicated code and calculate the values many 
times and take average). This is because in unified memory the system decides the access to 
global or shared memory and in my case that thing is not making much of difference
Test and compare performance of doubles vs floats.

Usually double increased the time of computation compared to the float and the it was not 
as precise as the floating-point operation and sometimes we can reduce the precision and 
round of the value to misinterpret result but yes, I have observed that the amount of time 
that It took for double to complete was more than that of the floating point operation.
Some versions of Cuda may or may not support double precision.
USING Float size:(Main Reference Comparisons)
        128     512      1024    2048
        
GPU   0.0744   0.0567  0.5670  0.0974

SEQ   0.00168  0.18911 1.3487 47.6565

OPENMP 0.00407 0.03434 0.30945 11.6471

USING double size:

            128      512       1024     2048
      
GPU      0.09415   0.0847047  0.07360  0.10985

SEQ     0.01168371 0.574244 1.5407   48.7247

OPENMP  0.0097   0.0898416  0.35915  13.8581

## Conclusion:

## GPU will outperform in higher order of matrices

I given a task that has high parallelism GPU’s can be used and the task can be parallelized 
easily. But If we are implementing something with shared memory multiprocessing then 
OPENMP could be better. If we are choosing to interact over a network of systems in that
case MPI could be a better choice too














