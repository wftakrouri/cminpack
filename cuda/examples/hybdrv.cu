#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <string.h>
#include <cminpack.h>


#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <hybrd1.cu>

#define real __cminpack_real__

struct refnum {
    int nprob, nfev, njev;
};

#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() Runtime API error in file <%s>, line %i : %s.\n",
                file, line, cudaGetErrorString( err) );
       // exit(-1);
    }
}


//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
const unsigned int NUM_OBSERVATIONS = 15; 
const unsigned int NUM_PARAMS = 9; 
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
// 
//  fixed arrangement of threads to be run 
// 
const unsigned int NUM_THREADS = 512;
const unsigned int NUM_THREADS_PER_BLOCK = 128;
const unsigned int NUM_BLOCKS = NUM_THREADS / NUM_THREADS_PER_BLOCK;

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//
// The struct for returning results from the GPU
//

typedef struct    
{
    real fnorm1;
	real fnorm2;
    int info;
    real solution[NUM_PARAMS];
} ResultType;

__cminpack_attr__
void vecfcn(int n, const real *x, real *fvec, int nprob)
{
	int k;
	for (k=0; k<n; ++k)
    {
      fvec[k] = (3.0-x[k]*2.0)*x[k]+1.0;
      if (k>0)
        fvec[k] -= x[k-1];
      if (k<n-1)
        fvec[k] -= x[k+1]*2.0;
    }
} 

__cminpack_attr__  //__device__ 
int fcn_nn(void *p, int n, const real *x, real *fvec, int iflag)
{
    struct refnum *hybrdtest = (struct refnum *)p;
    vecfcn(n, x, fvec, hybrdtest->nprob);
    if (iflag == 1) {
        hybrdtest->nfev++;
    }
    if (iflag == 2) {
        hybrdtest->njev++;
    }

    return 0;
}

__cminpack_attr__
void hybipt(int n, real *x, int nprob, real factor)
{
    /* Local variables */
    real h__;
    int j;
    real tj;

    --x;

    /* Function Body */

    /*     selection of initial point. */

    switch (nprob) {
	case 1:
            /*     rosenbrock function. */
            x[1] = -1.2;
            x[2] = 1.;
            break;
	case 2:
            /*     powell singular function. */
            x[1] = 3.;
            x[2] = -1.;
            x[3] = 0.;
            x[4] = 1.;
            break;
	case 3:
            /*     powell badly scaled function. */
            x[1] = 0.;
            x[2] = 1.;
            break;
	case 4:
            /*     wood function. */
            x[1] = -3.;
            x[2] = -1.;
            x[3] = -3.;
            x[4] = -1.;
            break;
	case 5:
            /*     helical valley function. */
            x[1] = -1.;
            x[2] = 0.;
            x[3] = 0.;
            break;
	case 6:
            /*     watson function. */
            for (j = 1; j <= n; ++j) {
                x[j] = 0.;
            }
            break;
	case 7:
            /*     chebyquad function. */
            h__ = 1. / (real) (n+1);
            for (j = 1; j <= n; ++j) {
                x[j] = (real) j * h__;
            }
            break;
	case 8:
            /*     brown almost-linear function. */
            for (j = 1; j <= n; ++j) {
                x[j] = .5;
            }
            break;
	case 9:
	case 10:
            /*     discrete boundary value and integral equation functions. */
            h__ = 1. / (real) (n+1);
            for (j = 1; j <= n; ++j) {
                tj = (real) j * h__;
                x[j] = tj * (tj - 1.);
            }
            break;
	case 11:
            /*     trigonometric function. */
            h__ = 1. / (real) (n);
            for (j = 1; j <= n; ++j) {
                x[j] = h__;
            }
            break;
	case 12:
            /*     variably dimensioned function. */
            h__ = 1. / (real) (n);
            for (j = 1; j <= n; ++j) {
                x[j] = 1. - (real) j * h__;
            }
            break;
	case 13:
        case 14:
            /*     broyden tridiagonal and banded functions. */
            for (j = 1; j <= n; ++j) {
                x[j] = -1.;
            }
            break;
    }

    /*     compute multiple of initial point. */

    if (factor == 1.) {
	return;
    }
    if (nprob == 6) {
        for (j = 1; j <= n; ++j) {
            x[j] = factor;
        }
    } else {
        for (j = 1; j <= n; ++j) {
            x[j] = factor * x[j];
        }
    }
} /* initpt_ */


//--------------------------------------------------------------------------
// the kernel in the GPU
//--------------------------------------------------------------------------
__global__ void mainKernel(ResultType  pResults[])
{
   int i,ic,k,n,ntries=1;
   struct refnum hybrdtest;
   int info;
   real factor,fnorm1,fnorm2,tol;

   real fnm[60];
   real fvec[40];
   real x[40];
   real wa[2660];
   const int lwa = 2660;

    tol = sqrt(__cminpack_func__(dpmpar)(1));
	
    ic = 0;
    factor = 1.;
	hybrdtest.nprob=6;

    hybipt(NUM_PARAMS,x,hybrdtest.nprob,factor);
    vecfcn(NUM_PARAMS,x,fvec,hybrdtest.nprob);
    fnorm1 = __cminpack_func__(enorm)(n,fvec);
    hybrdtest.nfev = 0;
    hybrdtest.njev = 0;
    info = __cminpack_func__(hybrd1)(__cminpack_param_fcn_nn__ &hybrdtest,NUM_PARAMS,x,fvec,tol,wa,lwa);
	fnorm2 = __cminpack_func__(enorm)(NUM_PARAMS,fvec);
	hybrdtest.njev /= NUM_PARAMS;
	 factor *= 10.;
    
//----------------------------------
//save the results in global memory
//---------------------------------- 
   int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
   pResults[threadId].fnorm1 = fnorm1;
   pResults[threadId].fnorm2 = fnorm2;
   pResults[threadId].info = info;
   for (int j=0; j<9; j++) 
   {
	   pResults[threadId].solution[j] = x[j];
	
   }
} 

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
int main (int argc, char** argv)
{
	int pause;
    fprintf (stderr, "\Hybdrv Algorithm \n");

    //  ...............................................................
    // choose the fastest GPU device
    //  ...............................................................
    unsigned int GPU_ID = 1;  // not actually :-)
    // unsigned int GPU_ID =  cutGetMaxGflopsDeviceId() ;
    cudaSetDevice(GPU_ID); 
    fprintf (stderr, " CUDA device chosen = %d \n", GPU_ID);

    // ....................................................... 
    //  get memory in the GPU to store the results 
    // ....................................................... 
    ResultType * results_GPU = 0;
    cutilSafeCall( cudaMalloc( &results_GPU,  NUM_THREADS * sizeof(ResultType)) );

    // ....................................................... 
    //  get memory in the CPU to store the results 
    // ....................................................... 
    ResultType * results_CPU = 0;
    cutilSafeCall( cudaMallocHost( &results_CPU, NUM_THREADS * sizeof(ResultType)) );

    // ....................................................... 
    //  Launch the kernel
    // ....................................................... 
    fprintf (stderr, " \nlaunching the kernel num. blocks = %d, threads per block = %d\n total threads = %d\n\n", NUM_BLOCKS, NUM_THREADS_PER_BLOCK, NUM_THREADS);

    mainKernel<<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>> ( results_GPU );

    // ....................................................... 
    // Block CPU until all threads are exectued
    // ....................................................... 
    cudaThreadSynchronize(); 
    fprintf (stderr, " GPU processing done \n\n");

    // ....................................................... 
    // copy back to CPU the results
    // ....................................................... 
    cutilSafeCall( cudaMemcpy( results_CPU, results_GPU, 
                               NUM_THREADS * sizeof(ResultType),
                               cudaMemcpyDeviceToHost
                               ) );
	
	// ....................................................... 
    // print all computed results 
    // ....................................................... 
    
	
	
	// ....................................................... 
    // check all the threads computed the same results
    // ....................................................... 
	for(int kk=0;kk<NUM_PARAMS;kk++)
		fprintf (stderr, "%lf\n",results_CPU[0].solution[kk]);
	
	
	
	bool ok = true;
    for (unsigned int i = 1; i<NUM_THREADS; i++) {
	if ( memcmp (&results_CPU[0], &results_CPU[i], sizeof(ResultType)) != 0) {
            // warning: may the padding bytes be different ?
            ok = false;
	}
    } // for

	if (ok) {
		fprintf (stderr, " !!! all threads computed the same results !!! \n\n");
    } 
	else{
		fprintf (stderr, "ERROR in results of threads \n");
    }

    cutilSafeCall(cudaFree(results_GPU));
    cutilSafeCall(cudaFreeHost(results_CPU));
    cudaThreadExit();
	
	cudaFree(results_GPU);
    cudaFree(results_CPU);
    scanf("%d",&pause);
} 