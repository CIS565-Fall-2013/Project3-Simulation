#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define blockSize 256
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define SHARED 0

#define RNEIGHBOR 12
#define ALIGNMENT 0.7
#define SEPARATION 2.4
#define COHESION 0.1
#define RANGE 300


void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt, int time);
void initCuda(int N, int P);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, float* nbodptr, int width, int height);
void cudaUpdateVBOPre(float * vbodptr, int width, int height);
#endif
