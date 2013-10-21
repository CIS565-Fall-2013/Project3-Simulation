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

//#define blockSize 128
//int blockSize = 128;
#define TILE_SIZE 8
//#define NUM_TILES 1 //N_FOR_VIS/TILE_SIZE
#define N_FOR_VIS 32
#define COLLISION_RAD 5
#define NUM_TILES N_FOR_VIS/TILE_SIZE
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define SHARED 0

//0 is symplectic Euler
//1 is forward Euler
#define INTEGRATION_TYPE 0

void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt, int blockSize);
void initCuda(int N, int blockSize);
void cudaUpdatePBO(float4 * pbodptr, int width, int height, int blockSize);
void cudaUpdateVBO(float * vbodptr, int width, int height, int blockSize);
void cleanupCuda();
#endif
