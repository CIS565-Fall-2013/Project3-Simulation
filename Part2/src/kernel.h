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

#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define SHARED 1
#define EULER 1
#define MIN_WANDER .1
#define MAX_WANDER .5
#define MIN_VELOCITY .1
#define MAX_VELOCITY 3.0
#define BLIND_ANGLE 5/180 * PI
#define WALL_DEPTH 50
#define BUFFER 10

void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt);
void initCuda(int N);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, float* vbodptrn, int width, int height);
#endif
