#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "constraints.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define blockSize 8
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

void checkCUDAError(const char *msg, int line);
void update(float dt);
void initCuda(int dimx, int dimz, float dx, float dz, float y0, float mass);
void cudaUpdateVAO(float * vbodptr, float * nbodptr);
void freeCuda();
#endif