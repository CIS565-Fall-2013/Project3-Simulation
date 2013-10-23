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

#define BLOCK_SIZE 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define SHARED 1

#define STARTING_VEL 1.0
#define FLOCKING_RADIUS 20

#define COHESION_COEFF 0.0001
#define ALIGNMENT_COEFF 0.02
#define ALIGNMENT_BONUS 1.1
#define SEPARATION_COEFF 1
#define SEPARATION_RADIUS 10
#define SEPARATION_EPSILON 1

#define ATTRACTOR_RADIUS 120
#define ATTRACTOR_COEFF .0001
#define ATTRACTOR_DEFLECTION 0.2

void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt);
void initCuda(int N);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, int width, int height);
#endif
