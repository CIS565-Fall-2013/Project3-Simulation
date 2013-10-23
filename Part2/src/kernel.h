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
#define SHARED 0
#define EULER 0
#define RK2 1
#define RK4 2
#define FLOCK_TO_TARGET 0

using glm::vec3;
using glm::vec4;
using glm::length;
using glm::normalize;

void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt, vec3 target, bool recall);
void initCuda(int N);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, int width, int height);
void cudaUpdateSBO(float * sbodptr, int width, int height);
#endif
