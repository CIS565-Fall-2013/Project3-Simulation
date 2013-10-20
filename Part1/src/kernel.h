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

void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt, bool customSimulation);
void initCuda(int N);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, int width, int height);

inline __device__ glm::vec3 safeNormalize (glm::vec3 vectorToBeNormalized);		// normalize only if length > 0
#endif
