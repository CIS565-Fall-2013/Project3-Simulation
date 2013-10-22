#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glFields.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define blockSize 32
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

//TODO: Add parameterized control for each boid having different rules.
struct BoidProps{
	glm::vec3 pos;
	glm::vec3 vel;
	float rollAngle;
};

void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt);
void initCuda(int N, glm::vec3 mapDims);
void cudaUpdateVBO(float * vbodptr);
#endif
