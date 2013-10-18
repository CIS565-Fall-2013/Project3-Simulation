#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
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
#define PART2 1
#define NEIGHBOR			70
#define SNEIGHBOR			5
#define VIEWANGLE			140.0/180.0*PI
#define FLOCKFORCE			4e7
#define COHESIONFORCE		2e7
#define SEPARATIONFORCE		1e7
#define RANGE				200.0

void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt);
void initCuda(int N);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, int width, int height);

struct isOutofRange{
	__host__ __device__
	bool operator()(const glm::vec4 v)
	{
		return (glm::length(glm::vec3(v.x,v.y,v.z)) > RANGE);
	}
};

#endif
