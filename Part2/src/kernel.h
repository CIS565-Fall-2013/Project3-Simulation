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

#define RNEIGHBOR 10
#define ALIGNMENT 1
#define SEPARATION 0.7
#define COHESION 0.2
#define RANGE 200

//glm::vec3 calAlignment(glm::vec4 my, glm::vec4 their);
//glm::vec3 calSeparation(glm::vec4 my, glm::vec4 their);
//glm::vec3 calCohesion(glm::vec4 my, glm::vec4 their);
//glm::vec3 navieFlocking();
//glm::vec3 sharedMemFlocking();
void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt, int time);
void initCuda(int N, int P);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, int width, int height);
void cudaUpdateVBOPre(float * vbodptr, int width, int height);
#endif
