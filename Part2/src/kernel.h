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

#define scene_scale 1.1f

#define STIFFNESS 100.0f
#define REF_DENSITY 1000.0f
#define VISCOSITY 75.0f
#define SIMULATIONSCALE 1.0f
#define GRAVITY glm::vec3(0,0,-9.8)
#define RADIUS 0.0451f
#define MASS REF_DENSITY*4.0f/3.0f*PI*RADIUS*RADIUS*RADIUS
#define kernelSize 3*RADIUS
#define kernelSizeSqr kernelSize * kernelSize
#define pi 3.14159265358979f

#define XMAX 1.0f * scene_scale/2
#define YMAX 1.0f * scene_scale/2
#define ZMAX 13.0f * scene_scale/2
#define EPSIL 0.0001f
#define COLLIDE_BUFFER 0.2f
#define DRAG 0.97f

#define B_XMAX 0.50f
#define B_YMAX 0.50f
#define B_ZMIN 1.1f


void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt);
void cudaNBodyUpdateVelocityVerletWrapper(float dt);
void cudaSPHUpdateWrapper(float dt);
void resetSim(int N);
void initCuda(int N);
void freeCuda(int N);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, int width, int height);
#endif
