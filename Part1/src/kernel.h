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

void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt, bool customSimulation);
void initCuda(int N, const glm::vec4 &camera_position);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, int width, int height);
void setDevicePrefetch (bool prefetchEnabled);

void moveCameraToNextFlock (glm::vec3 &cameraPos);
void	setCameraUpdate (bool shouldCameraUpdate);
glm::vec4	getCurrentCameraPosition ();
void		setCurrentCameraPosition (const glm::vec4 &camera_position);

inline __device__ glm::vec3 safeNormalize (glm::vec3 vectorToBeNormalized);		// normalize only if length > 0
__device__ bool isApproximately (const float &a, const float &b);
__device__ glm::vec3 sharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos);
__device__ glm::vec3 pfSharedMemAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos);
__device__ glm::vec3 FlockGlobal (int N, float DT, glm::vec4 my_pos, glm::vec4 *pos, glm::vec3 *vel);
__device__ glm::vec3 naiveAcc(int N, glm::vec4 my_pos, glm::vec4 * their_pos);
#endif
