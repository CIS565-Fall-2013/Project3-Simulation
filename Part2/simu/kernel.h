#ifndef _KERNEL_H
#define _KERNEL_H

int initCUDABuffer( cudaGraphicsResource* &vboResource, int massXNum, int massYNum );

void updateVelWrapper( cudaGraphicsResource* &vboResource, float2 restlen, int massXNum, int massYNum );
void updatePosWrapper( cudaGraphicsResource* &vboResource, cudaGraphicsResource* normalVboSrc,
                       float dt, float elapse, float windFactor,
                       float2 restlen, int massXNum, int massYNum, glm::vec3 &sphere, float radius );

void updateNormalWrapper( cudaGraphicsResource* &normalvboSrc, int massXNum, int massYNum );
int cleanupCUDA();

#endif