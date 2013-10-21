#ifndef _KERNEL_H
#define _KERNEL_H

int initCUDABuffer( cudaGraphicsResource* &vboResource, int massXNum, int massYNum );

void updateVelWrapper( cudaGraphicsResource* &vboResource, float2 restlen, int massXNum, int massYNum );
void updatePosWrapper( cudaGraphicsResource* &vboResource, float dt, int massXNum, int massYNum );
int cleanupCUDA();

#endif