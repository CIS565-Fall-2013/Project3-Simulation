#ifndef _KERNEL_H
#define _KERNEL_H

int initCUDABuffer( cudaGraphicsResource* &vboResource, int massXNum, int massYNum );

void clothSimKernelWrapper( cudaGraphicsResource* &vboResource, int massXNum, int massYNum );

int cleanupCUDA();

#endif