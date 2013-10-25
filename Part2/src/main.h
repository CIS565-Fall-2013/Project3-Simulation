#ifndef MAIN_H
#define MAIN_H

#include <GL/glew.h>
#include <GL/glut.h>

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "glslUtility.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "kernel.h"
#include "utilities.h"

#if CUDA_VERSION >= 5000
    #include <helper_cuda.h>
    #include <helper_cuda_gl.h>
    #define compat_getMaxGflopsDeviceId() gpuGetMaxGflopsDeviceId() 
#else
    #include <cutil_inline.h>
    #include <cutil_gl_inline.h>
    #define compat_getMaxGflopsDeviceId() cutGetMaxGflopsDeviceId()
#endif

using namespace std;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------

GLuint positionLocation = 0;
GLuint normalLocation = 1;
GLuint texcoordsLocation = 2;
const char *attributeLocations[] = { "Position", "Texcoords" };
GLuint planeVBO = (GLuint)NULL;
GLuint planeNBO = (GLuint)NULL;
GLuint planeTBO = (GLuint)NULL;
GLuint planeIBO = (GLuint)NULL;
GLuint displayImage;
GLuint program;

//GLOBAL CONSTANTS
float width = 20;
float height = 20;
int xdim = 101;
int zdim = 101;
float initialY = 10; //initial y position
float mass = 1;
float dt = 0.006325;

float dx;
float dz;

float fovy = 60.0f;
float zNear = 0.01;
float zFar = 100;

glm::mat4 projection;
glm::mat4 view;
glm::vec3 cameraPosition(28, 11, 10);
glm::vec3 cameraRef(10, 4, 10);
//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int window_width=1000; int window_height=1000;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

void display();
void keyboard(unsigned char key, int x, int y);

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void init(int argc, char* argv[]);
void initCuda();
void initTextures();
void initVAO();
void initShaders(GLuint program);

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deleteTexture(GLuint* tex);
void shut_down(int return_code);

#endif
