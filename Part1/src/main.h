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
#include <time.h>
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
																// See glslUtility::createProgram for the glBindAttribute call...
GLuint positionLocation = 0;									// use to access the Position attribute in vertex shaders 
GLuint texcoordsLocation = 1;									// use to access Texcoords attribute in vertex shaders
GLuint colorLocation = 1;

const char *attributeLocations[] = { "Position", "Texcoords" };
const char *planetAttributeLocations[] = {"Position", "Color" };

GLuint pbo = (GLuint)NULL;										// texture pixel buffer object. PBO points to the buffer object with float4 with each w
																// component containing the height at a particular texel
GLuint planeVBO = (GLuint)NULL;
GLuint planeTBO = (GLuint)NULL;
GLuint planeIBO = (GLuint)NULL;
GLuint planetVBO = (GLuint)NULL;
GLuint planetIBO = (GLuint)NULL;
GLuint planetCBO = (GLuint)NULL;

GLuint displayImage;											// texture map that will be sampled in heightVS. The values are stored in the PBO (float4) with each w
																// component containing the height at a particular texel

GLuint program[2];												// heightField shaders and planet shaders

// constants for accessing program[i] where i is either equal to HEIGHT_FIELD or PASS_THROUGH
const unsigned int HEIGHT_FIELD = 0;
const unsigned int PASS_THROUGH = 1;

// resolution of the plane
const int field_width  = 800;
const int field_height = 800;

// camera parameters
float fovy = 60.0f;
float zNear = 0.10;
float zFar = 5.0;

glm::mat4 projection;
glm::mat4 view;
glm::vec3 cameraPosition(1.75,1.75,1.35);

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width=1000; int height=1000;

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


void initPBO(GLuint* pbo);
void initCuda();
void initTextures();
void initVAO();
void initShaders(GLuint * program);

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);
void shut_down(int return_code);

#endif
