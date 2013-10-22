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

enum {EULER, VELVERLET, RK4, NUMBEROFINTEGRATIONS};

using namespace std;

class cam
{
	public:
	float rad;
	float theta, phi;
	glm::vec3 pos;
	cam();
	void reset();
	void setFrame();
};

//-------------------------------
//------------GL STUFF-----------
//-------------------------------

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Texcoords" };
GLuint pbo = (GLuint)NULL;
GLuint planeVBO = (GLuint)NULL;
GLuint planeTBO = (GLuint)NULL;
GLuint planeIBO = (GLuint)NULL;
GLuint planetVBO = (GLuint)NULL;
GLuint planetIBO = (GLuint)NULL;
GLuint displayImage;
GLuint program[2];

const unsigned int HEIGHT_FIELD = 0;
const unsigned int PASS_THROUGH = 1;

const int field_width_pbo = 100;
const int field_height_pbo = 100;

const int field_width  = 10;
const int field_height = 10;

float fovy = 60.0f;
float zNear = 0.10;
float zFar = 25.0;

int integration = EULER;

glm::mat4 projection;
glm::mat4 view;
glm::mat4 projectionView;
glm::vec3 cameraPosition(1.75,1.75,1.35);

cam mouseCam;
float lastx = 0.0f;
float lasty = 0.0f;
float motion=0.2;
bool LMB=false;
bool MMB=false;
bool RMB=false;

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width=1280; int height=720;
int numberOfIterations = 0;
float totalElapsedTime = 0.0f;

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
void mouseMovement(int x, int y);
void mouseMovementUpdate(int x, int y);
void mouseClick(int button, int state, int x, int y);

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
