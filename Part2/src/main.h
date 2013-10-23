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
#include <ctime>

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
//GLuint texcoordsLocation = 1;
GLuint colorLocation = 1;
GLuint normalLocation = 2;

//const char *attributeLocations[] = { "Position", "Texcoords" };
const char *attributeLocations[] = { "Position", "Color", "Normal" };
GLuint pbo = (GLuint)NULL;
GLuint planeVBO = (GLuint)NULL;
GLuint planeTBO = (GLuint)NULL;
GLuint planeIBO = (GLuint)NULL;
GLuint planetVBO = (GLuint)NULL;
GLuint planetIBO = (GLuint)NULL;
GLuint displayImage;
GLuint program[2];


GLuint droidIBO = (GLuint)NULL;
GLuint droidVBO = (GLuint)NULL;
GLuint droidNBO = (GLuint)NULL;
GLuint droidCBO = (GLuint)NULL;

GLuint predatorVBO = (GLuint)NULL;
GLuint predatorIBO = (GLuint)NULL;


const unsigned int DROID = 0;
const unsigned int PREDATOR = 1;

const int field_width  = 800;
const int field_height = 800;

float fovy = 45.0f;
float zNear = 0.10;
float zFar = 100.0;

glm::mat4 projection;
glm::mat4 modelview;
glm::mat4 view;
glm::vec3 cameraPosition(1.35,1.75,0);
glm::vec3 lightPosition(0.35,2.75,3);
glm::vec3 lightColor(0.9,0.9,1);
int mouse_old_x, mouse_old_y;
float rotY = 45.0f, rotX = 45.0f;
float eye_distance = 2.0f;
glm::vec3 lookat(0.0f, 0.0f, 0.0f);
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
void mouse(int button, int state, int x, int y);
void mouseLeft(int x, int y);
void mouseMiddle(int x, int y);
void mouseRight(int x, int y);

void camera();

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void init(int argc, char* argv[]);


void initPBO(GLuint* pbo);
void initCuda();
void initTextures();
void initVAO();
void initShaders(GLuint * program);
void initDroid();
void initPredator();

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);
void shut_down(int return_code);

#endif
