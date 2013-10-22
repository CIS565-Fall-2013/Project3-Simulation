#include "glFields.h"

//-------------------------------
//------------GL STUFF-----------
//-------------------------------

const char *attributeLocations[] = { "vs_position", "vs_up", "vs_forward", "vs_color", "vs_shape" };

GLuint boidVBO = (GLuint)NULL;
GLuint boidIBO = (GLuint)NULL;
GLuint displayImage;
GLuint program[1];
