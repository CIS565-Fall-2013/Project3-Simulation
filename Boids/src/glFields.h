#pragma once

#include "glm/glm.hpp"
#include <GL/glew.h>
#include <GL/glut.h>

//-------------------------------
//------------GL STUFF-----------
//-------------------------------

const GLuint positionLocation = 0;//vec4
const GLuint upLocation = 1;//vec3
const GLuint forwardLocation = 2;//vec3
const GLuint colorLocation = 3;//vec3
const GLuint shapeLocation = 4;//vec4

//Interleaving parameters
const GLuint boidVBOStride = 4 + 3 + 3 + 3 + 4;//vec4+3*vec3+vec4
const GLuint boidVBO_PositionOffset = 0;
const GLuint boidVBO_UpOffset = 4;
const GLuint boidVBO_ForwardOffset = 7;
const GLuint boidVBO_ColorOffset = 10;
const GLuint boidVBO_ShapeOffset = 13;

extern const char *attributeLocations[];

extern GLuint boidVBO;
extern GLuint boidIBO;
extern GLuint displayImage;
extern GLuint program[];

const unsigned int BOIDS = 0;
