#include "glFields.h"

//-------------------------------
//------------GL STUFF-----------
//-------------------------------

const char *attributeLocations[] = { "vs_position", "vs_up", "vs_forward", "vs_color", "vs_shape" };

GLuint boidVBO = (GLuint)NULL;
GLuint boidIBO = (GLuint)NULL;
GLuint displayImage;
GLuint program[1];

float fovy = 60.0f;
float zNear = 0.10;
float zFar = 25.0;

glm::mat4 projection = glm::mat4(1.0f);
glm::mat4 view = glm::mat4(1.0f);
glm::vec3 cameraPosition(5.4,0.0,0.5);
glm::vec3 lightPosition(10,0,10);
glm::vec3 mapDims(5,5,5);