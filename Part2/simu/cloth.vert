# version 400

layout (location = 0) in vec4 glVertex;
layout (location = 1) in vec3 glNormal;

//out vec2 texcoord;
out vec3 mynormal;
out vec4 myvertex;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat3 NormalMatrix;

void main() {

	gl_Position = ProjectionMatrix * ModelViewMatrix * glVertex ; 
    mynormal = NormalMatrix * glNormal ; 
    myvertex = ModelViewMatrix * glVertex ; 
}