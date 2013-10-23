#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_modelviewMatrix;

in vec4 Position;



void main(void)
{
	gl_Position = Position;
}
