#version 330

layout (location = 0) in vec4 Position;
layout (location = 2) in float Angle;


void main(void)
{
	gl_Position = Position;
}
