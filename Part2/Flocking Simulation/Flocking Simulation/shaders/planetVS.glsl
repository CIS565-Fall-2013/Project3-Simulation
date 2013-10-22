#version 330

in vec4 position;
in vec3 velocity;

out vec3 gVelocity;

void main(void)
{
	gl_Position = position;
	gVelocity = velocity;
}
