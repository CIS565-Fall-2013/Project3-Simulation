#version 330

in vec4 Position;
in vec4 Velocity;

out vec4 pos;
out vec4 vel;

void main(void)
{
	//gl_Position = Position;
	pos = Position;
	vel = Velocity;
}
