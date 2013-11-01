#version 330

in vec4 Position;
in vec4 Velocities;

out vec4 velocity;

void main(void)
{
	gl_Position = Position;

	velocity = Velocities;
}
