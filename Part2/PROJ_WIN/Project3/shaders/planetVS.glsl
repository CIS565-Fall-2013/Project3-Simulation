#version 330

in vec4 Position;
out vec2 gs_Velocity;

void main(void)
{
	gl_Position = vec4(Position.xy, 0.0, 1.0);
	gs_Velocity = Position.zw;
}