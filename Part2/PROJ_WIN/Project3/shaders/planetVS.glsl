#version 330

in vec4 Position;

void main(void)
{
	gl_Position = vec4(Position.xy,0,1);
}
