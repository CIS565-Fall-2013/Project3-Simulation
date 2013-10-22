#version 330

in vec4 Position;
in vec3 Color;

out vColor
{
	vec3 color;
}vertex;

void main(void)
{
	gl_Position = Position;
	vertex.color = Color;
}
