#version 330

in vec4 Position;
in vec3 Color;
in vec3 Velocity;

out vColor
{
	vec3 color;
	vec3 velocity;
}vertex;


void main(void)
{
	gl_Position = Position;
	vertex.color = Color;
	vertex.velocity = Velocity;
}
