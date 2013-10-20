#version 330

uniform mat4 u_projMatrix;

in vec4 Position;

out vec3 fs_normal;
out vec3 fs_light_vector;
out vec3 fs_color;

void main(void)
{
	gl_Position = Position;
}
