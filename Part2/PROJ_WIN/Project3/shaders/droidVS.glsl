#version 330

uniform mat4 u_modelviewMatrix;
uniform mat4 u_projMatrix;
uniform vec3 u_cameraPos;
uniform vec3 u_lightPos;
uniform vec3 u_lightColor;

in vec4 Position;
in vec4 Normal;
in vec3 Color;

out vec3 fs_normal;
out vec3 fs_position;
out vec3 fs_color;

out vec3 fs_camera_pos;
out vec3 fs_light_position;
out vec3 fs_light_color;

void main(void)
{
	fs_color = Color;
	vec4 temp = transpose(inverse(u_modelviewMatrix)) * Normal;
	fs_normal = temp.xyz;

	fs_camera_pos = u_cameraPos;
	fs_light_color = u_lightColor;
	fs_light_position = u_lightPos - Position.xyz;

	gl_Position = u_projMatrix * u_modelviewMatrix * Position;
}
