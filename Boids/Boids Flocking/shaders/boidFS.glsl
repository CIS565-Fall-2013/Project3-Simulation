#version 330

uniform mat4 u_projMatrix;
uniform vec3 u_lightPos;

in vec3 Normal;
in vec3 Color;
in vec2 TexCoord;
out vec4 FragColor;

void main()
{

	//TODO: More diffuse shading
    FragColor = vec4(color,1.0);
}