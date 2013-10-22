#version 330

uniform mat4 u_projMatrix;
uniform mat4 u_viewMatrix;
uniform vec3 u_lightPos;


in vec3 fs_EyeNormal;
in vec3 fs_Color;
in vec3 fs_EyeLightVector;
in vec2 fs_TexCoord;


out vec4 FragColor;

void main()
{

	 vec4 diffuseColor = vec4(fs_Color, 1.0);
    
    //calculate diffuse term and clamp to the range [0, 1]
    float diffuseTerm = clamp(dot(normalize(fs_EyeNormal), normalize(fs_EyeLightVector)), 0.0, 1.0);
    
	//TODO: Use texture coordinates in some way
    FragColor = diffuseColor*(diffuseTerm*0.9+0.1);//Small 5% ambient term
}