#version 330

uniform mat4 u_projMatrix;
uniform sampler2D u_height;

in vec4 Position;
in vec2 Texcoords;

out vec2 v_Texcoords;
out float f_height;

void main(void)
{
    v_Texcoords = Texcoords;
    vec4 pos = Position;
    f_height = texture2D(u_height, Texcoords).w;
    //pos.z = -0.01-clamp(f_height,0.0,2.0);
	pos.z=-0.5f;//-f_height/2.0f;
    pos = u_projMatrix * pos;
    gl_Position = pos;
}