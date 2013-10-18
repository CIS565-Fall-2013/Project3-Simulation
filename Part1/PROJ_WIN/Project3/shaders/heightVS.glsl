#version 400

in vec4 Position;
in vec2 Texcoords;

out vec4 v_Position;
out vec2 v_Texcoords;

void main(void)
{

	v_Position = Position;
	v_Texcoords = Texcoords;

/*
    v_Texcoords = Texcoords;
    vec4 pos = Position;
    f_height = texture2D(u_height, Texcoords).w;
    pos.z = -0.01-clamp(f_height,0.0,2.0);
    pos = u_projMatrix * pos;
    gl_Position = pos;
*/
}