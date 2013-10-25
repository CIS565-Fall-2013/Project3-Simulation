uniform mat4 u_projMatrix;

attribute vec4 Position;
attribute vec4 Normal;

varying vec4 v_Normal;

void main(void)
{
	v_Normal = Normal;
    vec4 pos = u_projMatrix * Position;
    gl_Position = pos;
}