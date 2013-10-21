uniform mat4 u_projMatrix;
attribute vec4 Position;

void main(void)
{
    vec4 pos = u_projMatrix * Position;
	pos.z += 0.01;
	gl_Position = pos;
}
