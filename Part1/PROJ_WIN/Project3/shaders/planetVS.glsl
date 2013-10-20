#version 330

in vec4 Position;

out VertexData
{
	int vid;
} VertexOut;

void main(void)
{
	gl_Position = Position;
	VertexOut.vid = gl_VertexID;
}
