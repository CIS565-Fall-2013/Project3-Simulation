#version 330

in vec4 Position;

out VertexData {
	float orientation;
} VertexOut;

void main(void)
{
	gl_Position = vec4(Position.xy,0,1);
	VertexOut.orientation = Position.z;
}
