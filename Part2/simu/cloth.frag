# version 400

layout (location = 0) out vec4 gl_FragColor;

uniform sampler2D SpriteTex;

//in vec2 texcoord;
in vec3 mynormal;

void main (void) 
{       
	gl_FragColor = vec4(1.0,1.0,1.0,1.0);//texture2D( tex1, texcoord );
}