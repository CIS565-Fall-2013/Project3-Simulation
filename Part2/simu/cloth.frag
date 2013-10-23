# version 400

layout (location = 0) out vec4 gl_FragColor;

uniform sampler2D SpriteTex;
uniform vec4 kd;
uniform vec4 ks;
uniform float shininess;
uniform vec4 lightPos;

//in vec2 texcoord;
in vec3 mynormal;
in vec4 myvertex;

void main (void) 
{   
	vec3 lightVec;
	vec3 eyeVec;
	vec3 normalVec;
	vec3 H;

	eyeVec = normalize( -myvertex.xyz );
	normalVec = normalize( mynormal );
	lightVec = normalize( lightPos.xyz - myvertex.xyz );
	H = normalize( lightVec + eyeVec );
	
	vec4 color = kd * max( dot( normalVec, lightVec ), 0.0f ) +
	             ks * pow( max( dot( normalVec, H ), 0.0f ), shininess );

	gl_FragColor = color;
	
    //gl_FragColor = vec4(1.0,1.0,1.0,1.0);//texture2D( tex1, texcoord );
}