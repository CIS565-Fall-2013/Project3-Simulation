#version 330

in vec3 WorldCoord;
in vec3 ToCam;
in vec3 Up;
in vec3 Right;
in vec2 TexCoord;
out vec4 FragColor;

void main()
{
	// takes the "texture" coordinates produces in the GS and uses them to decide where in the quad this fragment is. 
	// We discard any fragments outside of our desired radius in order to simulate the edge of the sphere.
	vec2 coord = 2.01 * (TexCoord - vec2(0.5));
    float r = length(coord);
    if (r >= 1.0) { discard; }

	//the center object is a star, so simply color it white
	float dist = length(WorldCoord);
    if(dist <= 0.01)
    {
        FragColor = vec4(1.0);
        return;
    }

	// calculating the fake intersection point and its lighting
	vec3 N = Right*-coord.x + Up*coord.y + ToCam*sqrt(1-r*r);
    vec3 L = normalize(-WorldCoord);
    float light = 0.1 + 0.9*clamp(dot(N,L),0.0, 1.0)*exp(-dist);
    vec3 color = vec3(0.4, 0.1, 0.6);
    FragColor = vec4(color*light,1.0);
}