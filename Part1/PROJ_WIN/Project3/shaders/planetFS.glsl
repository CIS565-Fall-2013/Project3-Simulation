#version 330

in vec3 WorldCoord;
in vec3 ToCam;
in vec3 Up;
in vec3 Right;
in vec2 TexCoord;
out vec4 FragColor;

void main()
{

	vec4 ranCol = vec4(abs(ToCam.x),abs(ToCam.y),abs(ToCam.z),1.0);

	vec2 coord = 2.01 * (TexCoord - vec2(0.5));
    float r = length(coord);
    if (r >= 1.0) { discard; }

	float dist = length(WorldCoord);
    if(dist <= 0.01)
    {
        FragColor = vec4(1.0);
        return;
    }

    vec3 N = Right*-coord.x + Up*coord.y + ToCam*sqrt(1-r*r);
    vec3 L = normalize(-WorldCoord);
    float light = 0.1 + 0.9*clamp(dot(N,L),0.0, 1.0)*exp(-dist);
    vec3 color = ranCol.xyz; //vec3(0.4, 0.1, 0.6);
    FragColor = vec4(color*light,1.0);
} 