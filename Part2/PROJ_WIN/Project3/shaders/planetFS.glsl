#version 330

in vec3 WorldCoord;
in vec3 ToCam;
in vec3 Up;
in vec3 Right;
in vec2 TexCoord;

in vec3 fs_color;

out vec4 FragColor;

void main()
{
	//float dist = length(WorldCoord);
	//vec2 coord = 2.01 * (TexCoord - vec2(0.5));
 //   float r = length(coord);
 //   if (r >= 1.0) { discard; }

	//vec3 N = Right*-coord.x + Up*coord.y + ToCam*sqrt(1-r*r);
 //   vec3 L = normalize(-WorldCoord);
 //   float light = 0.1 + 0.9*clamp(dot(N,L),0.0, 1.0)*exp(-dist);
 //   vec3 color = vec3(1.0f);
 //   FragColor = vec4(color*light,1.0);

	//FragColor = vec4(1.0);

	FragColor.x = abs(fs_color.x);
	FragColor.y = abs(fs_color.y);
	FragColor.z = abs(fs_color.z);
	FragColor.w = 1.0f;
} 
