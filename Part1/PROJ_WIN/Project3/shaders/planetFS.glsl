//void main(void)
//{
//	gl_FragColor = vec4(1.0);
//}

#version 330

in vec3 WorldCoord;
in vec3 ToCam;
in vec3 Up;
in vec3 Right;
in vec2 TexCoord;
out vec4 FragColor;

void main()
{
	
	vec2 coord = 2.01 * (TexCoord - vec2(0.5));
    float r = length(coord);
	if (r >= 1.0) { discard; }
    float dist = length(WorldCoord);	
	if(dist <= 0.01)
    {
      if (r >= 0.8 && r <=1.0) {
        FragColor = vec4(0.25);
      } else if (r >= 0.6 && r <= 0.8) {
        FragColor = vec4(0.35);
      } else if (r >= 0.4 && r <= 0.6) {
        FragColor = vec4(0.5);
      } else if (r >=0.2 && r <= 0.4) {
        FragColor = vec4(0.75);
      } else
        FragColor = vec4(1.0);

      return;
    }
    
	vec3 N = Right*-coord.x + Up*coord.y + ToCam*sqrt(1-r*r);
    vec3 L = normalize(-WorldCoord);
    float light = 0.1 + 0.9*clamp(dot(N,L),0.0, 1.0)*exp(-dist);
    //vec3 color = vec3(0.4, 0.1, 0.6);
    // Unique color display
	vec3 color = abs(normalize(WorldCoord));
    FragColor = vec4(color*light,1.0);
}