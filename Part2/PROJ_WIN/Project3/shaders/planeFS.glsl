varying vec4 v_Normal;

void main(void)
{
	vec4 light = vec4(1.0, 1.0, 1.0, 0.0);
	gl_FragColor = vec4(0.25,0.65,0.85,1.0) * clamp(abs(dot(v_Normal, light)), 0.0, 1.0);
}