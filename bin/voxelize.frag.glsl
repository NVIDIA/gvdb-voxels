#version 440
#extension GL_ARB_shader_image_size : enable
#extension GL_ARB_shader_image_load_store : enable

layout (binding = 0, r32f ) coherent uniform writeonly image3D volumeTexture;
layout (location = 0) out vec4 FragColor;

in vec3 gs_WorldNormal;
in vec3 gs_Color;
in mat3 gs_SwizzleMatrixInv;

uniform vec3 uTexRes;
uniform int  uSamples;

void main()
{
    vec3 coord = (gs_SwizzleMatrixInv * vec3(gl_FragCoord.xy, uSamples*gl_FragCoord.z )) * uTexRes / uSamples;
	ivec3 ic = ivec3(coord);	

	imageStore( volumeTexture, ic, vec4(1,1,1,1) );	
	imageStore( volumeTexture, ic-ivec3(1,0,0), vec4(1,1,1,1) );
	imageStore( volumeTexture, ic-ivec3(0,1,0), vec4(1,1,1,1) );
	imageStore( volumeTexture, ic-ivec3(0,0,1), vec4(1,1,1,1) );
	imageStore( volumeTexture, ic+ivec3(1,0,0), vec4(1,1,1,1) );
	imageStore( volumeTexture, ic+ivec3(0,1,0), vec4(1,1,1,1) );
	imageStore( volumeTexture, ic+ivec3(0,0,1), vec4(1,1,1,1) );
}