#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject 
{
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPos;

void main() 
{
    //ubo.proj * ubo.view * ubo.model 
    gl_Position = vec4(inPos, 1.0);
}
