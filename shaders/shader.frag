#version 450

// Input from vertex shader
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

// Uniforms
layout(binding = 1) uniform sampler2D texSampler;

// Push constants for opacity
layout(push_constant) uniform PushConstants {
    layout(offset = 64) float opacity;
} push;

// Output
layout(location = 0) out vec4 outColor;

void main() {
    // Sample texture directly
    vec4 texColor = texture(texSampler, fragTexCoord);
    outColor = vec4(texColor.rgb, 1.0);
}
