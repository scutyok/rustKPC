#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D font_texture;

void main() {
    // DEBUG: Ignore font texture, just show the color (should show white text if mesh is correct)
    out_color = frag_color;
}
