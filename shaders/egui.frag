#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D font_texture;

void main() {
    vec4 tex_color = texture(font_texture, frag_uv);
    // Multiply vertex color by texture (font atlas)
    // egui uses premultiplied alpha
    out_color = frag_color * tex_color;
}
