#pragma once

#include <cstdint>
#include <string>

// Frame capture for render debugging, driven by environment variables so it runs unattended.
//
//   KYTY_FRAME_DUMP=1650,1700   flip numbers (frame counter) to capture
//   KYTY_FRAME_DUMP_DIR=path    output folder (default _FrameDump)
//   KYTY_FRAME_DUMP_NOIMG=1     skip the PNGs (draw lists only)
//
// For each captured frame, <dir>/frame_<n>/draws.txt lists every draw and dispatch with its
// shaders, render targets and bound textures (null-bound ones marked), and every render target
// written during the frame plus the presented image is saved as a PNG.
namespace Libs::Graphics {

class RenderContext;

namespace FrameCapture {

[[nodiscard]] bool Active();

// Called by the GPU thread for every flip; starts and finishes captures.
void OnFlip(uint64_t presented_address);
// Counts one recorded draw (per-frame totals are logged at flips).
void NoteDraw();

// Records produced while a frame is captured; they attach to the next draw or dispatch.
void NoteTarget(const char* kind, uint32_t slot, uint32_t image_index, uint64_t address,
                uint32_t format, uint32_t width, uint32_t height);
void NoteTexture(uint64_t address, uint32_t image_index, uint32_t format, uint32_t width,
                 uint32_t height, bool storage, const char* note);
// A buffer range the next draw or dispatch reads or writes, or a copy/fill done by the command
// processor (kind names it).
void NoteBuffer(const char* kind, uint64_t address, uint64_t size, bool written);
// A command-stream event (buffer entered or left, debug marker) written in order.
void NoteMarker(const std::string& text);
void NoteDraw(const char* kind, uint64_t vs_hash, uint64_t ps_hash, const std::string& detail);
void NoteDispatch(uint64_t cs_hash, uint32_t x, uint32_t y, uint32_t z, bool indirect);

void SetContext(RenderContext* context);

// Counts every flip (captures or not); lets per-frame debug switches act once per frame.
[[nodiscard]] uint64_t FlipSerial();

// Registers an image (key = generation << 32 | index) to be saved as PNG at the frame's end.
void NoteImageForDump(uint64_t key, uint64_t address, const char* role);

} // namespace FrameCapture
} // namespace Libs::Graphics
