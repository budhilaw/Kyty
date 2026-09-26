#include "graphics/host_gpu/renderer/frameCapture.h"
#include "kernel/memory.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <unordered_map>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace Libs::Graphics::FrameCapture {

namespace {

struct CapturedImage {
	uint64_t    key     = 0; // generation << 32 | index
	uint64_t    address = 0;
	std::string role;
};

// An image copy recorded into the GPU stream; converted after the frame's last flush.
struct PendingDownload {
	CapturedImage           image;
	std::unique_ptr<Buffer> buffer;
	vk::Format              format;
	uint32_t                width;
	uint32_t                height;
};

struct State {
	RenderContext*           context = nullptr;
	std::set<uint64_t>       frames;
	std::string              dir     = "_FrameDump";
	uint64_t                 flips   = 0;
	bool                     active  = false;
	uint64_t                 frame   = 0;
	uint32_t                 index   = 0;
	std::string              pending;
	std::string              text;
	std::vector<CapturedImage> images;
	bool                     configured = false;
	// KYTY_FRAME_DUMP_TRIGGER=path: when that file appears, the next frame is captured and the
	// file is deleted (captures a screen without knowing its frame number).
	std::string trigger;
	// KYTY_FRAME_DUMP_TRACK=addr,...: after every dispatch that writes one of these images, a
	// copy is taken at that point of the frame (shows which pass produced a bad value).
	std::vector<uint64_t>                        track;
	std::vector<std::pair<uint64_t, uint64_t>>   dispatch_storage; // key, address
	std::vector<PendingDownload>                 snapshots;
};

State& S() {
	static State state;
	if (!state.configured) {
		state.configured = true;
		if (const char* list = std::getenv("KYTY_FRAME_DUMP"); list != nullptr) {
			const char* cursor = list;
			while (*cursor != '\0') {
				char*      end   = nullptr;
				const auto value = std::strtoull(cursor, &end, 10);
				if (end == cursor) {
					break;
				}
				state.frames.insert(value);
				cursor = *end == ',' ? end + 1 : end;
			}
		}
		if (const char* dir = std::getenv("KYTY_FRAME_DUMP_DIR"); dir != nullptr) {
			state.dir = dir;
		}
		if (const char* trigger = std::getenv("KYTY_FRAME_DUMP_TRIGGER"); trigger != nullptr) {
			state.trigger = trigger;
		}
		if (const char* list = std::getenv("KYTY_FRAME_DUMP_TRACK"); list != nullptr) {
			const char* cursor = list;
			while (*cursor != '\0') {
				char*      end   = nullptr;
				const auto value = std::strtoull(cursor, &end, 16);
				if (end == cursor) {
					break;
				}
				state.track.push_back(value);
				cursor = *end == ',' ? end + 1 : end;
			}
		}
	}
	return state;
}

void AddImage(uint64_t key, uint64_t address, const std::string& role) {
	auto& s = S();
	for (const auto& image: s.images) {
		if (image.key == key) {
			return;
		}
	}
	s.images.push_back({key, address, role});
}

// Minimal PNG writer: 8-bit RGBA, zlib "stored" blocks (no compression).
uint32_t Crc(const uint8_t* data, size_t size, uint32_t crc = 0xffffffffu) {
	static uint32_t table[256] = {};
	static bool     init       = false;
	if (!init) {
		for (uint32_t n = 0; n < 256; n++) {
			uint32_t c = n;
			for (int k = 0; k < 8; k++) {
				c = (c & 1u) != 0 ? 0xedb88320u ^ (c >> 1u) : c >> 1u;
			}
			table[n] = c;
		}
		init = true;
	}
	for (size_t i = 0; i < size; i++) {
		crc = table[(crc ^ data[i]) & 0xffu] ^ (crc >> 8u);
	}
	return crc;
}

void PutBe32(std::vector<uint8_t>& out, uint32_t value) {
	out.push_back(static_cast<uint8_t>(value >> 24u));
	out.push_back(static_cast<uint8_t>(value >> 16u));
	out.push_back(static_cast<uint8_t>(value >> 8u));
	out.push_back(static_cast<uint8_t>(value));
}

void Chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
	PutBe32(out, static_cast<uint32_t>(data.size()));
	const auto start = out.size();
	out.insert(out.end(), type, type + 4);
	out.insert(out.end(), data.begin(), data.end());
	PutBe32(out, Crc(out.data() + start, out.size() - start) ^ 0xffffffffu);
}

bool WritePng(const std::filesystem::path& path, uint32_t width, uint32_t height,
              const std::vector<uint8_t>& rgba) {
	std::vector<uint8_t> raw;
	raw.reserve(static_cast<size_t>(height) * (width * 4u + 1u));
	for (uint32_t y = 0; y < height; y++) {
		raw.push_back(0);
		raw.insert(raw.end(), rgba.begin() + static_cast<ptrdiff_t>(y) * width * 4,
		           rgba.begin() + static_cast<ptrdiff_t>(y + 1u) * width * 4);
	}
	std::vector<uint8_t> z {0x78, 0x01};
	uint32_t             a = 1;
	uint32_t             b = 0;
	for (size_t offset = 0; offset < raw.size();) {
		const auto block = std::min<size_t>(65535, raw.size() - offset);
		z.push_back(offset + block == raw.size() ? 1 : 0);
		z.push_back(static_cast<uint8_t>(block));
		z.push_back(static_cast<uint8_t>(block >> 8u));
		z.push_back(static_cast<uint8_t>(~block));
		z.push_back(static_cast<uint8_t>((~block) >> 8u));
		for (size_t i = 0; i < block; i++) {
			a = (a + raw[offset + i]) % 65521u;
			b = (b + a) % 65521u;
		}
		z.insert(z.end(), raw.begin() + static_cast<ptrdiff_t>(offset),
		         raw.begin() + static_cast<ptrdiff_t>(offset + block));
		offset += block;
	}
	PutBe32(z, (b << 16u) | a);
	std::vector<uint8_t> png {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
	std::vector<uint8_t> ihdr;
	PutBe32(ihdr, width);
	PutBe32(ihdr, height);
	ihdr.insert(ihdr.end(), {8, 6, 0, 0, 0});
	Chunk(png, "IHDR", ihdr);
	Chunk(png, "IDAT", z);
	Chunk(png, "IEND", {});
	auto* file = std::fopen(path.string().c_str(), "wb");
	if (file == nullptr) {
		return false;
	}
	std::fwrite(png.data(), 1, png.size(), file);
	std::fclose(file);
	return true;
}

float Half(uint16_t h) {
	const uint32_t sign = (h >> 15u) & 1u;
	const uint32_t exp  = (h >> 10u) & 0x1fu;
	const uint32_t man  = h & 0x3ffu;
	float          value;
	if (exp == 0) {
		value = std::ldexp(static_cast<float>(man), -24);
	} else if (exp == 31) {
		value = man != 0 ? 0.0f : 65504.0f;
	} else {
		value = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
	}
	return sign != 0 ? -value : value;
}

float SmallFloat(uint32_t bits, uint32_t mantissa_bits) {
	const uint32_t exp = bits >> mantissa_bits;
	const uint32_t man = bits & ((1u << mantissa_bits) - 1u);
	if (exp == 0) {
		return std::ldexp(static_cast<float>(man), -14 - static_cast<int>(mantissa_bits));
	}
	if (exp == 31) {
		return 65504.0f;
	}
	return std::ldexp(static_cast<float>(man | (1u << mantissa_bits)),
	                  static_cast<int>(exp) - 15 - static_cast<int>(mantissa_bits));
}

uint8_t Unit(float value) {
	return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// Bytes per texel for the formats the capture can convert; 0 means unsupported.
uint32_t TexelBytes(vk::Format format) {
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8Uint: return 1;
		case vk::Format::eR16Uint:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16Unorm: return 2;
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eB10G11R11UfloatPack32:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32Uint: return 4;
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16G16B16A16Uint:
		case vk::Format::eR32G32Sfloat: return 8;
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eR32G32B32A32Uint: return 16;
		default: return 0;
	}
}

// Largest color channel of a float texel before any clamping; false for non-float formats.
bool RawColorMax(vk::Format format, const uint8_t* src, float& out) {
	uint16_t h[4] {};
	float    f[4] {};
	uint32_t u32 = 0;
	switch (format) {
		case vk::Format::eR16Sfloat: std::memcpy(h, src, 2); out = Half(h[0]); return true;
		case vk::Format::eR16G16Sfloat:
			std::memcpy(h, src, 4);
			out = std::max(Half(h[0]), Half(h[1]));
			return true;
		case vk::Format::eR16G16B16A16Sfloat:
			std::memcpy(h, src, 8);
			out = std::max({Half(h[0]), Half(h[1]), Half(h[2])});
			return true;
		case vk::Format::eR32Sfloat: std::memcpy(f, src, 4); out = f[0]; return true;
		case vk::Format::eR32G32Sfloat: std::memcpy(f, src, 8); out = std::max(f[0], f[1]); return true;
		case vk::Format::eR32G32B32A32Sfloat:
			std::memcpy(f, src, 16);
			out = std::max({f[0], f[1], f[2]});
			return true;
		case vk::Format::eB10G11R11UfloatPack32:
			std::memcpy(&u32, src, 4);
			out = std::max({SmallFloat(u32 & 0x7ffu, 6), SmallFloat((u32 >> 11u) & 0x7ffu, 6),
			                SmallFloat((u32 >> 22u) & 0x3ffu, 5)});
			return true;
		default: return false;
	}
}

void ToRgba(vk::Format format, const uint8_t* src, uint8_t* dst) {
	uint32_t u32 = 0;
	std::memcpy(&u32, src, std::min<uint32_t>(4, TexelBytes(format)));
	uint16_t h[4] {};
	float    f[4] {};
	switch (format) {
		case vk::Format::eR8Unorm: dst[0] = dst[1] = dst[2] = src[0]; dst[3] = 255; return;
		case vk::Format::eR8G8Unorm: dst[0] = src[0]; dst[1] = src[1]; dst[2] = 0; dst[3] = 255; return;
		case vk::Format::eR16Sfloat:
			std::memcpy(h, src, 2);
			dst[0] = dst[1] = dst[2] = Unit(Half(h[0]));
			dst[3] = 255;
			return;
		case vk::Format::eR16Unorm:
			std::memcpy(h, src, 2);
			dst[0] = dst[1] = dst[2] = static_cast<uint8_t>(h[0] >> 8u);
			dst[3] = 255;
			return;
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb: std::memcpy(dst, src, 4); return;
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
			dst[0] = src[2]; dst[1] = src[1]; dst[2] = src[0]; dst[3] = src[3];
			return;
		case vk::Format::eA2B10G10R10UnormPack32:
			dst[0] = static_cast<uint8_t>((u32 & 0x3ffu) >> 2u);
			dst[1] = static_cast<uint8_t>(((u32 >> 10u) & 0x3ffu) >> 2u);
			dst[2] = static_cast<uint8_t>(((u32 >> 20u) & 0x3ffu) >> 2u);
			dst[3] = static_cast<uint8_t>((u32 >> 30u) * 85u);
			return;
		case vk::Format::eA2R10G10B10UnormPack32:
			dst[2] = static_cast<uint8_t>((u32 & 0x3ffu) >> 2u);
			dst[1] = static_cast<uint8_t>(((u32 >> 10u) & 0x3ffu) >> 2u);
			dst[0] = static_cast<uint8_t>(((u32 >> 20u) & 0x3ffu) >> 2u);
			dst[3] = static_cast<uint8_t>((u32 >> 30u) * 85u);
			return;
		case vk::Format::eB10G11R11UfloatPack32:
			dst[0] = Unit(SmallFloat(u32 & 0x7ffu, 6));
			dst[1] = Unit(SmallFloat((u32 >> 11u) & 0x7ffu, 6));
			dst[2] = Unit(SmallFloat((u32 >> 22u) & 0x3ffu, 5));
			dst[3] = 255;
			return;
		case vk::Format::eR16G16Sfloat:
			std::memcpy(h, src, 4);
			dst[0] = Unit(Half(h[0])); dst[1] = Unit(Half(h[1])); dst[2] = 0; dst[3] = 255;
			return;
		case vk::Format::eR32Sfloat:
			std::memcpy(f, src, 4);
			dst[0] = dst[1] = dst[2] = Unit(f[0]);
			dst[3] = 255;
			return;
		case vk::Format::eR32Uint:
			dst[0] = static_cast<uint8_t>(u32); dst[1] = static_cast<uint8_t>(u32 >> 8u);
			dst[2] = static_cast<uint8_t>(u32 >> 16u); dst[3] = 255;
			return;
		case vk::Format::eR16G16B16A16Sfloat:
			std::memcpy(h, src, 8);
			for (int i = 0; i < 4; i++) {
				dst[i] = Unit(Half(h[i]));
			}
			return;
		case vk::Format::eR16G16B16A16Unorm:
			std::memcpy(h, src, 8);
			for (int i = 0; i < 4; i++) {
				dst[i] = static_cast<uint8_t>(h[i] >> 8u);
			}
			return;
		case vk::Format::eR32G32B32A32Sfloat:
			std::memcpy(f, src, 16);
			for (int i = 0; i < 4; i++) {
				dst[i] = Unit(f[i]);
			}
			return;
		// Integer formats show their low byte, which is where small ids and flags live.
		case vk::Format::eR8Uint: dst[0] = dst[1] = dst[2] = src[0]; dst[3] = 255; return;
		case vk::Format::eR16Uint:
			std::memcpy(h, src, 2);
			dst[0] = static_cast<uint8_t>(h[0]); dst[1] = static_cast<uint8_t>(h[0] >> 8u);
			dst[2] = 0; dst[3] = 255;
			return;
		case vk::Format::eR16G16B16A16Uint:
			std::memcpy(h, src, 8);
			for (int i = 0; i < 4; i++) {
				dst[i] = static_cast<uint8_t>(h[i]);
			}
			return;
		case vk::Format::eR32G32Sfloat:
			std::memcpy(f, src, 8);
			dst[0] = Unit(f[0]); dst[1] = Unit(f[1]); dst[2] = 0; dst[3] = 255;
			return;
		case vk::Format::eR32G32B32A32Uint:
			for (int i = 0; i < 4; i++) {
				dst[i] = src[i * 4];
			}
			return;
		default: std::memset(dst, 0, 4); return;
	}
}

// Records a copy of the image into the GPU stream at the current point; false with a reason
// written to the frame text when the image cannot be converted.
bool StartDownload(State& s, const CapturedImage& captured, std::vector<PendingDownload>& out) {
	auto& cache     = s.context->GetTextureCache();
	auto& scheduler = s.context->GetCommandScheduler();
	auto& graphics  = s.context->GetGraphics();
	const ImageId id(static_cast<uint32_t>(captured.key), static_cast<uint32_t>(captured.key >> 32u));
	auto* image = cache.TryGetImage(id);
	// 32-bit float depth is read through its depth aspect and reported as R32 float.
	const bool depth32 = image != nullptr && image->info.IsDepth() &&
	                     (image->info.pixel_format == vk::Format::eD32Sfloat ||
	                      image->info.pixel_format == vk::Format::eD32SfloatS8Uint);
	if (image == nullptr || image->backing.image == nullptr ||
	    (image->info.IsDepth() && !depth32) || image->info.samples > 1) {
		s.text += fmt::format("IMAGE {} addr=0x{:010x} skipped (gone, depth or msaa)\n",
		                      captured.role, captured.address);
		return false;
	}
	const auto format = depth32 ? vk::Format::eR32Sfloat : image->info.pixel_format;
	const auto texel  = TexelBytes(format);
	const auto width  = image->info.extent.width;
	const auto height = image->info.extent.height;
	if (texel == 0) {
		s.text += fmt::format("IMAGE {} addr=0x{:010x} format={} not convertible\n", captured.role,
		                      captured.address, vk::to_string(format));
		return false;
	}
	const uint64_t size   = static_cast<uint64_t>(width) * height * texel;
	auto           buffer = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
	                                                 vk::BufferUsageFlagBits::eTransferDst, size);
	vk::BufferImageCopy copy {};
	copy.imageSubresource.aspectMask =
	    depth32 ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;
	copy.imageSubresource.layerCount = 1;
	copy.imageExtent                 = vk::Extent3D {width, height, 1};
	image->Download(std::span(&copy, 1), buffer->Handle(), 0, size);
	out.push_back({captured, std::move(buffer), format, width, height});
	// The stencil plane of a depth/stencil image is saved too, as 8-bit unsigned.
	if (depth32 && image->info.pixel_format == vk::Format::eD32SfloatS8Uint) {
		const uint64_t stencil_size = static_cast<uint64_t>(width) * height;
		auto stencil_buffer = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
		                                               vk::BufferUsageFlagBits::eTransferDst,
		                                               stencil_size);
		vk::BufferImageCopy stencil_copy {};
		stencil_copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eStencil;
		stencil_copy.imageSubresource.layerCount = 1;
		stencil_copy.imageExtent                 = vk::Extent3D {width, height, 1};
		image->Download(std::span(&stencil_copy, 1), stencil_buffer->Handle(), 0, stencil_size);
		out.push_back({{captured.key, image->info.stencil.address,
		                captured.role == "depth" ? std::string("stencil") : captured.role + "_stencil"},
		               std::move(stencil_buffer), vk::Format::eR8Uint, width, height});
	}
	return true;
}

void Finish(uint64_t presented_address) {
	auto& s   = S();
	auto  dir = std::filesystem::path(s.dir) / ("frame_" + std::to_string(s.frame));
	std::error_code error;
	std::filesystem::create_directories(dir, error);
	s.text += fmt::format("\nPRESENTED address=0x{:010x}\n", presented_address);

	// KYTY_FRAME_DUMP_NOIMG=1 writes only draws.txt, fast enough for runs of consecutive frames.
	static const bool no_images = std::getenv("KYTY_FRAME_DUMP_NOIMG") != nullptr;
	if (s.context != nullptr && !no_images) {
		auto&                        scheduler = s.context->GetCommandScheduler();
		std::vector<PendingDownload> pending   = std::move(s.snapshots);
		s.snapshots.clear();
		for (const auto& captured: s.images) {
			(void)StartDownload(s, captured, pending);
		}
		scheduler.FlushAndWait();
		for (auto& item: pending) {
			const auto  texel = TexelBytes(item.format);
			const auto* data  = item.buffer->Mapped().data();
			std::vector<uint8_t> rgba(static_cast<size_t>(item.width) * item.height * 4u);
			std::string float_stats;
			{
				double   sum     = 0.0;
				float    max     = 0.0f;
				uint64_t bad     = 0;
				uint64_t samples = 0;
				float    value   = 0.0f;
				for (size_t i = 0; i < static_cast<size_t>(item.width) * item.height; i++) {
					if (!RawColorMax(item.format, data + i * texel, value)) {
						break;
					}
					if (!std::isfinite(value)) {
						bad++;
						continue;
					}
					sum += value;
					max = std::max(max, value);
					samples++;
				}
				if (samples != 0 || bad != 0) {
					float_stats = fmt::format(" float_max={:.4g} float_mean={:.4g} nan_inf={}", max,
					                          samples != 0 ? sum / static_cast<double>(samples) : 0.0,
					                          bad);
				}
			}
			for (size_t i = 0; i < static_cast<size_t>(item.width) * item.height; i++) {
				ToRgba(item.format, data + i * texel, rgba.data() + i * 4u);
			}
			const auto name = fmt::format("{}_{:010x}_{}.png", item.image.role, item.image.address,
			                              vk::to_string(item.format));
			// The PNG is written opaque so transparent layers stay visible; the alpha range is logged.
			uint32_t alpha_min = 255;
			uint32_t alpha_max = 0;
			uint32_t rgb_max   = 0;
			uint64_t lit       = 0;
			for (size_t i = 0; i < rgba.size(); i += 4) {
				alpha_min = std::min<uint32_t>(alpha_min, rgba[i + 3]);
				alpha_max = std::max<uint32_t>(alpha_max, rgba[i + 3]);
				const uint32_t m = std::max({rgba[i], rgba[i + 1], rgba[i + 2]});
				rgb_max          = std::max(rgb_max, m);
				lit += (m != 0 ? 1 : 0);
			}
			if (alpha_min != alpha_max) {
				std::vector<uint8_t> alpha(rgba.size());
				for (size_t i = 0; i < rgba.size(); i += 4) {
					alpha[i] = alpha[i + 1] = alpha[i + 2] = rgba[i + 3];
					alpha[i + 3]                          = 255;
				}
				WritePng(dir / (name.substr(0, name.size() - 4) + "_alpha.png"), item.width,
				         item.height, alpha);
			}
			for (size_t i = 3; i < rgba.size(); i += 4) {
				rgba[i] = 255;
			}
			WritePng(dir / name, item.width, item.height, rgba);
			s.text += fmt::format(
			    "IMAGE {} addr=0x{:010x} {}x{} {} -> {} alpha={}..{} rgb_max={} lit={:.1f}%{}{}\n",
			    item.image.role, item.image.address, item.width, item.height,
			    vk::to_string(item.format), name, alpha_min, alpha_max, rgb_max,
			    100.0 * static_cast<double>(lit) / static_cast<double>(rgba.size() / 4), float_stats,
			    item.image.address == presented_address ? "  [PRESENTED]" : "");
		}
	}
	if (auto* file = std::fopen((dir / "draws.txt").string().c_str(), "wb"); file != nullptr) {
		std::fwrite(s.text.data(), 1, s.text.size(), file);
		std::fclose(file);
	}
	LOGF("FRAMECAPTURE: frame %" PRIu64 " written to %s\n", s.frame, dir.string().c_str());
}

} // namespace

void SetContext(RenderContext* context) {
	S().context = context;
}

bool Active() {
	return S().active;
}

std::atomic<uint64_t> g_flip_serial {0};

uint64_t FlipSerial() {
	return g_flip_serial.load(std::memory_order_acquire);
}

std::atomic<uint32_t> g_frame_draws {0};
std::atomic<uint32_t> g_frame_passes {0};

void NotePassEnd() {
	g_frame_passes.fetch_add(1, std::memory_order_relaxed);
}

void NoteDraw() {
	g_frame_draws.fetch_add(1, std::memory_order_relaxed);
}

void OnFlip(uint64_t presented_address) {
	const auto flip = g_flip_serial.fetch_add(1, std::memory_order_acq_rel) + 1;
	{
		// Draws per frame, logged every 30 flips: the Uncharted selector draws ~700 when the scene
		// renders and ~22 when the game culled it (the blob), so a single run is classifiable.
		static uint32_t min_draws = UINT32_MAX;
		static uint32_t max_draws = 0;
		const auto      draws     = g_frame_draws.exchange(0, std::memory_order_relaxed);
		static uint32_t max_passes = 0;
		max_passes = std::max(max_passes, g_frame_passes.exchange(0, std::memory_order_relaxed));
		min_draws                 = std::min(min_draws, draws);
		max_draws                 = std::max(max_draws, draws);
		if (flip % 30u == 0u) {
			std::printf("FRAMEDRAWS flip=%llu min=%u max=%u passes=%u\n", static_cast<unsigned long long>(flip),
			            min_draws, max_draws, max_passes);
			max_passes = 0;
			min_draws = UINT32_MAX;
			max_draws = 0;
		}
	}
	Libs::LibKernel::Memory::NoteGpuFrame();
	auto& s = S();
	if (s.frames.empty() && s.trigger.empty()) {
		return;
	}
	s.flips++;
	if (!s.trigger.empty() && !s.active && (s.flips % 8) == 0) {
		std::error_code error;
		if (std::filesystem::exists(s.trigger, error)) {
			std::filesystem::remove(s.trigger, error);
			s.frames.insert(s.flips + 1);
		}
	}
	if (s.active) {
		Finish(presented_address);
		s.active = false;
		s.images.clear();
		s.text.clear();
		s.pending.clear();
	}
	if (s.frames.contains(s.flips)) {
		s.active = true;
		s.frame  = s.flips;
		s.index  = 0;
		s.text   = fmt::format("FRAME {} (draws and dispatches in submission order)\n", s.frame);
	}
}

void NoteTarget(const char* kind, uint32_t slot, uint32_t image_index, uint64_t address,
                uint32_t format, uint32_t width, uint32_t height) {
	auto& s = S();
	if (!s.active) {
		return;
	}
	(void)image_index;
	s.pending += fmt::format("    {}[{}] addr=0x{:010x} {}x{} fmt={}\n", kind, slot, address, width,
	                         height, vk::to_string(static_cast<vk::Format>(format)));
}

void NoteTexture(uint64_t address, uint32_t image_index, uint32_t format, uint32_t width,
                 uint32_t height, bool storage, const char* note) {
	auto& s = S();
	if (!s.active) {
		return;
	}
	(void)image_index;
	s.pending += fmt::format("    {} addr=0x{:010x} {}x{} fmt={}{}{}\n", storage ? "storage" : "texture",
	                         address, width, height, vk::to_string(static_cast<vk::Format>(format)),
	                         note != nullptr ? " " : "", note != nullptr ? note : "");
}

void NoteBuffer(const char* kind, uint64_t address, uint64_t size, bool written) {
	auto& s = S();
	if (!s.active) {
		return;
	}
	s.pending += fmt::format("    {} {} 0x{:010x}..0x{:010x} ({} bytes)\n", kind,
	                         written ? "WRITE" : "read", address, address + size, size);
	// KYTY_FRAME_DUMP_COMPARE=1: a read buffer's GPU copy against guest memory (first 4 KiB);
	// a difference means the draw sees data the CPU has since replaced (or the reverse).
	static const bool compare = std::getenv("KYTY_FRAME_DUMP_COMPARE") != nullptr;
	if (compare && !written && size >= 16) {
		const auto            bytes = std::min<uint64_t>(size & ~uint64_t {3}, 4096);
		std::vector<uint8_t> guest(bytes);
		std::vector<uint8_t> gpu(bytes);
		if (Libs::LibKernel::Memory::TryReadBacking(address, guest.data(), bytes) &&
		    Libs::LibKernel::Memory::PeekGpuCopy(address, gpu.data(), bytes)) {
			uint64_t diff = 0;
			uint64_t first = UINT64_MAX;
			for (uint64_t i = 0; i < bytes; i++) {
				if (guest[i] != gpu[i]) {
					diff++;
					first = std::min(first, i);
				}
			}
			if (diff != 0) {
				uint32_t g = 0;
				uint32_t v = 0;
				std::memcpy(&g, guest.data() + (first & ~uint64_t {3}), 4);
				std::memcpy(&v, gpu.data() + (first & ~uint64_t {3}), 4);
				s.pending += fmt::format("      MISMATCH {} of {} bytes, first +0x{:x} guest={:08x} "
				                         "gpu={:08x}\n",
				                         diff, bytes, first, g, v);
			}
		}
	}
	// Small constant blocks (exposure, focus, counters) are printed as words, as the GPU last
	// left them: a capture drains the GPU before its draw list is written.
	if (size <= 64 && size >= 4) {
		uint32_t words[16] {};
		if (Libs::LibKernel::Memory::ReadGpuBackingOrDownload(address, words, size & ~uint64_t {3})) {
			s.pending += "      words:";
			for (uint64_t i = 0; i < size / 4; i++) {
				s.pending += fmt::format(" {:08x}({:g})", words[i], std::bit_cast<float>(words[i]));
			}
			s.pending += "\n";
		}
		return;
	}
	// KYTY_FRAME_DUMP_CS=hash: the larger buffers of that compute shader get a word histogram.
	static const uint64_t histogram_cs = [] {
		const char* value = std::getenv("KYTY_FRAME_DUMP_CS");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	}();
	if (histogram_cs != 0 && CurrentWriterShader() == histogram_cs && size >= 4) {
		const auto            bytes = std::min<uint64_t>(size & ~uint64_t {3}, 1u << 20u);
		std::vector<uint32_t> words(bytes / 4);
		if (Libs::LibKernel::Memory::ReadGpuBackingOrDownload(address, words.data(), bytes)) {
			std::unordered_map<uint32_t, uint32_t> counts;
			uint32_t                               max_value = 0;
			for (const auto word: words) {
				counts[word]++;
				max_value = std::max(max_value, word);
			}
			std::vector<std::pair<uint32_t, uint32_t>> top(counts.begin(), counts.end());
			std::sort(top.begin(), top.end(),
			          [](const auto& a, const auto& b) { return a.second > b.second; });
			s.pending += fmt::format("      histogram: words={} distinct={} max={:#x} zeros={:.1f}%:",
			                         words.size(), counts.size(), max_value,
			                         100.0 * counts[0] / static_cast<double>(words.size()));
			for (size_t i = 0; i < std::min<size_t>(top.size(), 8); i++) {
				s.pending += fmt::format(" {:#x}x{}", top[i].first, top[i].second);
			}
			s.pending += fmt::format("\n      first words:");
			for (size_t i = 0; i < std::min<size_t>(words.size(), 16); i++) {
				s.pending += fmt::format(" {:08x}", words[i]);
			}
			s.pending += "\n";
		}
	}
}

void NoteMarker(const std::string& text) {
	auto& s = S();
	if (!s.active) {
		return;
	}
	s.text += "  -- " + text + "\n";
}

void NoteDraw(const char* kind, uint64_t vs_hash, uint64_t ps_hash, const std::string& detail) {
	auto& s = S();
	if (!s.active) {
		return;
	}
	s.text += fmt::format("#{} {} vs={:016x} ps={:016x} {}\n{}", s.index++, kind, vs_hash, ps_hash,
	                      detail, s.pending);
	s.pending.clear();
	// Targets written by draws stay listed; they are snapshotted at the next dispatch, where no
	// render pass is open.
}

void NoteDispatch(uint64_t cs_hash, uint32_t x, uint32_t y, uint32_t z, bool indirect) {
	auto& s = S();
	if (!s.active) {
		return;
	}
	const auto index = s.index;
	s.text += fmt::format("#{} dispatch cs={:016x} groups={}x{}x{}{}\n{}", s.index++, cs_hash, x, y,
	                      z, indirect ? " indirect" : "", s.pending);
	s.pending.clear();
	static const bool no_images = std::getenv("KYTY_FRAME_DUMP_NOIMG") != nullptr;
	if (s.context != nullptr && !no_images) {
		for (const auto& [key, address]: s.dispatch_storage) {
			if (std::ranges::find(s.track, address) != s.track.end()) {
				(void)StartDownload(s, {key, address, fmt::format("snap{:04}_{:016x}", index, cs_hash)},
				                    s.snapshots);
			}
		}
	}
	s.dispatch_storage.clear();
}

// Images are recorded from these helpers so the frame's targets can be dumped at the flip.
void NoteImageForDump(uint64_t key, uint64_t address, const char* role) {
	auto& s = S();
	if (s.active) {
		AddImage(key, address, role);
		const std::string_view kind(role);
		if (!s.track.empty() && (kind == "storage" || kind == "rt" || kind == "depth") &&
		    std::ranges::find(s.dispatch_storage, std::pair {key, address}) ==
		        s.dispatch_storage.end()) {
			s.dispatch_storage.emplace_back(key, address);
		}
	}
}

} // namespace Libs::Graphics::FrameCapture
