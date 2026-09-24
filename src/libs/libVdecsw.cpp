#include "common/abi.h"
#include "common/logging/log.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/videoDec2Decoder.h"
#include "kernel/memory.h"
#include "loader/symbolDatabase.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs {

LIB_VERSION("Vdecsw", 1, "Vdecsw", 1, 1);

namespace Vdecsw {

constexpr int32_t VDECSW_ERROR_API_FAIL = static_cast<int32_t>(0x81510100u);
constexpr int32_t VDECSW_ERROR_STRUCT_SIZE = static_cast<int32_t>(0x81510101u);
constexpr int32_t VDECSW_ERROR_ARGUMENT_POINTER = static_cast<int32_t>(0x81510102u);
constexpr int32_t VDECSW_ERROR_DECODER_INSTANCE = static_cast<int32_t>(0x81510103u);
constexpr int32_t VDECSW_ERROR_OUTPUT_PENDING = static_cast<int32_t>(0x81510115u);
constexpr int32_t VDECSW_ERROR_INPUT_QUEUE_EMPTY = static_cast<int32_t>(0x81510116u);
constexpr int32_t VDECSW_ERROR_DECODE_PENDING = static_cast<int32_t>(0x81510117u);
constexpr int32_t VDECSW_ERROR_OUTPUT_INFO = static_cast<int32_t>(0x8151010fu);
constexpr int32_t VDECSW_ERROR_CODEC_TYPE = static_cast<int32_t>(0x81510204u);

constexpr size_t VDECSW_COMPUTE_MEMORY_SIZE   = 0x4000;
constexpr size_t VDECSW_CPU_MEMORY_SIZE       = 0x100000;
constexpr size_t VDECSW_GPU_MEMORY_SIZE       = 0x200000;
constexpr size_t VDECSW_MAX_FRAME_BUFFER_SIZE = 0xc00000;

using VdecswDecoder      = void*;
using VdecswComputeQueue = void*;

struct VdecswComputeMemoryInfo {
	size_t this_size;
	size_t cpu_gpu_memory_size;
	void*  cpu_gpu_memory;
};

struct VdecswComputeConfigInfo {
	size_t   this_size;
	uint16_t compute_pipe_id;
	uint16_t compute_queue_id;
	bool     check_memory_type;
	uint8_t  reserved0;
	uint16_t reserved1;
};

struct VdecswDecoderConfigInfo {
	size_t             this_size;
	uint32_t           resource_type;
	uint32_t           codec_type;
	uint32_t           profile;
	uint32_t           max_level;
	int32_t            max_frame_width;
	int32_t            max_frame_height;
	int32_t            max_dpb_frame_count;
	uint32_t           decode_input_queue_depth;
	VdecswComputeQueue compute_queue;
	uint64_t           cpu_affinity_mask;
	int32_t            cpu_thread_priority;
	bool               optimize_progressive_video;
	bool               check_memory_type;
	uint8_t            reserved0;
	uint8_t            reserved1;
	uint64_t           extra0;
	uint64_t           extra1;
};

struct VdecswDecoderMemoryInfo {
	size_t   this_size;
	size_t   cpu_memory_size;
	void*    cpu_memory;
	size_t   gpu_memory_size;
	void*    gpu_memory;
	size_t   cpu_gpu_memory_size;
	void*    cpu_gpu_memory;
	size_t   max_frame_buffer_size;
	uint32_t frame_buffer_alignment;
	uint32_t reserved0;
};

struct VdecswInputData {
	size_t   this_size;
	void*    au_data;
	size_t   au_size;
	uint64_t pts_data;
	uint64_t dts_data;
	uint64_t attached_data;
};

struct VdecswInputSync {
	size_t   this_size;
	uint64_t attached_data;
	uint32_t input_count;
	uint32_t reserved0;
};

struct VdecswFrameBuffer {
	size_t this_size;
	void*  frame_buffer;
	size_t frame_buffer_size;
};

struct VdecswOutputInfo {
	size_t   this_size;
	bool     is_valid;
	bool     is_error_frame;
	bool     is_discarded_frame;
	uint8_t  picture_count;
	uint32_t codec_type;
	uint32_t frame_width;
	uint32_t frame_pitch;
	uint32_t frame_height;
	void*    frame_buffer;
	size_t   frame_buffer_size;
	uint32_t frame_format;
	uint32_t frame_pitch_in_bytes;
};

static_assert(sizeof(VdecswComputeMemoryInfo) == 24);
static_assert(sizeof(VdecswComputeConfigInfo) == 16);
static_assert(sizeof(VdecswDecoderConfigInfo) == 80);
static_assert(sizeof(VdecswDecoderMemoryInfo) == 72);
static_assert(sizeof(VdecswInputData) == 48);
static_assert(sizeof(VdecswInputSync) == 24);
static_assert(sizeof(VdecswFrameBuffer) == 24);
static_assert(sizeof(VdecswOutputInfo) == 56);

struct PendingInput {
	std::vector<uint8_t> data;
	uint64_t             pts      = 0;
	uint64_t             dts      = 0;
	uint64_t             attached = 0;
};

struct DecodedFrame {
	std::vector<uint8_t>            data;
	VideoDec2::Decoder::Output      output;
	VideoDec2::Decoder::PictureInfo picture;
	bool                            has_picture = false;
};

enum class RequestKind { Input, Flush, Reset };

struct DecodeRequest {
	RequestKind  kind = RequestKind::Input;
	PendingInput input;
};

constexpr size_t MaxReadyFrames = 3;

struct DecoderState {
	VideoDec2::Decoder::Instance* instance = nullptr;
	std::deque<DecodeRequest>     requests;
	std::deque<uint64_t>          consumed;
	std::deque<DecodedFrame>      ready;
	std::vector<uint8_t>          staging;
	VdecswFrameBuffer             frame_buffer {};
	bool                          has_frame_buffer = false;
	bool                          finalizing       = false;
	bool                          busy             = false;
	bool                          stop             = false;
	uint64_t                      frames_in        = 0;
	uint64_t                      frames_out       = 0;
	std::mutex                    mutex;
	std::condition_variable       wake;
	std::thread                   worker;
};

static int32_t Fail(const char* where, int32_t code) {
	LOGF("Vdecsw %s failed: 0x%08x\n", where, static_cast<uint32_t>(code));
	return code;
}

static std::mutex                                               g_decoder_mutex;
static std::unordered_map<void*, std::unique_ptr<DecoderState>> g_decoders;
static std::mutex                                               g_picture_mutex;
static std::unordered_map<const void*, VideoDec2::Decoder::PictureInfo> g_pictures;

static DecoderState* GetDecoder(VdecswDecoder decoder) {
	std::scoped_lock lock(g_decoder_mutex);
	const auto       it = g_decoders.find(decoder);
	return it != g_decoders.end() ? it->second.get() : nullptr;
}

// Frames are decoded ahead on a worker so the game thread only copies finished pictures.
static void DecodeWorker(DecoderState* state) {
	std::unique_lock lock(state->mutex);
	for (;;) {
		state->wake.wait(lock, [&] {
			return state->stop || (!state->requests.empty() && state->ready.size() < MaxReadyFrames);
		});
		if (state->stop) {
			return;
		}
		auto request = std::move(state->requests.front());
		state->requests.pop_front();
		state->busy = true;
		lock.unlock();
		DecodedFrame                          frame;
		const auto                            started = std::chrono::steady_clock::now();
		const VideoDec2::Decoder::FrameBuffer target {state->staging.data(), state->staging.size()};
		if (request.kind == RequestKind::Reset) {
			VideoDec2::Decoder::Reset(state->instance);
		} else if (request.kind == RequestKind::Flush) {
			(void)VideoDec2::Decoder::Drain(state->instance, target, &frame.output);
		} else {
			const auto& in     = request.input;
			const auto  result = VideoDec2::Decoder::Decode(
			    state->instance, {in.data.data(), in.data.size(), in.pts, in.dts, in.attached},
			    target, &frame.output);
			if (result != VideoDec2::Decoder::Result::Ok) {
				LOGF("Vdecsw decode failed: %d\n", static_cast<int>(result));
			}
		}
		static std::atomic<uint32_t> timing_count {0};
		if (timing_count.fetch_add(1) < 48) {
			LOGF("Vdecsw worker: kind=%d valid=%d ms=%.1f\n", static_cast<int>(request.kind),
			     frame.output.valid ? 1 : 0,
			     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
			         .count());
		}
		if (frame.output.valid) {
			frame.has_picture =
			    VideoDec2::Decoder::GetPictureInfo(state->staging.data(), &frame.picture);
			const auto used =
			    static_cast<size_t>(frame.output.pitch) * frame.output.height * 3u / 2u;
			const auto bytes =
			    used != 0 ? std::min(used, state->staging.size()) : state->staging.size();
			frame.data.assign(state->staging.begin(), state->staging.begin() + bytes);
		}
		lock.lock();
		state->busy = false;
		if (request.kind == RequestKind::Flush && !frame.output.valid) {
			state->finalizing = false;
			LOGF("Vdecsw drained: in=%" PRIu64 " out=%" PRIu64 " ready=%zu\n", state->frames_in,
			     state->frames_out, state->ready.size());
		}
		if (frame.output.valid) {
			state->ready.push_back(std::move(frame));
		}
	}
}

static void FillNoPictureOutput(DecoderState& state, VdecswOutputInfo* output_info) {
	output_info->is_valid             = false;
	output_info->is_error_frame       = false;
	output_info->is_discarded_frame   = false;
	output_info->picture_count        = 0;
	output_info->codec_type           = VideoDec2::Decoder::GetCodecType(state.instance);
	output_info->frame_width          = 0;
	output_info->frame_pitch          = 0;
	output_info->frame_height         = 0;
	output_info->frame_buffer         = state.frame_buffer.frame_buffer;
	output_info->frame_buffer_size    = state.frame_buffer.frame_buffer_size;
	output_info->frame_format         = 0;
	output_info->frame_pitch_in_bytes = 0;
}

static void ApplyDecodedOutput(const VideoDec2::Decoder::Output& decoded,
                               const VdecswFrameBuffer& target, VdecswOutputInfo* output_info) {
	output_info->is_valid             = true;
	output_info->is_error_frame       = decoded.error_frame;
	output_info->picture_count        = 1;
	output_info->codec_type           = decoded.codec_type;
	output_info->frame_width          = decoded.width;
	output_info->frame_pitch          = decoded.pitch;
	output_info->frame_height         = decoded.height;
	output_info->frame_buffer         = target.frame_buffer;
	output_info->frame_buffer_size    = target.frame_buffer_size;
	output_info->frame_pitch_in_bytes = decoded.pitch;
}

static int32_t KYTY_SYSV_ABI QueryComputeMemoryInfo(VdecswComputeMemoryInfo* info) {
	PRINT_NAME();

	if (info == nullptr) {
		return Fail(__func__, VDECSW_ERROR_ARGUMENT_POINTER);
	}
	if (info->this_size != sizeof(VdecswComputeMemoryInfo)) {
		return Fail(__func__, VDECSW_ERROR_STRUCT_SIZE);
	}
	info->cpu_gpu_memory_size = VDECSW_COMPUTE_MEMORY_SIZE;
	info->cpu_gpu_memory      = nullptr;
	return OK;
}

static int32_t KYTY_SYSV_ABI AllocateComputeQueue(const VdecswComputeConfigInfo* config,
                                                  const VdecswComputeMemoryInfo* memory,
                                                  VdecswComputeQueue*            queue) {
	PRINT_NAME();

	if (config == nullptr || memory == nullptr || queue == nullptr) {
		return Fail(__func__, VDECSW_ERROR_ARGUMENT_POINTER);
	}
	if (config->this_size != sizeof(VdecswComputeConfigInfo) ||
	    memory->this_size != sizeof(VdecswComputeMemoryInfo)) {
		return Fail(__func__, VDECSW_ERROR_STRUCT_SIZE);
	}
	*queue = memory->cpu_gpu_memory != nullptr ? memory->cpu_gpu_memory : queue;
	return OK;
}

static int32_t KYTY_SYSV_ABI ReleaseComputeQueue(VdecswComputeQueue queue) {
	PRINT_NAME();

	return queue != nullptr ? OK : VDECSW_ERROR_ARGUMENT_POINTER;
}

static int32_t KYTY_SYSV_ABI QueryDecoderMemoryInfo(const VdecswDecoderConfigInfo* config,
                                                    VdecswDecoderMemoryInfo*       memory) {
	PRINT_NAME();

	if (config == nullptr || memory == nullptr) {
		return Fail(__func__, VDECSW_ERROR_ARGUMENT_POINTER);
	}
	if (config->this_size != sizeof(VdecswDecoderConfigInfo) ||
	    memory->this_size != sizeof(VdecswDecoderMemoryInfo)) {
		LOGF("\t unexpected struct sizes: config=%" PRIu64 " memory=%" PRIu64 "\n",
		     static_cast<uint64_t>(config->this_size), static_cast<uint64_t>(memory->this_size));
		return Fail(__func__, VDECSW_ERROR_STRUCT_SIZE);
	}
	if (!VideoDec2::Decoder::IsCodecSupported(config->codec_type)) {
		return Fail(__func__, VDECSW_ERROR_CODEC_TYPE);
	}
	LOGF("\t codec=%u profile=%u level=%u size=%dx%d dpb=%d depth=%u\n", config->codec_type,
	     config->profile, config->max_level, config->max_frame_width, config->max_frame_height,
	     config->max_dpb_frame_count, config->decode_input_queue_depth);

	memory->cpu_memory_size        = VDECSW_CPU_MEMORY_SIZE;
	memory->cpu_memory             = nullptr;
	memory->gpu_memory_size        = VDECSW_GPU_MEMORY_SIZE;
	memory->gpu_memory             = nullptr;
	memory->cpu_gpu_memory_size    = VDECSW_GPU_MEMORY_SIZE;
	memory->cpu_gpu_memory         = nullptr;
	memory->max_frame_buffer_size  = VDECSW_MAX_FRAME_BUFFER_SIZE;
	memory->frame_buffer_alignment = 0x100;
	memory->reserved0              = 0;
	return OK;
}

static int32_t KYTY_SYSV_ABI CreateDecoder(const VdecswDecoderConfigInfo* config,
                                           const VdecswDecoderMemoryInfo* memory,
                                           VdecswDecoder*                 decoder) {
	PRINT_NAME();

	LOGF("Vdecsw CreateDecoder config=%p memory=%p\n", static_cast<const void*>(config),
	     static_cast<const void*>(memory));

	if (config == nullptr || memory == nullptr || decoder == nullptr) {
		return Fail(__func__, VDECSW_ERROR_ARGUMENT_POINTER);
	}
	if (config->this_size != sizeof(VdecswDecoderConfigInfo) ||
	    memory->this_size != sizeof(VdecswDecoderMemoryInfo)) {
		return Fail(__func__, VDECSW_ERROR_STRUCT_SIZE);
	}
	if (!VideoDec2::Decoder::IsCodecSupported(config->codec_type)) {
		return Fail(__func__, VDECSW_ERROR_CODEC_TYPE);
	}

	auto* instance = VideoDec2::Decoder::Create(
	    {config->codec_type, config->max_frame_width, config->max_frame_height});
	if (instance == nullptr) {
		return Fail(__func__, VDECSW_ERROR_API_FAIL);
	}

	auto state      = std::make_unique<DecoderState>();
	state->instance = instance;
	state->staging.resize(VDECSW_MAX_FRAME_BUFFER_SIZE);
	auto* handle  = state.get();
	state->worker = std::thread(DecodeWorker, handle);
	{
		std::scoped_lock lock(g_decoder_mutex);
		g_decoders.emplace(handle, std::move(state));
	}
	*decoder = handle;
	return OK;
}

static int32_t KYTY_SYSV_ABI DeleteDecoder(VdecswDecoder decoder) {
	PRINT_NAME();

	std::unique_ptr<DecoderState> state;
	{
		std::scoped_lock lock(g_decoder_mutex);
		const auto       it = g_decoders.find(decoder);
		if (it == g_decoders.end()) {
			return Fail(__func__, VDECSW_ERROR_DECODER_INSTANCE);
		}
		state = std::move(it->second);
		g_decoders.erase(it);
	}
	{
		std::scoped_lock lock(state->mutex);
		state->stop = true;
	}
	state->wake.notify_all();
	state->worker.join();
	VideoDec2::Decoder::Destroy(state->instance);
	return OK;
}

static int32_t KYTY_SYSV_ABI ResetDecoder(VdecswDecoder decoder) {
	PRINT_NAME();

	auto* state = GetDecoder(decoder);
	if (state == nullptr) {
		return Fail(__func__, VDECSW_ERROR_DECODER_INSTANCE);
	}
	{
		std::scoped_lock lock(state->mutex);
		state->requests.clear();
		state->ready.clear();
		state->consumed.clear();
		state->finalizing = false;
		state->requests.push_back({RequestKind::Reset, {}});
	}
	state->wake.notify_all();
	return OK;
}

static int32_t KYTY_SYSV_ABI SetDecodeInput(VdecswDecoder decoder, const VdecswInputData* input) {
	PRINT_NAME();

	static std::atomic<uint32_t> trace_count {0};
	if (trace_count.fetch_add(1) < 64) {
		LOGF("Vdecsw SetDecodeInput decoder=%p size=%" PRIu64 " au=%p bytes=%" PRIu64 "\n", decoder,
		     static_cast<uint64_t>(input != nullptr ? input->this_size : 0),
		     input != nullptr ? input->au_data : nullptr,
		     static_cast<uint64_t>(input != nullptr ? input->au_size : 0));
	}

	auto* state = GetDecoder(decoder);
	if (state == nullptr) {
		return Fail(__func__, VDECSW_ERROR_DECODER_INSTANCE);
	}
	if (input == nullptr) {
		return Fail(__func__, VDECSW_ERROR_ARGUMENT_POINTER);
	}
	if (input->this_size != sizeof(VdecswInputData)) {
		return Fail(__func__, VDECSW_ERROR_STRUCT_SIZE);
	}
	if (input->au_data == nullptr || input->au_size == 0) {
		return Fail(__func__, VDECSW_ERROR_ARGUMENT_POINTER);
	}
	{
		std::scoped_lock lock(state->mutex);
		const auto*      bytes = static_cast<const uint8_t*>(input->au_data);
		state->requests.push_back(
		    {RequestKind::Input,
		     {std::vector<uint8_t>(bytes, bytes + input->au_size), input->pts_data,
		      input->dts_data, input->attached_data}});
		// The access unit is copied, so the game may reuse its buffer at once.
		state->consumed.push_back(input->attached_data);
		state->finalizing = false;
		state->frames_in++;
	}
	state->wake.notify_all();
	return OK;
}

static int32_t KYTY_SYSV_ABI SetDecodeOutput(VdecswDecoder            decoder,
                                             const VdecswFrameBuffer* frame_buffer) {
	PRINT_NAME();

	static std::atomic<uint32_t> trace_count {0};
	if (trace_count.fetch_add(1) < 64) {
		LOGF("Vdecsw SetDecodeOutput decoder=%p size=%" PRIu64 " buffer=%p bytes=%" PRIu64 "\n",
		     decoder, static_cast<uint64_t>(frame_buffer != nullptr ? frame_buffer->this_size : 0),
		     frame_buffer != nullptr ? frame_buffer->frame_buffer : nullptr,
		     static_cast<uint64_t>(frame_buffer != nullptr ? frame_buffer->frame_buffer_size : 0));
	}

	auto* state = GetDecoder(decoder);
	if (state == nullptr) {
		return Fail(__func__, VDECSW_ERROR_DECODER_INSTANCE);
	}
	if (frame_buffer == nullptr) {
		return Fail(__func__, VDECSW_ERROR_ARGUMENT_POINTER);
	}
	if (frame_buffer->this_size != sizeof(VdecswFrameBuffer)) {
		return Fail(__func__, VDECSW_ERROR_STRUCT_SIZE);
	}
	if (frame_buffer->frame_buffer == nullptr || frame_buffer->frame_buffer_size == 0) {
		return Fail(__func__, VDECSW_ERROR_ARGUMENT_POINTER);
	}
	std::scoped_lock lock(state->mutex);
	state->frame_buffer     = *frame_buffer;
	state->has_frame_buffer = true;
	return OK;
}

static int32_t KYTY_SYSV_ABI TrySyncDecodeInput(VdecswDecoder decoder, VdecswInputSync* sync) {
	PRINT_NAME();

	auto* state = GetDecoder(decoder);
	if (state == nullptr) {
		return Fail(__func__, VDECSW_ERROR_DECODER_INSTANCE);
	}
	if (sync == nullptr) {
		return Fail(__func__, VDECSW_ERROR_ARGUMENT_POINTER);
	}
	if (sync->this_size != sizeof(VdecswInputSync)) {
		return Fail(__func__, VDECSW_ERROR_STRUCT_SIZE);
	}
	std::scoped_lock lock(state->mutex);
	if (!state->consumed.empty()) {
		sync->attached_data = state->consumed.front();
		sync->input_count   = 1;
		state->consumed.pop_front();
		return OK;
	}
	sync->attached_data = 0;
	sync->input_count   = 0;
	const auto code = state->requests.empty() && !state->busy ? VDECSW_ERROR_INPUT_QUEUE_EMPTY
	                                                          : VDECSW_ERROR_DECODE_PENDING;
	static std::atomic<uint32_t> end_log {0};
	if (state->finalizing && end_log.fetch_add(1) < 40) {
		LOGF("Vdecsw TrySyncDecodeInput at end: 0x%08x\n", static_cast<uint32_t>(code));
	}
	return code;
}

static int32_t KYTY_SYSV_ABI TrySyncDecodeOutput(VdecswDecoder decoder, VdecswOutputInfo* output) {
	PRINT_NAME();

	static std::atomic<uint32_t> trace_count {0};
	if (trace_count.fetch_add(1) < 64) {
		LOGF("Vdecsw TrySyncDecodeOutput decoder=%p output=%p size=%" PRIu64 "\n", decoder,
		     static_cast<void*>(output),
		     static_cast<uint64_t>(output != nullptr ? output->this_size : 0));
	}

	auto* state = GetDecoder(decoder);
	if (state == nullptr) {
		return Fail(__func__, VDECSW_ERROR_DECODER_INSTANCE);
	}
	if (output == nullptr) {
		return Fail(__func__, VDECSW_ERROR_ARGUMENT_POINTER);
	}
	if (output->this_size != sizeof(VdecswOutputInfo)) {
		return Fail(__func__, VDECSW_ERROR_STRUCT_SIZE);
	}
	std::scoped_lock lock(state->mutex);
	if (!state->has_frame_buffer) {
		return VDECSW_ERROR_OUTPUT_PENDING;
	}
	FillNoPictureOutput(*state, output);
	if (!state->ready.empty()) {
		auto frame = std::move(state->ready.front());
		state->ready.pop_front();
		const auto bytes = std::min(frame.data.size(), state->frame_buffer.frame_buffer_size);
		// Writing through the backing alias avoids a protection fault on every tracked page.
		const auto target = reinterpret_cast<uint64_t>(state->frame_buffer.frame_buffer);
		LibKernel::Memory::InvalidateMemory(target, bytes);
		if (!LibKernel::Memory::TryWriteBacking(target, frame.data.data(), bytes)) {
			std::memcpy(state->frame_buffer.frame_buffer, frame.data.data(), bytes);
		}
		state->frames_out++;
		ApplyDecodedOutput(frame.output, state->frame_buffer, output);
		if (frame.has_picture) {
			std::scoped_lock picture_lock(g_picture_mutex);
			g_pictures[state->frame_buffer.frame_buffer] = frame.picture;
		}
		state->has_frame_buffer = false;
		state->wake.notify_all();
		return OK;
	}
	if (state->requests.empty() && !state->busy && state->finalizing) {
		state->requests.push_back({RequestKind::Flush, {}});
		state->wake.notify_all();
	}
	static std::atomic<uint32_t> end_log {0};
	if (state->frames_out + 4 >= state->frames_in && state->frames_in > 100 &&
	    end_log.fetch_add(1) < 40) {
		LOGF("Vdecsw TrySyncDecodeOutput pending: in=%" PRIu64 " out=%" PRIu64 " finalizing=%d\n",
		     state->frames_in, state->frames_out, state->finalizing ? 1 : 0);
	}
	return VDECSW_ERROR_DECODE_PENDING;
}

static int32_t KYTY_SYSV_ABI FinalizeDecodeSequence(VdecswDecoder decoder) {
	PRINT_NAME();

	auto* state = GetDecoder(decoder);
	if (state == nullptr) {
		return Fail(__func__, VDECSW_ERROR_DECODER_INSTANCE);
	}
	{
		std::scoped_lock lock(state->mutex);
		state->finalizing = true;
		LOGF("Vdecsw FinalizeDecodeSequence: in=%" PRIu64 " out=%" PRIu64 " pending=%zu ready=%zu\n",
		     state->frames_in, state->frames_out, state->requests.size(), state->ready.size());
		if (state->requests.empty() && !state->busy) {
			state->requests.push_back({RequestKind::Flush, {}});
		}
	}
	state->wake.notify_all();
	return OK;
}

static int32_t KYTY_SYSV_ABI GetAvcPictureInfo(const VdecswOutputInfo* output, void* first,
                                               void* second) {
	PRINT_NAME();

	if (output == nullptr || first == nullptr) {
		return Fail(__func__, VDECSW_ERROR_ARGUMENT_POINTER);
	}
	if (output->this_size != sizeof(VdecswOutputInfo)) {
		return Fail(__func__, VDECSW_ERROR_STRUCT_SIZE);
	}
	if (!output->is_valid || output->picture_count == 0 || output->frame_buffer == nullptr) {
		return Fail(__func__, VDECSW_ERROR_OUTPUT_INFO);
	}
	VideoDec2::Decoder::PictureInfo decoded {};
	{
		std::scoped_lock picture_lock(g_picture_mutex);
		const auto       it = g_pictures.find(output->frame_buffer);
		if (it == g_pictures.end()) {
			return Fail(__func__, VDECSW_ERROR_OUTPUT_INFO);
		}
		decoded = it->second;
	}
	const auto fill = [&decoded](void* destination, bool valid) -> int32_t {
		auto*      bytes = static_cast<uint8_t*>(destination);
		const auto size  = *static_cast<const size_t*>(destination);
		if (size < 40 || size > 256) {
			return Fail(__func__, VDECSW_ERROR_STRUCT_SIZE);
		}
		std::memset(bytes + sizeof(size_t), 0, size - sizeof(size_t));
		bytes[8] = valid ? 1 : 0;
		if (!valid) {
			return OK;
		}
		std::memcpy(bytes + 16, &decoded.pts, sizeof(decoded.pts));
		std::memcpy(bytes + 24, &decoded.dts, sizeof(decoded.dts));
		std::memcpy(bytes + 32, &decoded.attached_data, sizeof(decoded.attached_data));
		if (size >= 120) {
			bytes[40]                                   = decoded.key_frame ? 1 : 0;
			bytes[41]                                   = static_cast<uint8_t>(decoded.profile);
			bytes[42]                                   = static_cast<uint8_t>(decoded.level);
			*reinterpret_cast<uint32_t*>(bytes + 44)    = (decoded.width + 15u) / 16u - 1u;
			*reinterpret_cast<uint32_t*>(bytes + 48)    = (decoded.height + 15u) / 16u - 1u;
			bytes[52]                                   = 1;
			bytes[53]                                   = decoded.crop_left != 0 ||
			                                                      decoded.crop_right != 0 ||
			                                                      decoded.crop_top != 0 ||
			                                                      decoded.crop_bottom != 0
			                                                  ? 1
			                                                  : 0;
			*reinterpret_cast<uint32_t*>(bytes + 56)    = decoded.crop_left;
			*reinterpret_cast<uint32_t*>(bytes + 60)    = decoded.crop_right;
			*reinterpret_cast<uint32_t*>(bytes + 64)    = decoded.crop_top;
			*reinterpret_cast<uint32_t*>(bytes + 68)    = decoded.crop_bottom;
			bytes[72]                                   = decoded.sar_width != 0 ? 1 : 0;
			bytes[73]                                   = decoded.sar_width != 0 ? 255 : 0;
			*reinterpret_cast<uint16_t*>(bytes + 74)    = decoded.sar_width;
			*reinterpret_cast<uint16_t*>(bytes + 76)    = decoded.sar_height;
			bytes[78]                                   = 1;
			bytes[79]                                   = 5;
			bytes[80]                                   = decoded.color_range == 2 ? 1 : 0;
			bytes[81]                                   = 1;
			bytes[82]                                   = decoded.color_primaries;
			bytes[83]                                   = decoded.color_trc;
			bytes[84]                                   = decoded.color_space;
		}
		return OK;
	};
	const auto result = fill(first, true);
	if (result != OK) {
		return result;
	}
	return second != nullptr ? fill(second, false) : OK;
}

void DumpDecoderState(FILE* out) {
	std::scoped_lock lock(g_decoder_mutex);
	for (const auto& [handle, state]: g_decoders) {
		std::scoped_lock state_lock(state->mutex);
		std::fprintf(out,
		             "vdecsw %p in=%llu out=%llu requests=%zu ready=%zu consumed=%zu busy=%d "
		             "has_frame_buffer=%d finalizing=%d\n",
		             handle, static_cast<unsigned long long>(state->frames_in),
		             static_cast<unsigned long long>(state->frames_out), state->requests.size(),
		             state->ready.size(), state->consumed.size(), state->busy ? 1 : 0,
		             state->has_frame_buffer ? 1 : 0, state->finalizing ? 1 : 0);
	}
}

LIB_DEFINE(InitVdecsw_1) {
	PRINT_NAME();

	LIB_FUNC("0moTubWCsTM", QueryComputeMemoryInfo);
	LIB_FUNC("hIgrg5h4V6s", AllocateComputeQueue);
	LIB_FUNC("fX-zOOefbbs", ReleaseComputeQueue);
	LIB_FUNC("A+2M7EivuOU", QueryDecoderMemoryInfo);
	LIB_FUNC("+L5ArV1tPGA", CreateDecoder);
	LIB_FUNC("ecUtPX+dBYk", DeleteDecoder);
	LIB_FUNC("veb-YBrOqo0", ResetDecoder);
	LIB_FUNC("aqMiF0AgUYI", SetDecodeInput);
	LIB_FUNC("rgtMCOpyBSc", SetDecodeOutput);
	LIB_FUNC("l4sQYy5wPkc", TrySyncDecodeInput);
	LIB_FUNC("kMBw37oH8nI", TrySyncDecodeOutput);
	LIB_FUNC("5Y6nZqIZvBg", FinalizeDecodeSequence);
	LIB_FUNC("ihNT-uuEAr4", GetAvcPictureInfo);
}

} // namespace Vdecsw

} // namespace Libs
