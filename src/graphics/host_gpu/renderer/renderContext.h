#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include <string>
#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <array>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

void PrintRecentShaders();
void SetGpuCheckpointsEnabled(bool enabled);
// Leaves an NV diagnostic checkpoint when KYTY_GPU_CHECKPOINTS is set; a no-op otherwise.
void GpuCheckpoint(vk::CommandBuffer command, uint64_t marker);
// KYTY_FLIP_TRACE=1: draws per color-target address since the last flip, reported at each flip.
bool        FlipTraceEnabled();
void        NoteColorTargetWrite(uint64_t address);
// Render target image per address (KYTY_FLIP_TRACE): a sampled texture at that address that
// resolves to another image reads memory the render target never wrote back.
void        NoteRenderTargetImage(uint64_t address, uint32_t image_index, uint32_t format,
                                  uint32_t width, uint32_t height);
void        CheckSampledAlias(uint64_t address, uint32_t image_index, uint32_t format,
                              uint32_t width, uint32_t height);
std::string TakeFrameTargetStats(uint64_t presented_address);
// Logs the command scheduler's submit and priority-operation state for hang reports.
void ReportSchedulerState();

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	Common::Mutex&      GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	void               PrepareBda();
	void               RunGarbageCollector();
	void               PrefetchReadbacks();

	// KYTY_GPU_TIMING=1: a timestamp after every draw and dispatch, aggregated per shader and
	// printed once a second, so GPU time can be attributed to passes.
	void GpuTimerBegin(vk::CommandBuffer command);
	void GpuTimerMark(vk::CommandBuffer command, uint64_t label, uint8_t kind, uint64_t detail = 0);
	// Copies an indirect dispatch's argument triple to host memory so the report can show it.
	void GpuTimerCaptureArgs(vk::CommandBuffer command, vk::Buffer args, uint64_t offset,
	                         uint64_t vaddr);
	bool GpuTimerActive() const { return m_timer_enabled && m_timer_current >= 0; }
	// Rewrites an indirect dispatch's argument triple into a scratch buffer with each count
	// clamped to the device limit. Returns false when the source has no device address.
	// Copies indirect arguments into a clamped scratch slot: a dispatch triple, or an indexed
	// draw when max_indices (indices in the bound index buffer) is non-zero.
	bool ClampIndirectArgs(vk::CommandBuffer command, const Buffer& source, uint64_t offset,
	                       vk::Buffer& out_buffer, uint64_t& out_offset, uint32_t max_indices = 0);
	void GpuTimerReport();

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	GraphicContext&           m_graphics;
	Common::Mutex             m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	bool                      m_fault_process_pending = false;
	// CPU-write generation at the last full BDA sync; UINT64_MAX forces the next one.
	uint64_t                  m_bda_synced_generation = UINT64_MAX;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;

	struct GpuTimerEntry {
		uint32_t query = 0;
		uint64_t label  = 0;
		uint64_t detail = 0;
		uint64_t vaddr  = 0;
		uint8_t  kind   = 0;
	};
	struct GpuTimerBlock {
		uint32_t                   first  = 0;
		uint32_t                   used   = 0;
		uint64_t                   tick   = 0;
		bool                       active = false;
		std::vector<GpuTimerEntry> entries;
	};
	struct GpuTimerStat {
		double   ms    = 0.0;
		uint64_t count = 0;
		uint64_t detail = 0; // groups of the slowest dispatch (x | y << 20 | z << 40)
		uint32_t raw_x  = 0; // unmasked x group count of an indirect dispatch
		uint64_t args_vaddr = 0;
		std::array<uint32_t, 8> around {}; // GPU memory from args-8 to args+24
		double   max_ms = 0.0;
		uint8_t  kind   = 0;
	};
	static constexpr uint32_t GpuTimerBlockQueries = 512;
	static constexpr uint32_t GpuTimerBlockCount   = 64;
	bool                      m_timer_enabled      = false;
	bool                      m_timer_initialized  = false;
	vk::QueryPool             m_timer_pool         = nullptr;
	std::vector<GpuTimerBlock> m_timer_blocks;
	std::unique_ptr<Buffer>    m_timer_args;
	vk::PipelineLayout         m_clamp_layout   = nullptr;
	vk::Pipeline               m_clamp_pipeline = nullptr;
	std::unique_ptr<Buffer>    m_clamp_scratch;
	uint32_t                   m_clamp_slot        = 0;
	bool                       m_clamp_initialized = false;
	uint64_t                   m_timer_pending_vaddr = 0;
	int                        m_timer_current = -1;
	std::unordered_map<uint64_t, GpuTimerStat> m_timer_stats;
	double                                     m_timer_kind_ms[3] {};
	uint64_t                                   m_timer_last_report = 0;
	void                                       GpuTimerCollect(GpuTimerBlock& block);
	std::chrono::steady_clock::time_point m_last_pipeline_save = std::chrono::steady_clock::now();
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
