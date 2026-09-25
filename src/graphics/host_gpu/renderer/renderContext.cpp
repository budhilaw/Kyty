#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/frameCapture.h"

#include "common/assert.h"
#include "common/threads.h"
#include "common/timer.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/indirect_args_clamp_spv.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <atomic>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache) {
	FrameCapture::SetContext(this);
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
	if (m_clamp_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_clamp_pipeline, nullptr);
	}
	if (m_clamp_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_clamp_layout, nullptr);
	}
}

namespace {
struct ClampPush {
	uint64_t src;
	uint64_t dst;
	uint32_t limit_x;
	uint32_t limit_y;
	uint32_t limit_z;
	uint32_t mode;
};
constexpr uint32_t ClampSlots = 4096;
} // namespace

void RenderContext::RecordClamp(vk::CommandBuffer command, uint64_t src, uint64_t dst,
                                uint32_t limit_x, uint32_t limit_y, uint32_t limit_z,
                                uint32_t mode, uint32_t groups) {
	const ClampPush push {src, dst, limit_x, limit_y, limit_z, mode};

	// The counts may come from a shader, a copy or a table upload recorded just before.
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);

	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_clamp_pipeline);
	command.pushConstants(m_clamp_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                      &push);
	GpuCheckpoint(command, 0xF000000000000003ull);
	command.dispatch(groups, 1, 1);

	vk::MemoryBarrier2 after {};
	after.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	after.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	after.dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect;
	after.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead;
	dependency.pMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);
}

bool RenderContext::ClampDrawArgs(vk::CommandBuffer command, uint64_t vaddr, const Buffer& source,
                                  uint64_t offset, uint32_t max_indices, vk::Buffer& out_buffer,
                                  uint64_t& out_offset) {
	// Persistent slots take the low half of the scratch buffer; the dispatch ring the high half.
	constexpr uint32_t BulkSlots = ClampSlots / 2;
	if (!source.HasDeviceAddress() || !EnsureClampPipeline()) {
		return false;
	}
	auto&    scheduler = GetCommandScheduler();
	uint32_t slot      = UINT32_MAX;
	bool     single    = false;
	if (const auto found = m_draw_clamp_index.find(vaddr); found != m_draw_clamp_index.end()) {
		slot        = found->second;
		auto& entry = m_draw_clamp_entries[slot];
		if (max_indices < entry.max_indices) {
			// A smaller index buffer than the slot was clamped for: clamp this draw on its own.
			entry.max_indices = max_indices;
			single            = true;
		} else if (max_indices > entry.max_indices) {
			entry.max_indices  = max_indices;
			m_draw_clamp_stale = true;
		}
	} else if (m_draw_clamp_entries.size() < BulkSlots) {
		slot = static_cast<uint32_t>(m_draw_clamp_entries.size());
		m_draw_clamp_entries.push_back({vaddr, max_indices});
		m_draw_clamp_index.emplace(vaddr, slot);
		single = true;
	} else {
		scheduler.EndRendering();
		return ClampIndirectArgs(command, source, offset, out_buffer, out_offset, max_indices);
	}
	const auto dst = m_clamp_scratch->BufferDeviceAddress() + uint64_t {slot} * 32u;
	if (single) {
		scheduler.EndRendering();
		RecordClamp(command, source.BufferDeviceAddress() + offset, dst, max_indices, 1u << 16u,
		            0, 1u, 1);
	} else if (TakeDrawClampStale() || m_draw_clamp_stale) {
		m_draw_clamp_stale = false;
		scheduler.EndRendering();
		// One dispatch refreshes every slot from the arguments' current buffers.
		struct Entry {
			uint64_t src;
			uint32_t max_indices;
			uint32_t pad;
		};
		std::vector<Entry> table(m_draw_clamp_entries.size());
		for (size_t i = 0; i < table.size(); i++) {
			const auto& entry = m_draw_clamp_entries[i];
			auto [buffer, buffer_offset] =
			    m_buffer_cache.ObtainBuffer(entry.vaddr, sizeof(vk::DrawIndexedIndirectCommand), false);
			const bool ok = buffer != nullptr && (buffer_offset & 3u) == 0 && buffer->HasDeviceAddress();
			table[i] = {ok ? buffer->BufferDeviceAddress() + buffer_offset : 0u, entry.max_indices, 0u};
		}
		const auto table_bytes = table.size() * sizeof(Entry);
		if (m_draw_clamp_table == nullptr || m_draw_clamp_table->Size() < table_bytes) {
			m_draw_clamp_table = std::make_unique<Buffer>(
			    m_graphics, m_command_scheduler, MemoryUsage::DeviceLocal, 0,
			    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
			        vk::BufferUsageFlagBits::eShaderDeviceAddress,
			    uint64_t {BulkSlots} * sizeof(Entry));
			SetVulkanObjectNameF(m_graphics.device, m_draw_clamp_table->Handle(),
			                     "Kyty.DrawArgsClampTable");
		}
		m_buffer_cache.WriteDataBuffer(*m_draw_clamp_table, 0, table.data(), table_bytes);
		RecordClamp(command, m_draw_clamp_table->BufferDeviceAddress(),
		            m_clamp_scratch->BufferDeviceAddress(), static_cast<uint32_t>(table.size()), 0,
		            0, 2u, static_cast<uint32_t>((table.size() + 63u) / 64u));
	}
	out_buffer = m_clamp_scratch->Handle();
	out_offset = uint64_t {slot} * 32u;
	return true;
}

bool RenderContext::ClampIndirectArgs(vk::CommandBuffer command, const Buffer& source,
                                      uint64_t offset, vk::Buffer& out_buffer,
                                      uint64_t& out_offset, uint32_t max_indices) {
	if (!source.HasDeviceAddress() || !EnsureClampPipeline()) {
		return false;
	}
	// Dispatch clamps use the ring in the high half; draw slots own the low half.
	const auto  slot  = ClampSlots / 2 + (m_clamp_slot++) % (ClampSlots / 2);
	const auto  dst   = uint64_t {slot} * 32u;
	const auto& limit = m_graphics.physical_device_properties.limits.maxComputeWorkGroupCount;
	RecordClamp(command, source.BufferDeviceAddress() + offset,
	            m_clamp_scratch->BufferDeviceAddress() + dst,
	            max_indices != 0 ? max_indices : limit[0],
	            max_indices != 0 ? (1u << 16u) : limit[1], limit[2], max_indices != 0 ? 1u : 0u, 1);
	out_buffer = m_clamp_scratch->Handle();
	out_offset = dst;
	return true;
}

bool RenderContext::EnsureClampPipeline() {
	if (!m_clamp_initialized) {
		m_clamp_initialized = true;
		const vk::PushConstantRange  push_range {vk::ShaderStageFlagBits::eCompute, 0,
		                                        sizeof(ClampPush)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push_range;
		if (m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_clamp_layout) !=
		    vk::Result::eSuccess) {
			return false;
		}
		const auto module = CompileSPV(INDIRECT_ARGS_CLAMP_SPV, m_graphics.device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName  = "main";
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage  = stage;
		pipeline_info.layout = m_clamp_layout;
		const auto result    = m_graphics.device.createComputePipelines(
            nullptr, 1, &pipeline_info, nullptr, &m_clamp_pipeline);
		m_graphics.device.destroyShaderModule(module, nullptr);
		if (result != vk::Result::eSuccess) {
			m_clamp_pipeline = nullptr;
			return false;
		}
		m_clamp_scratch = std::make_unique<Buffer>(
		    m_graphics, m_command_scheduler, MemoryUsage::DeviceLocal, 0,
		    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
		        vk::BufferUsageFlagBits::eShaderDeviceAddress,
		    uint64_t {ClampSlots} * 32u);
		SetVulkanObjectNameF(m_graphics.device, m_clamp_scratch->Handle(),
		                     "Kyty.IndirectArgsClamp");
	}
	return m_clamp_pipeline != nullptr;
}

static std::atomic<RenderContext*> g_debug_context {nullptr};

void ReportSchedulerState() {
	auto* context = g_debug_context.load();
	if (context == nullptr) {
		return;
	}
	auto&    scheduler   = context->GetCommandScheduler();
	size_t   depth       = 0;
	uint64_t head_tick   = 0;
	bool     active      = false;
	uint64_t active_tick = 0;
	scheduler.ReportPriorityQueue(&depth, &head_tick, &active, &active_tick);
	LOGF("\t SCHEDULER: current_tick=%" PRIu64 " gpu_tick=%" PRIu64 " priority_depth=%zu head=%" PRIu64
	     " active=%d active_tick=%" PRIu64 " work_since_submit=%u\n",
	     scheduler.CurrentTick(), scheduler.KnownGpuTick(), depth, head_tick, active ? 1 : 0,
	     active_tick, scheduler.WorkSinceSubmit());
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	g_debug_context.store(this);
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	// GPU tracking never removes execute rights, so retrying an execute fault would spin forever.
	if (!IsMapped(fault_vaddr, fault_size) || access == PageFaultAccess::Execute) {
		return false;
	}
	if (access == PageFaultAccess::Write) {
		if (uint64_t hit = 0; CoversRecentIndirectArgs(fault_vaddr & ~uint64_t {0xfff}, 0x1000, hit)) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1) < 4096) {
				LOGF("SHADERDUMP cpu write fault near args=0x%016" PRIx64 " at=0x%016" PRIx64
				     " gpu_thread=%d cpu_dirty=%d gpu_dirty=%d\n",
				     hit, fault_vaddr, GuestGpu::IsGpuThread() ? 1 : 0,
				     m_buffer_cache.IsRegionCpuModified(fault_vaddr, 1) ? 1 : 0,
				     m_buffer_cache.IsRegionGpuModified(fault_vaddr, 1) ? 1 : 0);
			}
		}
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		// Every read of GPU-written memory drains the GPU, so the trace reports how often it
		// happens and which pages keep causing it.
		if (Common::WaitTrace::Enabled()) {
			static Common::Mutex                           stats_mutex;
			static uint64_t                                stats_start = 0;
			static uint64_t                                stats_count = 0;
			static std::array<std::pair<uint64_t, uint64_t>, 8> stats_pages {};
			static std::array<uint64_t, 8>                      stats_gpu {};
			static std::array<uint32_t, 8>                      stats_thread {};
			static std::array<std::array<uint64_t, 5>, 8>       stats_stack {};
			Common::LockGuard                              lock(stats_mutex);
			const auto now  = Common::Timer::QueryPerformanceCounter();
			const auto page = fault_vaddr & ~uint64_t {0x3fff};
			const bool on_gpu_thread = GuestGpu::IsGpuThread();
			stats_count++;
			bool found = false;
			for (size_t i = 0; i < stats_pages.size(); i++) {
				if (stats_pages[i].first == page) {
					stats_pages[i].second++;
					stats_gpu[i] += on_gpu_thread ? 1 : 0;
					found = true;
					break;
				}
			}
			if (!found) {
				for (size_t i = 0; i < stats_pages.size(); i++) {
					if (stats_pages[i].first == 0) {
						stats_pages[i] = {page, 1};
						stats_gpu[i]   = on_gpu_thread ? 1 : 0;
#ifdef _WIN32
						stats_thread[i] = GetCurrentThreadId();
						void*      frames[32] {};
						const auto captured = RtlCaptureStackBackTrace(0, 32, frames, nullptr);
						const auto base = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
						size_t     kept = 0;
						for (unsigned f = 0; f < captured && kept < stats_stack[i].size(); f++) {
							HMODULE module = nullptr;
							if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
							                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
							                       static_cast<const char*>(frames[f]), &module) &&
							    reinterpret_cast<uint64_t>(module) == base) {
								stats_stack[i][kept++] = reinterpret_cast<uint64_t>(frames[f]) - base;
							}
						}
#endif
						break;
					}
				}
			}
			const auto frequency = Common::Timer::QueryPerformanceFrequency();
			if (stats_start == 0) {
				stats_start = now;
			} else if (frequency != 0 && now - stats_start >= frequency) {
				std::printf("READFAULT %" PRIu64 "/s:", stats_count);
				for (size_t i = 0; i < stats_pages.size(); i++) {
					if (stats_pages[i].first != 0) {
						std::printf(" 0x%" PRIx64 "x%" PRIu64 "(gpu%" PRIu64 " tid%" PRIu32 " [",
						            stats_pages[i].first, stats_pages[i].second, stats_gpu[i],
						            stats_thread[i]);
						for (const auto rva: stats_stack[i]) {
							std::printf("%" PRIx64 ",", rva);
						}
						std::printf("])");
					}
				}
				std::printf("\n");
				std::fflush(stdout);
				stats_start = now;
				stats_count = 0;
				stats_pages.fill({});
				stats_gpu.fill(0);
				stats_thread.fill(0);
				stats_stack.fill({});
			}
		}
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
#ifdef _WIN32
		// A page that stays unreadable after the readback will fault forever; say who holds it.
		MEMORY_BASIC_INFORMATION mbi {};
		if (VirtualQuery(reinterpret_cast<void*>(fault_vaddr), &mbi, sizeof(mbi)) != 0 &&
		    (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
			static std::atomic<uint32_t> stuck_log {0};
			if ((stuck_log.fetch_add(1) % 50000) == 0) {
				LOGF("READFAULT stuck: addr=0x%010" PRIx64 " protect=0x%lx state=0x%lx type=0x%lx "
				     "gpu_tracked=%d texture=%d\n",
				     fault_vaddr, mbi.Protect, mbi.State, mbi.Type,
				     m_buffer_cache.IsRegionGpuModified(fault_vaddr & ~uint64_t {0xfff}, 0x1000) ? 1 : 0,
				     m_texture_cache.IsRegionGpuModified(fault_vaddr & ~uint64_t {0xfff}, 0x1000) ? 1
				                                                                              : 0);
			}
		}
#endif
	}
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_bda_synced_generation = UINT64_MAX;
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Only a range that GPU resources still reference needs the pipe drained; thread
		// stacks and other CPU-only memory come and go many times a second.
		const bool referenced = m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		                        static_cast<bool>(m_texture_cache.FindImageFromRange(vaddr, size, false));
		if (referenced && m_command_scheduler.Active()) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	{
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 64) {
			LOGF("UNMAP: addr=0x%016" PRIx64 " size=0x%" PRIx64 " (drains the GPU)\n", vaddr, size);
			Common::WaitTrace::PrintHostStack("unmap");
		}
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	std::shared_lock lock(m_mapped_ranges_mutex);
	// Every dispatch used to walk every mapped buffer (about 15% of the GPU thread). Nothing
	// can need an upload unless some page became CPU-modified since the last full walk.
	const auto generation = m_buffer_cache.CpuGeneration();
	if (generation != m_bda_synced_generation) {
		// The game writes some page every frame, so the generation nearly always moved and the
		// full walk still cost ~16% of the GPU thread. Only the ranges written since the last
		// sync are walked now; a full walk remains for new mappings and log overflow.
		static const bool full_walk = std::getenv("KYTY_BDA_FULL_WALK") != nullptr;
		bool full = full_walk || m_bda_synced_generation == UINT64_MAX;
		thread_local std::vector<std::pair<uint64_t, uint64_t>> dirty;
		bool                                                    log_full = false;
		m_buffer_cache.TakeCpuDirtyLog(dirty, log_full);
		full = full || log_full;
		if (full) {
			m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
				m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
			});
		} else {
			for (const auto& [address, size]: dirty) {
				m_mapped_ranges.ForEachInRange(address, size, [this](uint64_t start, uint64_t end) {
					m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
				});
			}
		}
		m_bda_synced_generation = generation;
	}
	m_fault_process_pending = true;
}

void RenderContext::PrefetchReadbacks() {
	m_buffer_cache.PrefetchReadbacks();
}

void RenderContext::GpuTimerBegin(vk::CommandBuffer command) {
	if (!m_timer_initialized) {
		m_timer_initialized = true;
		m_timer_enabled     = std::getenv("KYTY_GPU_TIMING") != nullptr;
		if (m_timer_enabled) {
			vk::QueryPoolCreateInfo info {};
			info.queryType  = vk::QueryType::eTimestamp;
			info.queryCount = GpuTimerBlockQueries * GpuTimerBlockCount;
			if (m_graphics.device.createQueryPool(&info, nullptr, &m_timer_pool) !=
			    vk::Result::eSuccess) {
				m_timer_enabled = false;
			} else {
				m_timer_blocks.resize(GpuTimerBlockCount);
				for (uint32_t i = 0; i < GpuTimerBlockCount; i++) {
					m_timer_blocks[i].first = i * GpuTimerBlockQueries;
				}
				m_timer_args = std::make_unique<Buffer>(
				    m_graphics, m_command_scheduler, MemoryUsage::Download, 0,
				    vk::BufferUsageFlagBits::eTransferDst,
				    uint64_t {GpuTimerBlockQueries} * GpuTimerBlockCount * 32u);
			}
		}
	}
	m_timer_current = -1;
	if (!m_timer_enabled) {
		return;
	}
	for (size_t i = 0; i < m_timer_blocks.size(); i++) {
		auto& block = m_timer_blocks[i];
		if (block.active) {
			if (!m_command_scheduler.IsFree(block.tick)) {
				continue;
			}
			GpuTimerCollect(block);
		}
		block.active = true;
		block.used   = 0;
		block.tick   = m_command_scheduler.CurrentTick();
		block.entries.clear();
		command.resetQueryPool(m_timer_pool, block.first, GpuTimerBlockQueries);
		m_timer_current = static_cast<int>(i);
		GpuTimerMark(command, 0, 0);
		return;
	}
}

static std::array<std::atomic<uint64_t>, 64> g_recent_shaders {};
static std::atomic<uint32_t>                   g_recent_shader_index {0};

static std::atomic<bool> g_gpu_checkpoints {false};

void SetGpuCheckpointsEnabled(bool enabled) {
	g_gpu_checkpoints = enabled;
}

static std::unordered_map<uint64_t, uint32_t> g_frame_target_writes;

bool FlipTraceEnabled() {
	static const bool enabled = std::getenv("KYTY_FLIP_TRACE") != nullptr;
	return enabled;
}

void NoteColorTargetWrite(uint64_t address) {
	if (FlipTraceEnabled()) {
		g_frame_target_writes[address]++;
	}
}

struct RenderTargetRecord {
	uint32_t image  = 0;
	uint32_t format = 0;
	uint32_t width  = 0;
	uint32_t height = 0;
};
static std::unordered_map<uint64_t, RenderTargetRecord> g_render_target_images;

void NoteRenderTargetImage(uint64_t address, uint32_t image_index, uint32_t format, uint32_t width,
                           uint32_t height) {
	if (FlipTraceEnabled()) {
		g_render_target_images[address] = {image_index, format, width, height};
	}
}

void CheckSampledAlias(uint64_t address, uint32_t image_index, uint32_t format, uint32_t width,
                       uint32_t height) {
	if (!FlipTraceEnabled()) {
		return;
	}
	const auto found = g_render_target_images.find(address);
	if (found == g_render_target_images.end() || found->second.image == image_index) {
		return;
	}
	static std::atomic<uint32_t> log_count {0};
	if (log_count.fetch_add(1) < 64) {
		LOGF("RTALIAS addr=0x%010" PRIx64 " rt_image=%u rt_format=%u rt=%ux%u sampled_image=%u "
		     "sampled_format=%u sampled=%ux%u\n",
		     address, found->second.image, found->second.format, found->second.width,
		     found->second.height, image_index, format, width, height);
	}
}

std::string TakeFrameTargetStats(uint64_t presented_address) {
	std::vector<std::pair<uint64_t, uint32_t>> rows(g_frame_target_writes.begin(),
	                                                g_frame_target_writes.end());
	std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
	const auto presented = g_frame_target_writes.find(presented_address);
	std::string text     = fmt::format("presented_draws={} targets={}:",
	                                   presented == g_frame_target_writes.end() ? 0u : presented->second,
	                                   rows.size());
	for (size_t i = 0; i < rows.size() && i < 6; i++) {
		text += fmt::format(" {:x}x{}", rows[i].first, rows[i].second);
	}
	g_frame_target_writes.clear();
	return text;
}

void GpuCheckpoint(vk::CommandBuffer command, uint64_t marker) {
	if (g_gpu_checkpoints) {
		command.setCheckpointNV(reinterpret_cast<const void*>(marker));
	}
}

void PrintRecentShaders() {
	const auto end = g_recent_shader_index.load();
	std::printf("Last shaders sent to the GPU, oldest first:");
	for (uint32_t i = end > 64 ? end - 64 : 0; i < end; i++) {
		std::printf(" %016llx", static_cast<unsigned long long>(g_recent_shaders[i % 64].load()));
	}
	std::printf("\n");
	std::fflush(stdout);
}

void RenderContext::GpuTimerMark(vk::CommandBuffer command, uint64_t label, uint8_t kind,
                                 uint64_t detail) {
	if (label != 0) {
		g_recent_shaders[g_recent_shader_index.fetch_add(1) % 64].store(label | (uint64_t {kind} << 60u));
		if (g_gpu_checkpoints) {
			command.setCheckpointNV(reinterpret_cast<const void*>(label | (uint64_t {kind} << 60u)));
		}
	}
	if (m_timer_current < 0) {
		return;
	}
	auto& block = m_timer_blocks[static_cast<size_t>(m_timer_current)];
	if (block.used >= GpuTimerBlockQueries) {
		return;
	}
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_timer_pool,
	                       block.first + block.used);
	block.entries.push_back({block.first + block.used, label, detail, m_timer_pending_vaddr, kind});
	m_timer_pending_vaddr = 0;
	block.used++;
}

void RenderContext::GpuTimerCaptureArgs(vk::CommandBuffer command, vk::Buffer args,
                                        uint64_t offset, uint64_t vaddr) {
	if (m_timer_current < 0 || m_timer_args == nullptr) {
		return;
	}
	auto& block = m_timer_blocks[static_cast<size_t>(m_timer_current)];
	if (block.used >= GpuTimerBlockQueries) {
		return;
	}
	m_timer_pending_vaddr = vaddr;
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader |
	                       vk::PipelineStageFlagBits2::eAllTransfer;
	barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eCopy;
	barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	// The slot matches the timestamp the following GpuTimerMark records.
	const auto     lead = std::min<uint64_t>(offset, 8u);
	vk::BufferCopy copy {offset - lead, uint64_t {block.first + block.used} * 32u, 24u + lead};
	command.copyBuffer(args, m_timer_args->Handle(), 1, &copy);
}

void RenderContext::GpuTimerCollect(GpuTimerBlock& block) {
	block.active = false;
	if (block.used < 2) {
		return;
	}
	std::vector<uint64_t> results(static_cast<size_t>(block.used) * 2);
	const auto            result = m_graphics.device.getQueryPoolResults(
        m_timer_pool, block.first, block.used, results.size() * sizeof(uint64_t), results.data(),
        sizeof(uint64_t) * 2, vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
	if (result != vk::Result::eSuccess) {
		return;
	}
	const double period_ms = m_graphics.physical_device_properties.limits.timestampPeriod * 1e-6;
	for (uint32_t i = 1; i < block.used; i++) {
		if (results[i * 2 + 1] == 0 || results[(i - 1) * 2 + 1] == 0) {
			continue;
		}
		const auto  delta = results[i * 2] - results[(i - 1) * 2];
		const auto& entry = block.entries[i];
		auto&       stat  = m_timer_stats[entry.label];
		const auto ms = static_cast<double>(delta) * period_ms;
		stat.ms += ms;
		stat.count++;
		stat.kind = entry.kind;
		if (ms > stat.max_ms) {
			stat.max_ms = ms;
			stat.detail = entry.detail;
			if ((entry.detail >> 63u) != 0 && m_timer_args != nullptr) {
				const auto* words = reinterpret_cast<const uint32_t*>(
				    m_timer_args->Mapped().data() + uint64_t {entry.query} * 32u + 8u);
				stat.detail = (uint64_t {1} << 63u) | (uint64_t {words[0]} & 0xfffffu) |
				              ((uint64_t {words[1]} & 0xfffffu) << 20u) |
				              ((uint64_t {words[2]} & 0xfffffu) << 40u);
				stat.raw_x = words[0];
				stat.args_vaddr = entry.vaddr;
				std::copy_n(words - 2, 8, stat.around.begin());
			}
		}
		m_timer_kind_ms[entry.kind == 2 ? 2 : (entry.kind == 0 ? 0 : 1)] += static_cast<double>(delta) * period_ms;
	}
}

void RenderContext::GpuTimerReport() {
	if (!m_timer_enabled) {
		return;
	}
	const auto now       = Common::Timer::QueryPerformanceCounter();
	const auto frequency = Common::Timer::QueryPerformanceFrequency();
	if (m_timer_last_report == 0) {
		m_timer_last_report = now;
		return;
	}
	if (frequency == 0 || now - m_timer_last_report < frequency) {
		return;
	}
	m_timer_last_report = now;
	for (auto& block: m_timer_blocks) {
		if (block.active && static_cast<int>(&block - m_timer_blocks.data()) != m_timer_current &&
		    m_command_scheduler.IsFree(block.tick)) {
			GpuTimerCollect(block);
		}
	}
	std::vector<std::pair<uint64_t, GpuTimerStat>> rows(m_timer_stats.begin(), m_timer_stats.end());
	std::sort(rows.begin(), rows.end(),
	          [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
	std::printf("GPUTIME other=%.1fms dispatch=%.1fms draw=%.1fms\n", m_timer_kind_ms[0],
	            m_timer_kind_ms[1], m_timer_kind_ms[2]);
	for (size_t i = 0; i < rows.size() && i < 18; i++) {
		const auto& [label, stat] = rows[i];
		std::printf("  %s %016" PRIx64 " %7.1fms /%" PRIu64 " max=%.1fms groups=%" PRIu64 "x%" PRIu64
		            "x%" PRIu64 "%s\n",
		            stat.kind == 1 ? "cs " : (stat.kind == 2 ? "ps " : (stat.kind == 3 ? "csD" : "-- ")), label, stat.ms,
		            stat.count, stat.max_ms, stat.detail & 0xfffffu, (stat.detail >> 20u) & 0xfffffu,
		            (stat.detail >> 40u) & 0xfffffu, (stat.detail >> 63u) != 0 ? " indirect" : "");
		if ((stat.detail >> 63u) != 0) {
			std::printf("      indirect x=0x%08" PRIx32 " args=0x%010" PRIx64
			            " around(-8..+24)=%08x %08x | %08x %08x %08x | %08x %08x %08x\n",
			            stat.raw_x, stat.args_vaddr, stat.around[0], stat.around[1], stat.around[2], stat.around[3],
			            stat.around[4], stat.around[5], stat.around[6], stat.around[7]);
		}
	}
	std::fflush(stdout);
	m_timer_stats.clear();
	m_timer_kind_ms[0] = m_timer_kind_ms[1] = m_timer_kind_ms[2] = 0.0;
}

void RenderContext::RunGarbageCollector() {
	Common::WaitTrace::Scope gc_scope(Common::WaitTrace::Kind::GpuGarbage);
	GpuTimerReport();
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
	// Games rarely exit cleanly, so compiled pipelines are persisted while running.
	const auto now = std::chrono::steady_clock::now();
	if (now - m_last_pipeline_save > std::chrono::seconds(20)) {
		m_last_pipeline_save = now;
		m_pipeline_cache.SaveIfChanged();
	}
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	LOGF("\t EQEVENT: trigger id=%d context=%" PRIu32 " registrations=%zu\n", event_id, context_id,
	     registrations.size());
	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		LOGF("\t EQEVENT:   delivered eq=0x%016" PRIx64 " id=%d result=%d\n",
		     static_cast<uint64_t>(registration.eq), registration.event_id, result);
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
