#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/threads.h"
#include "common/timer.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
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
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
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
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		if (m_command_scheduler.Active()) {
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
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
		m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
	});
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

void PrintRecentShaders() {
	const auto end = g_recent_shader_index.load();
	std::printf("Last shaders sent to the GPU, oldest first:");
	for (uint32_t i = end > 64 ? end - 64 : 0; i < end; i++) {
		std::printf(" %016llx", static_cast<unsigned long long>(g_recent_shaders[i % 64].load()));
	}
	std::printf("\n");
	std::fflush(stdout);
}

void RenderContext::GpuTimerMark(vk::CommandBuffer command, uint64_t label, uint8_t kind) {
	if (label != 0) {
		g_recent_shaders[g_recent_shader_index.fetch_add(1) % 64].store(label | (uint64_t {kind} << 60u));
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
	block.entries.push_back({block.first + block.used, label, kind});
	block.used++;
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
		stat.ms += static_cast<double>(delta) * period_ms;
		stat.count++;
		stat.kind = entry.kind;
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
		std::printf("  %s %016" PRIx64 " %7.1fms /%" PRIu64 "\n",
		            stat.kind == 1 ? "cs " : (stat.kind == 2 ? "ps " : (stat.kind == 3 ? "csD" : "-- ")), label, stat.ms,
		            stat.count);
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
