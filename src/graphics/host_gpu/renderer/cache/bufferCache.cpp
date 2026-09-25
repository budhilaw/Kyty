#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/frameCapture.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/memory.h"

#include "common/timer.h"
#include <algorithm>
#include <cinttypes>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// A readback may complete after the game released the pages; the data is then unwanted.
static void WriteBackingIfMapped(uint64_t vaddr, const void* data, uint64_t size) {
	if (Libs::Graphics::LabelTraceEnabled()) {
		LOGF("DOWNLOAD buffer range=0x%010" PRIx64 "+0x%" PRIx64 "\n", vaddr, size);
	}
	if (!Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		LOGF("Memory: skipped readback into unmapped range addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n", vaddr, size);
	}
}


namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

bool BackingCopies() {
	static const bool enabled = std::getenv("KYTY_NO_BACKING_COPY") == nullptr;
	return enabled;
}

// Reads guest memory for an upload without touching protected pages: a range that spans
// two mappings fails the single backing read, so it is retried one page at a time.
void ReadGuestForUpload(uint64_t address, uint8_t* target, uint64_t size) {
	if (BackingCopies() && Libs::LibKernel::Memory::TryReadBacking(address, target, size)) {
		return;
	}
	constexpr uint64_t Page = 0x1000;
	for (uint64_t done = 0; done < size;) {
		const auto chunk = std::min<uint64_t>(size - done, Page - ((address + done) & (Page - 1u)));
		if (!BackingCopies() ||
		    !Libs::LibKernel::Memory::TryReadBacking(address + done, target + done, chunk)) {
			// The game may unmap a range a buffer still covers; its bytes upload as zeros.
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(address + done), &info, sizeof(info)) == 0 ||
			    info.State != MEM_COMMIT) {
				std::memset(target + done, 0, chunk);
			} else {
				std::memcpy(target + done, reinterpret_cast<const void*>(address + done), chunk);
			}
		}
		done += chunk;
	}
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

// Diagnostic trace of every cache operation touching a page that holds indirect dispatch
// arguments; enabled by the label trace.
static void TraceArgs(const char* what, uint64_t vaddr, uint64_t size, const std::string& extra) {
	if (!Libs::Graphics::LabelTraceEnabled() || size == 0) {
		return;
	}
	const auto begin = vaddr & ~uint64_t {0xfff};
	const auto end   = (vaddr + size + 0xfffu) & ~uint64_t {0xfff};
	if (uint64_t hit = 0; CoversRecentIndirectArgs(begin, end - begin, hit)) {
		LOGF("ARGSTRACE %s args=0x%010" PRIx64 " range=0x%010" PRIx64 "+0x%" PRIx64 " %s\n", what,
		     hit, vaddr, size, extra.c_str());
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, pages.first * sizeof(vk::DeviceAddress),
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(pages.first * sizeof(vk::DeviceAddress),
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	{
		// Snapshots belong to the buffer's GPU copy; a later buffer starts from a fresh upload.
		std::lock_guard lock(m_snapshot_mutex);
		const auto&     buffer = m_slot_buffers[id];
		for (auto page = buffer.CpuAddress(); page < buffer.CpuAddress() + buffer.Size();
		     page += TRACKER_PAGE_SIZE) {
			m_write_snapshots.erase(page);
		}
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		// Submitted work may still reference the buffer; let the GPU finish it first.
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 16) {
			LOGF("BufferCache: deleting buffer 0x%016" PRIx64 " while the scheduler is idle\n",
			     m_slot_buffers[id].CpuAddress());
			Common::WaitTrace::PrintHostStack("idle buffer delete");
		}
		if (m_scheduler.CurrentTick() > 0) {
			m_scheduler.Wait(m_scheduler.CurrentTick() - 1);
		}
		m_slot_buffers.erase(id);
	}
}

bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	{
		// A GPU-modified range can span several buffers; only this buffer's part is copied
		// from it, or the copy would read past its end.
		const auto begin = std::max(vaddr, buffer.CpuAddress());
		const auto end   = std::min(vaddr + size, buffer.CpuAddress() + buffer.Size());
		if (begin >= end) {
			return false;
		}
		vaddr = begin;
		size  = end - begin;
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    TraceArgs("download", start, end - start, "");
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}

	const auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	if (mapped == nullptr) {
		EXIT("BufferCache: download exceeds 64 MiB staging buffer capacity\n");
	}
	m_download_buffer.Commit();
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), m_download_buffer.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = m_download_buffer.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	m_scheduler.DeferPriorityOperation([this, mapped, offset, total_size, buffer_address,
	                                    copies = std::move(copies)] {
		m_download_buffer.Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			WriteBackMerged(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	});
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	InitializeReadbackQueue();
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

void BufferCache::InitializeReadbackQueue() {
	if (m_graphics.readback_queue == nullptr || std::getenv("KYTY_READBACK_QUEUE") == nullptr) {
		return;
	}
	vk::CommandPoolCreateInfo pool_info {};
	pool_info.queueFamilyIndex = m_graphics.queue_family;
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                  vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	if (m_graphics.device.createCommandPool(&pool_info, nullptr, &m_readback_pool) !=
	    vk::Result::eSuccess) {
		return;
	}
	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_readback_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = 1;
	if (m_graphics.device.allocateCommandBuffers(&allocate, &m_readback_command) !=
	    vk::Result::eSuccess) {
		m_readback_command = nullptr;
		return;
	}
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;
	vk::SemaphoreCreateInfo semaphore_info {};
	semaphore_info.pNext = &type_info;
	if (m_graphics.device.createSemaphore(&semaphore_info, nullptr, &m_readback_semaphore) !=
	    vk::Result::eSuccess) {
		m_readback_command = nullptr;
		return;
	}
	m_readback_buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                             vk::BufferUsageFlagBits::eTransferDst, 4 * MiB);
	SetVulkanObjectNameF(m_graphics.device, m_readback_buffer->Handle(), "Kyty.ReadbackQueue");
}

bool BufferCache::TryImmediateReadback(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	if (m_readback_command == nullptr || m_readback_buffer == nullptr) {
		return false;
	}
	constexpr uint64_t Block        = 64 * 1024;
	const auto         buffer_begin = buffer.CpuAddress();
	const auto         buffer_end   = buffer_begin + buffer.Size();
	const auto window_begin = std::max(Common::AlignDown(vaddr, Block), buffer_begin);
	const auto window_end =
	    std::min(std::max(Common::AlignUp(vaddr + size, Block), window_begin + Block), buffer_end);
	if (window_end <= window_begin) {
		return false;
	}
	const auto window_size = window_end - window_begin;
	for (const auto& pending: m_pending_downloads) {
		if (pending.begin < window_end && pending.begin + pending.size > window_begin) {
			return false;
		}
	}
	if (m_texture_cache.IsRegionGpuModified(window_begin, window_size)) {
		return false;
	}

	struct Range {
		uint64_t start;
		uint64_t end;
	};
	std::vector<Range> ranges;
	uint64_t           total = 0;
	m_gpu_modified_ranges.ForEachInRange(window_begin, window_size,
	                                     [&](uint64_t start, uint64_t end) {
		                                     ranges.push_back({start, end});
		                                     total += Common::AlignUp(end - start, 64);
	                                     });
	if (ranges.empty() || total > m_readback_buffer->Size()) {
		return false;
	}

	// Every write into the ranges must already have retired on the main queue.
	m_scheduler.GetMasterSemaphore().Refresh();
	const auto known  = m_scheduler.KnownGpuTick();
	uint64_t   writer = m_unbounded_write_tick;
	for (const auto& range: ranges) {
		for (auto block = range.start >> 16; block <= (range.end - 1) >> 16; block++) {
			const auto it = m_gpu_write_ticks.find(block);
			if (it == m_gpu_write_ticks.end()) {
				return false;
			}
			writer = std::max(writer, it->second);
		}
	}
	if (writer > known) {
		return false;
	}

	std::vector<vk::BufferCopy> copies;
	uint64_t                    offset = 0;
	for (const auto& range: ranges) {
		copies.emplace_back(range.start - buffer_begin, offset, range.end - range.start);
		offset += Common::AlignUp(range.end - range.start, 64);
	}

	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	m_readback_command.reset();
	EXIT_NOT_IMPLEMENTED(m_readback_command.begin(&begin) != vk::Result::eSuccess);
	m_readback_command.copyBuffer(buffer.Handle(), m_readback_buffer->Handle(),
	                              static_cast<uint32_t>(copies.size()), copies.data());
	vk::BufferMemoryBarrier host_barrier {};
	host_barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	host_barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host_barrier.buffer              = m_readback_buffer->Handle();
	host_barrier.offset              = 0;
	host_barrier.size                = total;
	m_readback_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                                   vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1,
	                                   &host_barrier, 0, nullptr);
	EXIT_NOT_IMPLEMENTED(m_readback_command.end() != vk::Result::eSuccess);

	// Waiting on the master timeline makes the main queue's writes visible to this copy.
	const uint64_t         signal_value  = ++m_readback_tick;
	const vk::Semaphore    wait_handle   = m_scheduler.GetMasterSemaphore().Handle();
	const uint64_t         wait_value    = writer;
	vk::PipelineStageFlags wait_stage    = vk::PipelineStageFlagBits::eTransfer;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount   = 1;
	timeline.pWaitSemaphoreValues      = &wait_value;
	timeline.signalSemaphoreValueCount = 1;
	timeline.pSignalSemaphoreValues    = &signal_value;
	vk::SubmitInfo submit {};
	submit.pNext                = &timeline;
	submit.waitSemaphoreCount   = 1;
	submit.pWaitSemaphores      = &wait_handle;
	submit.pWaitDstStageMask    = &wait_stage;
	submit.commandBufferCount   = 1;
	submit.pCommandBuffers      = &m_readback_command;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores    = &m_readback_semaphore;
	EXIT_NOT_IMPLEMENTED(m_graphics.readback_queue.submit(1, &submit, nullptr) !=
	                     vk::Result::eSuccess);

	{
		const auto frequency = Common::Timer::QueryPerformanceFrequency();
		const auto deadline  = Common::Timer::QueryPerformanceCounter() + frequency * 4 / 1000;
		uint64_t   counter   = 0;
		for (;;) {
			EXIT_NOT_IMPLEMENTED(m_graphics.device.getSemaphoreCounterValue(
			                         m_readback_semaphore, &counter) != vk::Result::eSuccess);
			if (counter >= signal_value) {
				break;
			}
			if (Common::Timer::QueryPerformanceCounter() >= deadline) {
				vk::SemaphoreWaitInfo wait_info {};
				wait_info.semaphoreCount = 1;
				wait_info.pSemaphores    = &m_readback_semaphore;
				wait_info.pValues        = &signal_value;
				EXIT_NOT_IMPLEMENTED(m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX) !=
				                     vk::Result::eSuccess);
				break;
			}
		}
	}

	m_readback_buffer->Invalidate(0, total);
	const auto* mapped = m_readback_buffer->Mapped().data();
	for (const auto& copy: copies) {
		WriteBackMerged(buffer_begin + copy.srcOffset,
		                                      mapped + copy.dstOffset, copy.size);
	}
	for (const auto& range: ranges) {
		m_gpu_modified_ranges.Subtract(range.start, range.end - range.start);
	}
	m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_size);
	return true;
}

BufferCache::~BufferCache() {
	if (m_readback_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_readback_semaphore, nullptr);
	}
	if (m_readback_pool != nullptr) {
		m_graphics.device.destroyCommandPool(m_readback_pool, nullptr);
	}
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	TraceArgs("invalidate", vaddr, size, "");
	// First CPU write to a page the GPU holds a copy of: remember its contents so the next
	// upload sends only what the CPU changed and GPU writes elsewhere in the page survive.
	if (!m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    !m_memory_tracker.IsRegionCpuModified(vaddr, size) && IsRegionRegistered(vaddr, size)) {
		SnapshotPagesForWrite(vaddr, size);
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	Common::WaitTrace::Scope readback_scope(Common::WaitTrace::Kind::GpuReadback);
	if (GuestGpu::IsGpuThread()) {
		SetGpuPhase("readback", vaddr);
	}
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		Common::WaitTrace::PrintHostStack("deferred readback");
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		// A read of data that a prefetch already has in flight only waits for that download.
		if (!m_gpu_modified_ranges.Intersects(vaddr, size) && TryWaitPendingDownload(vaddr, size)) {
			if (is_write) {
				SnapshotPagesForWrite(vaddr, size);
				m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			}
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		if (!is_write) {
			std::lock_guard lock(m_snapshot_mutex);
			if (m_read_fault_counts.size() > 65536) {
				m_read_fault_counts.clear();
			}
			// Game threads earn readbacks by polling a page. Writing back for the emulator's own
			// reads (descriptor tables, DCC codes) brought the corruption back and cost frames.
			for (auto page = vaddr & ~uint64_t {4095}; page < vaddr + size; page += 4096) {
				m_read_fault_counts[page]++;
			}
		}
		if (!is_write && TryImmediateReadback(buffer, vaddr, size)) {
			return;
		}

		// Widen nearby CPU reads so they share one GPU drain.
		constexpr uint64_t WindowSize   = 512 * 1024;
		const auto         buffer_begin = buffer.CpuAddress();
		const auto         buffer_end   = buffer_begin + buffer.Size();
		const auto window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
		const auto window_end = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);

		// Prefetches still in flight inside the window must land first: downloading over them
		// would find pages tracked as GPU-owned with no dirty bytes left to copy.
		for (auto it = m_pending_downloads.begin(); it != m_pending_downloads.end();) {
			if (it->begin < window_end && it->begin + it->size > window_begin) {
				m_scheduler.Wait(it->tick);
				m_scheduler.WaitPriorityOperations(it->tick);
				m_memory_tracker.UnmarkRegionAsGpuModified(it->begin, it->size);
				it = m_pending_downloads.erase(it);
			} else {
				++it;
			}
		}

		// The game reads a handful of small GPU-written buffers once per frame, and each read
		// that faults costs a full GPU drain. The first drain therefore also downloads every
		// other small GPU-modified range, so the rest of the frame's reads never fault.
		struct Extra {
			BufferId id;
			uint64_t begin;
			uint64_t size;
		};
		std::vector<Extra> extras;
		uint64_t           budget = 4 * MiB;
		m_gpu_modified_ranges.ForEach([&](uint64_t start, uint64_t end) {
			const auto range_size = end - start;
			if (range_size > 256 * 1024 || range_size > budget ||
			    (start >= window_begin && end <= window_end)) {
				return;
			}
			const auto* owner = m_page_table.Find(start >> PageTable::kPageBits);
			if (owner == nullptr || !*owner || &m_slot_buffers[*owner] == &buffer) {
				return;
			}
			// Adjacent buffers can merge into one range; only the owner's part is downloaded
			// here (and only that part is released from GPU ownership afterwards).
			const auto& owner_buffer = m_slot_buffers[*owner];
			const auto  owned =
			    std::min(range_size, owner_buffer.CpuAddress() + owner_buffer.Size() - start);
			extras.push_back({*owner, start, owned});
			budget -= owned;
		});

		bool downloaded = DownloadBufferMemory(buffer, window_begin, window_end - window_begin);
		for (const auto& extra: extras) {
			downloaded |= DownloadBufferMemory(m_slot_buffers[extra.id], extra.begin, extra.size);
		}
		if (downloaded) {
			const auto tick = m_scheduler.CurrentTick();
			m_scheduler.Wait(tick);
			m_scheduler.WaitPriorityOperations(tick);
			m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
			for (const auto& extra: extras) {
				m_memory_tracker.UnmarkRegionAsGpuModified(extra.begin, extra.size);
			}
		}
		if (is_write) {
			SnapshotPagesForWrite(vaddr, size);
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
	});
}

void BufferCache::RetirePendingDownloads(bool wait_all) {
	for (auto it = m_pending_downloads.begin(); it != m_pending_downloads.end();) {
		if (!wait_all && !m_scheduler.IsFree(it->tick)) {
			++it;
			continue;
		}
		m_scheduler.Wait(it->tick);
		m_scheduler.WaitPriorityOperations(it->tick);
		m_memory_tracker.UnmarkRegionAsGpuModified(it->begin, it->size);
		it = m_pending_downloads.erase(it);
	}
}

bool BufferCache::TryWaitPendingDownload(uint64_t vaddr, uint64_t size) {
	for (auto it = m_pending_downloads.begin(); it != m_pending_downloads.end(); ++it) {
		if (vaddr >= it->begin && vaddr + size <= it->begin + it->size) {
			m_scheduler.Wait(it->tick);
			m_scheduler.WaitPriorityOperations(it->tick);
			m_memory_tracker.UnmarkRegionAsGpuModified(it->begin, it->size);
			m_pending_downloads.erase(it);
			return true;
		}
	}
	return false;
}

// The game reads back a handful of small GPU-written buffers every frame, and a read that has
// to fetch its data drains the GPU. Issued as the frame ends, without waiting, the downloads
// have landed by the time those reads happen, so they no longer fault at all.
void BufferCache::PrefetchReadbacks() {
	RetirePendingDownloads(false);

	struct Range {
		BufferId id;
		uint64_t begin;
		uint64_t size;
	};
	std::vector<Range> ranges;
	uint64_t           budget = 4 * MiB;
	m_gpu_modified_ranges.ForEach([&](uint64_t start, uint64_t end) {
		auto range_size = end - start;
		if (range_size > 256 * 1024 || range_size > budget) {
			return;
		}
		const auto* owner = m_page_table.Find(start >> PageTable::kPageBits);
		if (owner == nullptr || !*owner) {
			return;
		}
		// Adjacent buffers can merge into one range; only the owner's part is downloaded here.
		const auto& buffer = m_slot_buffers[*owner];
		range_size = std::min(range_size, buffer.CpuAddress() + buffer.Size() - start);
		ranges.push_back({*owner, start, range_size});
		budget -= range_size;
	});
	if (ranges.empty()) {
		return;
	}

	const auto tick = m_scheduler.CurrentTick();
	for (const auto& range: ranges) {
		if (DownloadBufferMemory(m_slot_buffers[range.id], range.begin, range.size)) {
			m_pending_downloads.push_back({range.begin, range.size, tick});
		}
	}
	m_scheduler.Flush();
}

void BufferCache::RequestWriteback(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_snapshot_mutex);
	for (auto page = vaddr & ~uint64_t {4095}; page < vaddr + size; page += 4096) {
		auto& count = m_read_fault_counts[page];
		count       = std::max(count, 4u);
	}
}

bool BufferCache::PrefetchRange(uint64_t vaddr, uint64_t size) {
	if (!m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	constexpr uint64_t Page  = 4096;
	const auto         begin = vaddr & ~(Page - 1u);
	const auto         end   = (vaddr + size + Page - 1u) & ~(Page - 1u);
	for (const auto& pending: m_pending_downloads) {
		if (pending.begin < end && pending.begin + pending.size > begin) {
			return true;
		}
	}
	const auto* owner = m_page_table.Find(begin >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) {
		return true;
	}
	auto&      buffer = m_slot_buffers[*owner];
	const auto from   = std::max(begin, buffer.CpuAddress());
	const auto to     = std::min(end, buffer.CpuAddress() + buffer.Size());
	if (from >= to) {
		return true;
	}
	{
		// These pages are wanted on the host: let their readback reach guest memory.
		std::lock_guard lock(m_snapshot_mutex);
		for (auto page = from & ~(Page - 1u); page < to; page += Page) {
			m_read_fault_counts[page] += 4;
		}
	}
	if (DownloadBufferMemory(buffer, from, to - from)) {
		m_pending_downloads.push_back({from, to - from, m_scheduler.CurrentTick()});
	}
	return true;
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, PageTable::kAddressSpaceSize - end);
			}
			if (expands_right) {
				const auto minimum = CACHING_PAGESIZE * 2;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	TraceArgs("join", overlap.CpuAddress(), overlap.Size(),
	          fmt::format("into buffer=0x{:x}+0x{:x}", new_buffer.CpuAddress(), new_buffer.Size()));
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	TraceArgs("create", overlap.begin, overlap.end - overlap.begin, "");
	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

void BufferCache::RecordGpuBaseline(uint64_t address, const uint8_t* data, uint64_t size) {
	constexpr uint64_t Page = 4096;
	std::lock_guard    lock(m_snapshot_mutex);
	if (m_gpu_baselines.size() > 65536) {
		// Over 256 MiB of baselines: start over; pages without one use the older merge.
		m_gpu_baselines.clear();
	}
	for (uint64_t done = 0; done < size;) {
		const auto at    = address + done;
		const auto page  = at & ~(Page - 1u);
		const auto begin = at - page;
		const auto chunk = std::min<uint64_t>(size - done, Page - begin);
		// Only pages the game polls are ever written back; others need no baseline. Copying
		// every uploaded byte cost ~14% of the GPU thread on the Language screen.
		static const bool all_pages = std::getenv("KYTY_READBACK") != nullptr;
		if (!all_pages) {
			const auto polled = m_read_fault_counts.find(page);
			if (polled == m_read_fault_counts.end() || polled->second < 4u) {
				done += chunk;
				continue;
			}
		}
		auto&      entry = m_gpu_baselines[page];
		if (!entry) {
			entry = std::make_unique<GpuBaseline>();
		}
		std::memcpy(entry->bytes.data() + begin, data + done, chunk);
		// Only whole dwords become known; a partial dword stays unknown.
		for (auto dword = (begin + 3u) / 4u; dword < (begin + chunk) / 4u; dword++) {
			entry->valid[dword / 64u] |= uint64_t {1} << (dword % 64u);
		}
		done += chunk;
	}
}

void BufferCache::WriteBackMerged(uint64_t vaddr, const uint8_t* data, uint64_t size) {
	// Whole-range readbacks wrote stale GPU bytes over newer guest data (writes the tracker never
	// saw) and corrupted game objects in about half of all PPSA05684 runs. Readbacks now write
	// only bytes the GPU changed since the last upload (m_gpu_baselines). The game still needs
	// them: its frame markers are GPU-written. KYTY_NO_READBACK=1 drops readbacks entirely.
	// Default: only pages the game polls are written back (frame markers the CPU spins on);
	// KYTY_READBACK=1 writes back every GPU-changed byte, KYTY_NO_READBACK=1 none at all.
	static const bool no_readback  = std::getenv("KYTY_NO_READBACK") != nullptr;
	static const bool all_readback = std::getenv("KYTY_READBACK") != nullptr;
	if (no_readback) {
		return;
	}
	// GPU writes through raw pointers mark whole pages only after the dispatch; bytes the CPU
	// changed since its first write to such a page are newer than the GPU copy and are kept.
	constexpr uint64_t   Page = TRACKER_PAGE_SIZE;
	std::vector<uint8_t> merged;
	// 4 KiB pages the GPU did not change at all: they are not written back.
	constexpr uint64_t BasePage   = 4096;
	const auto         first_page = vaddr & ~(BasePage - 1u);
	std::vector<bool>  untouched((vaddr + size - first_page + BasePage - 1u) / BasePage, false);
	{
		std::lock_guard lock(m_snapshot_mutex);
		// Bytes with a GPU baseline: the GPU changed a byte only if it differs from what was
		// uploaded, so every other byte keeps the guest's current value.
		for (auto page = first_page; page < vaddr + size; page += BasePage) {
			if (!all_readback) {
				const auto polled = m_read_fault_counts.find(page);
				if (polled == m_read_fault_counts.end() || polled->second < 4u) {
					untouched[(page - first_page) / BasePage] = true;
					continue;
				}
			}
			const auto found = m_gpu_baselines.find(page);
			if (found == m_gpu_baselines.end()) {
				continue;
			}
			auto& baseline = *found->second;
			{
				const auto begin    = std::max(page, vaddr);
				const auto length   = std::min(page + BasePage, vaddr + size) - begin;
				bool       all_valid = true;
				for (const auto word: baseline.valid) {
					all_valid = all_valid && word == ~uint64_t {0};
				}
				if (all_valid && std::memcmp(data + (begin - vaddr),
				                             baseline.bytes.data() + (begin - page), length) == 0) {
					untouched[(page - first_page) / BasePage] = true;
					continue;
				}
			}
			std::array<uint8_t, 4096> current {};
			if (!Libs::LibKernel::Memory::TryReadBacking(page, current.data(), BasePage)) {
				continue;
			}
			const auto begin = std::max(page, vaddr);
			const auto end   = std::min(page + BasePage, vaddr + size);
			const auto* gpu_page = data + (begin - vaddr);
			const auto  length   = end - begin;
			// Common case: the guest already holds exactly what the GPU has.
			if (std::memcmp(gpu_page, current.data() + (begin - page), length) != 0) {
				for (auto address = begin; address < end; address++) {
					const auto offset = address - page;
					const auto dword  = offset / 4u;
					if ((baseline.valid[dword / 64u] & (uint64_t {1} << (dword % 64u))) == 0) {
						continue;
					}
					const auto gpu = data[address - vaddr];
					if (gpu == baseline.bytes[offset] && gpu != current[offset]) {
						if (merged.empty()) {
							merged.assign(data, data + size);
						}
						merged[address - vaddr] = current[offset];
					}
				}
			}
			std::memcpy(baseline.bytes.data() + (begin - page), gpu_page, length);
		}
		for (auto page = vaddr & ~(Page - 1u); page < vaddr + size; page += Page) {
			const auto found = m_write_snapshots.find(page);
			if (found == m_write_snapshots.end()) {
				continue;
			}
			std::vector<uint8_t> current(Page);
			if (!Libs::LibKernel::Memory::TryReadBacking(page, current.data(), Page)) {
				continue;
			}
			const auto begin = std::max(page, vaddr);
			const auto end   = std::min(page + Page, vaddr + size);
			for (auto address = begin; address < end; address++) {
				const auto offset = address - page;
				if (current[offset] != found->second[offset]) {
					if (merged.empty()) {
						merged.assign(data, data + size);
					}
					merged[address - vaddr] = current[offset];
				}
			}
		}
	}
	const auto* source = merged.empty() ? data : merged.data();
	// Write back only runs of pages the GPU may have changed.
	for (uint64_t index = 0; index < untouched.size();) {
		if (untouched[index]) {
			index++;
			continue;
		}
		auto run_end = index;
		while (run_end < untouched.size() && !untouched[run_end]) {
			run_end++;
		}
		const auto begin = std::max(first_page + index * BasePage, vaddr);
		const auto end   = std::min(first_page + run_end * BasePage, vaddr + size);
		WriteBackingIfMapped(begin, source + (begin - vaddr), end - begin);
		index = run_end;
	}
}

void BufferCache::SnapshotPagesForWrite(uint64_t vaddr, uint64_t size) {
	constexpr uint64_t Page = TRACKER_PAGE_SIZE;
	std::lock_guard    lock(m_snapshot_mutex);
	if (m_write_snapshots.size() > 4096) {
		m_write_snapshots.clear();
	}
	for (auto page = vaddr & ~(Page - 1u); page < vaddr + size; page += Page) {
		// The page was just downloaded, so its current contents are the GPU copy; an older
		// snapshot would hide CPU changes made since then.
		m_write_snapshots.erase(page);
		if (!IsRegionRegistered(page, Page)) {
			continue;
		}
		std::vector<uint8_t> data(Page);
		if (Libs::LibKernel::Memory::TryReadBacking(page, data.data(), Page)) {
			m_write_snapshots.emplace(page, std::move(data));
		}
	}
}

void BufferCache::AppendUploadCopies(Buffer& buffer, uint64_t address, uint64_t bytes,
                                     std::vector<vk::BufferCopy>& copies,
                                     uint64_t& total_size) noexcept {
	constexpr uint64_t Page = TRACKER_PAGE_SIZE;
	std::lock_guard    lock(m_snapshot_mutex);
	const auto         push = [&](uint64_t begin, uint64_t length) {
		if (length != 0) {
			copies.emplace_back(total_size, buffer.Offset(begin), length);
			total_size += length;
		}
	};
	uint64_t run_begin = address;
	for (auto page = address & ~(Page - 1u); page < address + bytes; page += Page) {
		const auto found = m_write_snapshots.find(page);
		if (found == m_write_snapshots.end()) {
			continue;
		}
		const auto           begin = std::max(page, address);
		const auto           end   = std::min(page + Page, address + bytes);
		std::vector<uint8_t> current(Page);
		const bool readable = Libs::LibKernel::Memory::TryReadBacking(page, current.data(), Page);
		const auto snapshot = std::move(found->second);
		m_write_snapshots.erase(found);
		if (!readable) {
			continue;
		}
		// Everything before this page is uploaded whole; inside it only changed dwords go.
		push(run_begin, begin - run_begin);
		for (uint64_t offset = begin - page; offset < end - page;) {
			if (std::memcmp(current.data() + offset, snapshot.data() + offset, 4) == 0) {
				offset += 4;
				continue;
			}
			auto changed_end = offset + 4;
			while (changed_end < end - page &&
			       std::memcmp(current.data() + changed_end, snapshot.data() + changed_end, 4) != 0) {
				changed_end += 4;
			}
			push(page + offset, changed_end - offset);
			offset = changed_end;
		}
		run_begin = end;
	}
	push(run_begin, address + bytes - run_begin);
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    TraceArgs("upload", address, bytes,
		              fmt::format("into buffer=0x{:x}+0x{:x}", buffer.CpuAddress(), buffer.Size()));
		    AppendUploadCopies(buffer, address, bytes, copies, total_size);
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		TraceArgs("sync-from-image", vaddr, size, "");
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			ReadGuestForUpload(address, mapped + copy.srcOffset, copy.size);
			RecordGpuBaseline(address, mapped + copy.srcOffset, copy.size);
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		auto*      target  = temporary->Mapped().data() + copy.srcOffset;
		ReadGuestForUpload(address, target, copy.size);
		RecordGpuBaseline(address, target, copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer range addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}

	TraceArgs("obtain", vaddr, size,
	          fmt::format("written={} texel={} cpu_dirty={} gpu_dirty={}", is_written, is_texel_buffer,
	                      m_memory_tracker.IsRegionCpuModified(vaddr, size),
	                      m_memory_tracker.IsRegionGpuModified(vaddr, size)));
	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr && Libs::LibKernel::Memory::TryReadBacking(vaddr, mapped, size)) {
			m_stream_buffer.Commit();
			TraceArgs("obtain-stream", vaddr, size, "");
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	TraceArgs("obtain-buffer", vaddr, size,
	          fmt::format("buffer=0x{:x}+0x{:x}", buffer.CpuAddress(), buffer.Size()));
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		m_gpu_modified_ranges.Add(vaddr, size);
		const auto tick = m_scheduler.CurrentTick();
		if (size <= 4 * MiB) {
			for (auto block = vaddr >> 16; block <= (vaddr + size - 1) >> 16; block++) {
				m_gpu_write_ticks[block] = tick;
			}
		} else {
			m_unbounded_write_tick = tick;
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr) {
		EXIT("BufferCache: failed to map staging memory for guest image backing\n");
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(vaddr, staging, size) &&
	    !Libs::LibKernel::Memory::TryReadPrtBacking(vaddr, staging, size)) {
		// The descriptor points at memory with no backing (a stale or corrupt texture); upload
		// zeros instead of stopping the emulator.
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 16) {
			LOGF("BufferCache: image backing 0x%016" PRIx64 "+0x%" PRIx64 " unreadable, zero-filled\n",
			     vaddr, size);
		}
		std::memset(staging, 0, size);
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	FrameCapture::NoteBuffer("fill", vaddr, size, true);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (dst_memory) {
		FrameCapture::NoteBuffer("copy-dst", dst_vaddr, size, true);
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		// The source is read through the backing alias: its guest page can still be
		// read-protected, and faulting on it would drain the GPU for a CPU-side copy.
		auto* destination = reinterpret_cast<void*>(dst_vaddr);
		if (!BackingCopies() || !Libs::LibKernel::Memory::TryReadBacking(src_vaddr, destination, size)) {
			std::memcpy(destination, reinterpret_cast<const void*>(src_vaddr), size);
		}
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
		if (uint64_t hit = dst_vaddr; src_gds && size == 4u && Libs::Graphics::LabelTraceEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1) < (1u << 20u)) {
				uint32_t    words[3] {};
				const bool  ok = Libs::LibKernel::Memory::TryReadBacking(hit, words, sizeof(words));
				LOGF("SHADERDUMP dma to args=0x%016" PRIx64 " dst=0x%016" PRIx64 " size=%" PRIu64
				     " src_gds=%d src=0x%" PRIx64 " host=%08x %08x %08x read=%d cpu_dirty=%d gpu_dirty=%d"
				     " gpu_bytes=%d\n",
				     hit, dst_vaddr, size, src_gds ? 1 : 0, src_vaddr, words[0], words[1], words[2],
				     ok ? 1 : 0, IsRegionCpuModified(hit, 12) ? 1 : 0, IsRegionGpuModified(hit, 12) ? 1 : 0,
				     m_gpu_modified_ranges.Intersects(hit, 12) ? 1 : 0);
			}
		}
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	// Ownership validation below expects no download to be in flight.
	RetirePendingDownloads(true);

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_IF(!DownloadBufferMemory(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			TraceArgs("gc-delete", buffer.CpuAddress(), buffer.Size(), "");
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
