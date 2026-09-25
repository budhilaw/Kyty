#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <map>
#include <span>
#include <utility>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = uint64_t {1} << (40 - CACHING_PAGEBITS);
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	// ReadMemory whose GPU waits give up after timeout_ns; false when the data is not there yet.
	[[nodiscard]] bool     ReadMemoryBounded(uint64_t vaddr, uint64_t size, uint64_t timeout_ns);
	// Copies the GPU's current bytes of a resident range into out without touching guest
	// memory or tracking state. False when the range is not GPU-owned, resident, or in time.
	// args: the page holds indirect arguments; its copy stays valid until a write overlapping it
	// or a raw-pointer pass is recorded (descriptor pages drop on any recorded write).
	[[nodiscard]] bool PeekGpuRange(uint64_t vaddr, uint64_t size, void* out, uint64_t timeout_ns,
	                                bool args = false);
	// Recorded GPU work writes [vaddr, vaddr+size): descriptor pages all drop, args pages that
	// overlap it drop.
	void InvalidatePeekCache(uint64_t vaddr, uint64_t size) {
		m_peek_generation++;
		std::erase_if(m_peek_pages, [&](const PeekPage& p) {
			return p.page < vaddr + size && vaddr < p.page + 4096;
		});
	}
	// Recorded GPU work may write anywhere (raw pointers).
	void InvalidatePeekCache() {
		m_peek_generation++;
		m_peek_pages.clear();
	}
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	void               SnapshotPagesForWrite(uint64_t vaddr, uint64_t size);
	void               WriteBackMerged(uint64_t vaddr, const uint8_t* data, uint64_t size);
	void               AppendUploadCopies(Buffer& buffer, uint64_t address, uint64_t bytes,
	                                      std::vector<vk::BufferCopy>& copies,
	                                      uint64_t& total_size) noexcept;
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	[[nodiscard]] uint64_t CpuGeneration() const { return m_memory_tracker.CpuGeneration(); }
	void TakeCpuDirtyLog(std::vector<std::pair<uint64_t, uint64_t>>& out, bool& full) {
		m_memory_tracker.TakeCpuDirtyLog(out, full);
	}
	void               RunGarbageCollector();
	// Records a staged upload of host data into a device buffer.
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	bool               WaitForReadback(uint64_t tick);

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 40, 16>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Queues backing publication; callers wait before clearing dirty pages or reusing their data.
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);

	// A download issued ahead of the CPU read that will need it. Its pages stay tracked as
	// GPU-owned until the data has landed, so an early read waits for this tick only.
	struct PendingDownload {
		uint64_t begin = 0;
		uint64_t size  = 0;
		uint64_t tick  = 0;
	};
	void RetirePendingDownloads(bool wait_all);
	bool TryWaitPendingDownload(uint64_t vaddr, uint64_t size);
	std::vector<PendingDownload> m_pending_downloads;
	std::unordered_set<uint64_t> m_downloaded_pages;
	void                         NoteDownloaded(uint64_t begin, uint64_t size);

	// Tick of the last GPU write per 64 KiB block, and of the last write too large to record
	// per block. A CPU read whose writers have all retired is served from a second queue
	// without waiting for the rest of the queued frame.
	std::unordered_map<uint64_t, uint64_t> m_gpu_write_ticks;
	uint64_t                               m_unbounded_write_tick = 0;
	vk::CommandPool                        m_readback_pool        = nullptr;
	vk::CommandBuffer                      m_readback_command     = nullptr;
	vk::Semaphore                          m_readback_semaphore   = nullptr;
	uint64_t                               m_readback_tick        = 0;
	std::unique_ptr<Buffer>                m_readback_buffer;
	void                                   InitializeReadbackQueue();
	[[nodiscard]] bool TryImmediateReadback(Buffer& buffer, uint64_t vaddr, uint64_t size);

public:
	void PrefetchReadbacks();
	// Starts an asynchronous download of GPU-modified pages around [vaddr, vaddr+size) unless one
	// is already pending; the guest copy updates when the GPU reaches it. Returns true if the
	// range is GPU-modified (the caller is reading data older than the GPU's).
	bool PrefetchRange(uint64_t vaddr, uint64_t size);
	// True when every 4 KiB page of the range has had a GPU download land in guest memory.
	[[nodiscard]] bool WasDownloaded(uint64_t vaddr, uint64_t size) const;
	// Marks [vaddr, vaddr+size) as needed by the host (indirect draw/dispatch arguments): its
	// readbacks reach guest memory even though the game never polls it.
	void RequestWriteback(uint64_t vaddr, uint64_t size);
	// A shader wrote [vaddr, vaddr+size) through a raw pointer: its pages keep a GPU baseline
	// from their next upload, so readbacks write only bytes the GPU changed.
	void NoteRawPointerWrite(uint64_t vaddr, uint64_t size);

private:
	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// Contents of GPU-owned pages at the moment the CPU first wrote to them. The upload of such
	// a page sends only the bytes the CPU changed, so GPU writes elsewhere in the page survive.
	std::unordered_map<uint64_t, std::vector<uint8_t>> m_write_snapshots;
	std::recursive_mutex                               m_snapshot_mutex;
	// What the GPU copy of each 4 KiB page held after the last upload or readback. A readback
	// writes only bytes the GPU changed relative to it, so untouched bytes keep the guest value.
	struct GpuBaseline {
		std::array<uint8_t, 4096> bytes {};
		std::array<uint64_t, 16>  valid {}; // one bit per dword
	};
	std::unordered_map<uint64_t, std::unique_ptr<GpuBaseline>> m_gpu_baselines;
	void RecordGpuBaseline(uint64_t address, const uint8_t* data, uint64_t size);
	// CPU read faults per 4 KiB page; pages the game polls (frame markers) get readbacks.
	std::unordered_map<uint64_t, uint32_t> m_read_fault_counts;
	std::unordered_set<uint64_t>           m_raw_write_pages; // 4 KiB pages
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	// Pages fetched from the GPU for the emulator's own descriptor reads (PeekGpuRange), valid
	// until GPU work is recorded again.
	struct PeekPage {
		uint64_t                  page       = 0;
		uint64_t                  generation = 0;
		bool                      args       = false;
		std::array<uint8_t, 4096> bytes {};
	};
	std::vector<PeekPage> m_peek_pages;
	uint64_t              m_peek_generation = 1;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
