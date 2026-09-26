#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>

namespace Libs::Graphics {

struct GraphicContext;

// Why a thread blocks on the GPU: counted per reason, logged as GPUWAITS every 5 s.
enum class GpuWaitReason : uint32_t { Other, Peek, Readback, ForceReadback, Pending, Flip, Count };
void                 SetGpuWaitReason(GpuWaitReason reason);
[[nodiscard]] GpuWaitReason GetGpuWaitReason();
// KYTY_WAIT_STATS=1: one blocking wait of us microseconds, by the calling thread.
void RecordGpuWait(GpuWaitReason reason, uint64_t us);
struct GpuWaitScope {
	explicit GpuWaitScope(GpuWaitReason reason): m_saved(GetGpuWaitReason()) { SetGpuWaitReason(reason); }
	~GpuWaitScope() { SetGpuWaitReason(m_saved); }
	KYTY_CLASS_NO_COPY(GpuWaitScope);

private:
	GpuWaitReason m_saved;
};

class MasterSemaphore {
public:
	explicit MasterSemaphore(GraphicContext& graphics);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		return m_current_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		return m_gpu_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] bool     IsFree(uint64_t tick) const noexcept { return KnownGpuTick() >= tick; }
	[[nodiscard]] uint64_t NextTick() noexcept {
		return m_current_tick.fetch_add(1, std::memory_order_release);
	}
	[[nodiscard]] vk::Semaphore Handle() const noexcept { return m_semaphore; }

	void Refresh();
	void Wait(uint64_t tick);
	// False when the tick is still pending after timeout_ns.
	[[nodiscard]] bool TryWait(uint64_t tick, uint64_t timeout_ns);

private:
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
