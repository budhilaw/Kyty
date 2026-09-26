#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/timer.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"

#include <cstdlib>
#include <string>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <cinttypes>
#include <vector>
#include <cstdio>
#include <windows.h>
#else
#include <immintrin.h>
#define YieldProcessor() _mm_pause()
#endif

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

// After a device loss the driver can say which GPU address faulted and how it was accessed.
static void PrintCheckpoints(GraphicContext& graphics) {
	if (!graphics.checkpoints_enabled) {
		return;
	}
	for (auto queue: {graphics.queue, graphics.readback_queue}) {
		if (!queue) {
			continue;
		}
		const auto points = queue.getCheckpointDataNV();
		std::printf("GPU checkpoints (%zu):\n", points.size());
		for (const auto& point: points) {
			const auto marker = reinterpret_cast<uint64_t>(point.pCheckpointMarker);
			std::printf("  stage=%s kind=%u hash=%016llx\n", vk::to_string(point.stage).c_str(),
			            static_cast<unsigned>(marker >> 60u),
			            static_cast<unsigned long long>(marker & ((1ull << 60u) - 1u)));
		}
	}
	std::fflush(stdout);
}

static void PrintDeviceFault(GraphicContext& graphics) {
	PrintCheckpoints(graphics);
	if (!graphics.device_fault_enabled) {
		std::printf("Device fault details unavailable (VK_EXT_device_fault not supported)\n");
		return;
	}
	vk::DeviceFaultCountsEXT counts {};
	if (graphics.device.getFaultInfoEXT(&counts, nullptr) != vk::Result::eSuccess &&
	    counts.addressInfoCount == 0 && counts.vendorInfoCount == 0) {
		std::printf("Device fault query failed\n");
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendor(counts.vendorInfoCount);
	counts.vendorBinarySize = 0;
	vk::DeviceFaultInfoEXT info {};
	info.pAddressInfos = addresses.data();
	info.pVendorInfos  = vendor.data();
	(void)graphics.device.getFaultInfoEXT(&counts, &info);
	std::printf("Device fault: %s\n", info.description.data());
	for (const auto& a: addresses) {
		std::printf("  address type=%s address=0x%llx precision=0x%llx\n",
		            vk::to_string(a.addressType).c_str(),
		            static_cast<unsigned long long>(a.reportedAddress),
		            static_cast<unsigned long long>(a.addressPrecision));
	}
	for (const auto& v: vendor) {
		std::printf("  vendor: %s code=0x%llx data=0x%llx\n", v.description.data(),
		            static_cast<unsigned long long>(v.vendorFaultCode),
		            static_cast<unsigned long long>(v.vendorFaultData));
	}
	std::fflush(stdout);
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

namespace {
thread_local GpuWaitReason g_wait_reason = GpuWaitReason::Other;

std::atomic<uint64_t> g_wait_count[2][static_cast<size_t>(GpuWaitReason::Count)] {};
std::atomic<uint64_t> g_wait_us[2][static_cast<size_t>(GpuWaitReason::Count)] {};
std::atomic<uint64_t> g_wait_last_print {0};

// Records one blocking wait (the tick was not reached when it began).
class WaitStat {
public:
	WaitStat(): m_start(Common::Timer::QueryPerformanceCounter()) {}
	~WaitStat() {
		static const bool enabled = std::getenv("KYTY_WAIT_STATS") != nullptr;
		if (!enabled) {
			return;
		}
		const auto now       = Common::Timer::QueryPerformanceCounter();
		const auto frequency = Common::Timer::QueryPerformanceFrequency();
		RecordGpuWait(g_wait_reason, (now - m_start) * 1'000'000ull / frequency);
	}
	KYTY_CLASS_NO_COPY(WaitStat);

private:
	uint64_t m_start;
};
} // namespace

void RecordGpuWait(GpuWaitReason wait_reason, uint64_t us) {
	static const bool enabled = std::getenv("KYTY_WAIT_STATS") != nullptr;
	if (!enabled) {
		return;
	}
	{
		const auto now       = Common::Timer::QueryPerformanceCounter();
		const auto frequency = Common::Timer::QueryPerformanceFrequency();
		const auto gpu       = GuestGpu::IsGpuThread() ? 1 : 0;
		const auto reason    = static_cast<size_t>(wait_reason);
		g_wait_count[gpu][reason].fetch_add(1, std::memory_order_relaxed);
		g_wait_us[gpu][reason].fetch_add(us, std::memory_order_relaxed);
		auto last = g_wait_last_print.load(std::memory_order_relaxed);
		if (last == 0) {
			g_wait_last_print.compare_exchange_strong(last, now);
			return;
		}
		if (now - last < frequency * 5 ||
		    !g_wait_last_print.compare_exchange_strong(last, now)) {
			return;
		}
		static const char* names[] = {"other", "peek", "readback", "force", "pending", "flip"};
		std::string line = "GPUWAITS 5s:";
		for (int thread = 0; thread < 2; thread++) {
			line += thread == 1 ? " | gpu:" : " game:";
			for (size_t r = 0; r < static_cast<size_t>(GpuWaitReason::Count); r++) {
				const auto n  = g_wait_count[thread][r].exchange(0);
				const auto t  = g_wait_us[thread][r].exchange(0);
				if (n != 0) {
					line += " " + std::string(names[r]) + "=" + std::to_string(n) + "/" +
					        std::to_string(t / 1000) + "ms";
				}
			}
		}
		std::printf("%s\n", line.c_str());
	}
}

void SetGpuWaitReason(GpuWaitReason reason) {
	g_wait_reason = reason;
}

GpuWaitReason GetGpuWaitReason() {
	return g_wait_reason;
}

bool MasterSemaphore::TryWait(uint64_t tick, uint64_t timeout_ns) {
	if (IsFree(tick)) {
		return true;
	}
	Refresh();
	if (IsFree(tick)) {
		return true;
	}
	WaitStat stat;
	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;
	const auto result        = m_graphics.device.waitSemaphores(&wait_info, timeout_ns);
	if (result == vk::Result::eTimeout) {
		return false;
	}
	if (result != vk::Result::eSuccess) {
		PrintRecentShaders();
		EXIT("GPU wait failed: %s (tick %" PRIu64 ")\n", vk::to_string(result).c_str(), tick);
	}
	Refresh();
	return true;
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}
	WaitStat stat;

	// Thousands of these waits happen per second, mostly for work the GPU finishes within
	// microseconds. A kernel wait costs a scheduler round trip of the better part of a
	// millisecond each time, so the counter is polled briefly before falling back to one.
	{
		const auto frequency = Common::Timer::QueryPerformanceFrequency();
		const auto deadline =
		    Common::Timer::QueryPerformanceCounter() + frequency * 2 / 1000; // 2 ms
		while (Common::Timer::QueryPerformanceCounter() < deadline) {
			for (int i = 0; i < 64; i++) {
				YieldProcessor();
			}
			Refresh();
			if (IsFree(tick)) {
				return;
			}
		}
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	// A runaway shader never signals when the driver's timeout detection is off; report the
	// last shaders instead of hanging the emulator forever.
	auto result = m_graphics.device.waitSemaphores(&wait_info, 30'000'000'000ull);
	if (result == vk::Result::eTimeout) {
		PrintRecentShaders();
		PrintDeviceFault(m_graphics);
		EXIT("GPU did not finish within 30 s (tick %" PRIu64 "); a shader is probably looping forever\n",
		     tick);
	}
	if (result != vk::Result::eSuccess) {
		PrintRecentShaders();
		PrintDeviceFault(m_graphics);
		EXIT("GPU wait failed: %s (tick %" PRIu64 ")\n", vk::to_string(result).c_str(), tick);
	}
	Refresh();
}

} // namespace Libs::Graphics
