#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/timer.h"
#include "graphics/host_gpu/graphicContext.h"

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
static void PrintDeviceFault(GraphicContext& graphics) {
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

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

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
