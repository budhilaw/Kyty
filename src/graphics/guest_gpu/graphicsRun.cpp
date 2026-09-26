#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/timer.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/frameCapture.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <chrono>
#include <map>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <unordered_map>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <thread>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_mutex_owned   = false;
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;

struct DrawIndirectArgs {
	uint32_t vertex_count_per_instance;
	uint32_t instance_count;
	uint32_t start_vertex_location;
	uint32_t start_instance_location;
};

struct DrawIndexedIndirectArgs {
	uint32_t index_count_per_instance;
	uint32_t instance_count;
	uint32_t start_index_location;
	uint32_t base_vertex_location;
	uint32_t start_instance_location;
};

class GpuMutexLock final {
public:
	explicit GpuMutexLock(Common::Mutex& mutex): m_mutex(mutex) {
		if (g_gpu_mutex_owned) {
			EXIT("recursive GPU mutex acquisition\n");
		}
		g_gpu_mutex_owned = true;
		m_mutex.Lock();
	}
	~GpuMutexLock() {
		if (!g_gpu_mutex_owned) {
			EXIT("invalid GPU mutex release\n");
		}
		m_mutex.Unlock();
		g_gpu_mutex_owned = false;
	}

private:
	Common::Mutex& m_mutex;
};

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

// LOGF is silent by default, which discards the packet dump a hard EXIT depends on.
static void Pm4FatalPrintf(const char* format, ...) {
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fflush(stderr);
}

#define KYTY_PM4_FATAL_LOG(...)                                                                    \
	do {                                                                                           \
		LOGF(__VA_ARGS__);                                                                         \
		Pm4FatalPrintf(__VA_ARGS__);                                                               \
	} while (false)

// A lost packet boundary shows up well before the dword that fails to decode, so the trailing
// window has to be wide enough to walk back to the last valid header.
constexpr uint32_t kPm4DumpBefore = 64;
constexpr uint32_t kPm4DumpAfter  = 16;

// Fence addresses the command processors are currently blocked on, so the queue scheduler can
// look for their writers inside submissions that have not been parsed yet.
static std::array<std::atomic<uint64_t>, 8> g_awaited_fences {};

static bool IsAwaitedFence(uint64_t address) {
	for (const auto& slot: g_awaited_fences) {
		if (slot.load() == address) {
			return true;
		}
	}
	return false;
}

static void RememberAwaitedFence(uint64_t address) {
	for (auto& slot: g_awaited_fences) {
		uint64_t expected = 0;
		if (slot.load() == address) {
			return;
		}
		if (slot.compare_exchange_strong(expected, address)) {
			return;
		}
	}
}

// Recent packets, so a truncation report can show whether the preceding lengths tiled exactly up
// to it or whether the parser had already lost the boundary.
struct Pm4PacketTrace {
	uint32_t offset;
	uint32_t cmd_id;
	uint32_t consumed;
};

static thread_local std::array<Pm4PacketTrace, 12> g_pm4_trace {};
static thread_local uint32_t                       g_pm4_trace_pos = 0;

static void Pm4TraceRecord(uint32_t offset, uint32_t cmd_id, uint32_t consumed) {
	g_pm4_trace[g_pm4_trace_pos % g_pm4_trace.size()] = {offset, cmd_id, consumed};
	g_pm4_trace_pos++;
}

static void Pm4TraceDump() {
	const auto count =
	    std::min<uint32_t>(g_pm4_trace_pos, static_cast<uint32_t>(g_pm4_trace.size()));
	for (uint32_t i = 0; i < count; i++) {
		const auto& e = g_pm4_trace[(g_pm4_trace_pos - count + i) % g_pm4_trace.size()];
		LOGF("\t   prev[%" PRIu32 "] offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32
		     " consumed=%" PRIu32 " ends=0x%05" PRIx32 "\n",
		     i, e.offset, e.cmd_id, e.consumed, e.offset + e.consumed);
	}
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands,
                      std::vector<uint32_t> owned_commands, const uint32_t* guest_origin,
                      std::vector<uint64_t> reserved_flips) {
	if (draw_commands.empty()) {
		return;
	}
	GpuMutexLock lock(m_submission_mutex);
	Submission   submission;
	submission.reserved_flips    = std::move(reserved_flips);
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	submission.owned_commands    = std::move(owned_commands);
	submission.guest_origin      = guest_origin;
	submission.reset_processor   = m_graphics_done;
	m_graphics_done              = false;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands,
                             std::vector<uint32_t> owned_commands, const uint32_t* guest_origin) {
	EXIT_IF(commands.empty());
	GpuMutexLock lock(m_submission_mutex);

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type           = SubmissionType::Compute;
	submission.queue_id       = 1 + compute_queue;
	submission.commands       = commands;
	submission.owned_commands = std::move(owned_commands);
	submission.guest_origin   = guest_origin;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	GpuMutexLock lock(m_submission_mutex);
	Submission   submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.reset_processor = m_graphics_done;
	submission.flip_request_id = request_id;
	m_graphics_done            = false;
	Enqueue(std::move(submission));
}

void GuestGpu::Done() {
	GpuMutexLock lock(m_submission_mutex);
	if (!IsGpuThread()) {
		WaitForIdle();
	}
	m_graphics_done = true;
	m_done_num++;
}

// A suspend point only marks the frame boundary. Draining the GPU here deadlocks whenever a
// queue is waiting on a fence that the calling thread has still to submit.
void GuestGpu::SuspendPoint() {
	GpuMutexLock lock(m_submission_mutex);
	m_graphics_done = true;
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

void CommandProcessor::BufferInit() {
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	GetScheduler().Flush();
}

void CommandProcessor::BufferFlushIfBusy() {
	static const uint32_t threshold = [] {
		const char* text = std::getenv("KYTY_LABEL_FLUSH_WORK");
		return text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 0)) : 16u;
	}();
	if (GetScheduler().WorkSinceSubmit() >= threshold) {
		BufferFlush();
	}
}

void CommandProcessor::BufferFlushAndWait() {
	GetScheduler().FlushAndWait();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4("WaitCe");
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4("WaitDeDiff");
	}
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4("WaitForRewind");
	}
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	memcpy(dst, m_const_ram + offset / 4, static_cast<size_t>(dw_num) * 4);
}

// A hardware ring polls its wait address continuously, so it observes a value even if a later
// packet overwrites it. A queue here only polls while it is scheduled, so every GPU-side write to
// guest memory is recorded and a suspended wait can still be released by a pulse it missed.
namespace {

struct GuestGpuWrite {
	uint64_t address = 0;
	uint64_t value   = 0;
	uint64_t seq     = 0;
};

constexpr size_t              kGuestWriteHistory = 512;
std::array<GuestGpuWrite, kGuestWriteHistory> g_guest_writes {};
std::atomic<uint64_t>         g_guest_write_seq {0};
Common::Mutex                 g_guest_write_mutex;
// Last value the GPU wrote to each fence address, kept beyond the short history so a queue
// that fell behind can tell its fence was released before the guest recycled the memory.
std::unordered_map<uint64_t, uint64_t> g_last_gpu_writes;

} // namespace

// Addresses that a submitted command stream promises to write, recorded before execution. Lets a
// stall report distinguish "the guest never submitted the writer" from "we lost the write".
namespace {
std::array<std::atomic<uint64_t>, 1024>     g_promised_fences {};
std::array<std::atomic<const char*>, 1024>  g_promised_origins {};
std::array<std::atomic<const char*>, 1024>  g_promised_packets {};
std::atomic<uint32_t>                       g_promised_pos {0};
} // namespace

void NotePromisedFenceWrite(uint64_t address, const char* origin, const char* packet) {
	if (address == 0) {
		return;
	}
	const auto slot  = g_promised_pos.fetch_add(1, std::memory_order_acq_rel);
	const auto index = slot % g_promised_fences.size();
	g_promised_origins[index].store(origin, std::memory_order_release);
	g_promised_packets[index].store(packet, std::memory_order_release);
	g_promised_fences[index].store(address, std::memory_order_release);
}

// Reports the most recent stream that promised the address, so a stall can name the packet and
// the queue kind that should have released it.
static bool WasFencePromised(uint64_t address, const char** origin, const char** packet) {
	bool found = false;
	for (size_t index = 0; index < g_promised_fences.size(); index++) {
		if (g_promised_fences[index].load(std::memory_order_acquire) != address) {
			continue;
		}
		found   = true;
		*origin = g_promised_origins[index].load(std::memory_order_acquire);
		*packet = g_promised_packets[index].load(std::memory_order_acquire);
	}
	return found;
}

uint64_t CurrentGuestGpuWriteSeq() {
	return g_guest_write_seq.load(std::memory_order_acquire);
}

// Set KYTY_WATCH_ADDR to a guest address to find which GPU write path touches it.
void CheckGuestWatch(uint64_t address, uint64_t size, const char* who) {
	static const uint64_t watch = [] {
		const char* text = std::getenv("KYTY_WATCH_ADDR");
		return text != nullptr ? std::strtoull(text, nullptr, 0) : 0;
	}();
	if (watch == 0 || address > watch || watch >= address + size) {
		return;
	}
	LOGF("\t WATCHHIT: %s wrote 0x%016" PRIx64 "..0x%016" PRIx64 " covering 0x%016" PRIx64 "\n", who,
	     address, address + size, watch);
}

namespace {
struct PendingGuestWrite {
	uint64_t id      = 0;
	uint64_t address = 0;
	uint64_t value   = 0;
	uint32_t width   = 0;
};
constexpr size_t              kPendingGuestWrites = 4096;
std::array<PendingGuestWrite, kPendingGuestWrites> g_pending_writes {};
std::atomic<uint64_t>         g_pending_write_id {0};
Common::Mutex                 g_pending_write_mutex;
} // namespace

uint64_t NotePendingGuestGpuWrite(uint64_t address, uint64_t value, uint32_t width) {
	if (address == 0) {
		return 0;
	}
	const auto        id = g_pending_write_id.fetch_add(1, std::memory_order_acq_rel) + 1;
	{
		Common::LockGuard lock(g_pending_write_mutex);
		g_pending_writes[id % kPendingGuestWrites] = {id, address, value, width};
	}
	// A queue blocked on this address re-examines its wait only when the write sequence
	// moves, so the promise counts as a write for scheduling purposes.
	g_guest_write_seq.fetch_add(1, std::memory_order_acq_rel);
	return id;
}

void ResolvePendingGuestGpuWrite(uint64_t id) {
	if (id == 0) {
		return;
	}
	Common::LockGuard lock(g_pending_write_mutex);
	auto&             slot = g_pending_writes[id % kPendingGuestWrites];
	if (slot.id == id) {
		slot = {};
	}
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

bool PendingGuestGpuWriteSatisfies(uint64_t address, uint64_t ref, uint64_t mask, uint32_t func,
                                   uint64_t& value) {
	Common::LockGuard lock(g_pending_write_mutex);
	// The newest promise wins: a later write in the stream overrides an earlier one.
	const PendingGuestWrite* best = nullptr;
	for (const auto& slot: g_pending_writes) {
		if (slot.id != 0 && slot.address == address && (best == nullptr || slot.id > best->id)) {
			best = &slot;
		}
	}
	if (best == nullptr) {
		return false;
	}
	value = best->width == sizeof(uint32_t) ? (best->value & 0xffffffffu) : best->value;
	return TestWaitRegMemValue(value, ref, mask, func);
}

namespace {
std::atomic<const char*> g_gpu_phase {"idle"};
std::atomic<uint64_t>    g_gpu_phase_detail {0};
std::atomic<uint64_t>    g_gpu_heartbeat {0};
std::atomic<uint32_t>    g_gpu_os_thread_id {0};
} // namespace

uint32_t GpuOsThreadId() {
	return g_gpu_os_thread_id.load(std::memory_order_relaxed);
}

void SetGpuPhase(const char* phase, uint64_t detail) {
	g_gpu_phase.store(phase, std::memory_order_relaxed);
	g_gpu_phase_detail.store(detail, std::memory_order_relaxed);
	g_gpu_heartbeat.fetch_add(1, std::memory_order_relaxed);
}

const char* GpuPhase() {
	return g_gpu_phase.load(std::memory_order_relaxed);
}

uint64_t GpuPhaseDetail() {
	return g_gpu_phase_detail.load(std::memory_order_relaxed);
}

uint64_t GpuHeartbeat() {
	return g_gpu_heartbeat.load(std::memory_order_relaxed);
}

void NoteGuestGpuWriteValue(uint64_t address, uint64_t value) {
	if (address == 0) {
		return;
	}
	{
		Common::LockGuard lock(g_guest_write_mutex);
		if (g_last_gpu_writes.size() > 65536) {
			g_last_gpu_writes.clear();
		}
		g_last_gpu_writes[address] = value;
	}
	NoteGuestGpuWrite(address);
}

void NoteGuestGpuWrite(uint64_t address) {
	if (address == 0) {
		return;
	}
	// The value is only diagnostic. Reading GPU-written memory through the guest mapping
	// would fault and drain the GPU, so such a write is recorded without its value.
	uint64_t   value = UINT64_MAX;
	const auto width = (address & 7u) == 0 ? sizeof(uint64_t) : sizeof(uint32_t);
	uint64_t   read  = 0;
	if (Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, &read, width)) {
		value = read;
	}
	Common::LockGuard lock(g_guest_write_mutex);
	const auto        seq       = g_guest_write_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
	g_guest_writes[seq % kGuestWriteHistory] = {address, value, seq};
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

void DescribeGuestGpuWrites(uint64_t address, uint64_t baseline_seq) {
	Common::LockGuard lock(g_guest_write_mutex);
	uint32_t          found = 0;
	for (const auto& record: g_guest_writes) {
		if (record.address == address && record.seq != 0) {
			LOGF("\t STALLDUMP:     written value=0x%016" PRIx64 " seq=%" PRIu64 " baseline=%" PRIu64
			     " %s\n",
			     record.value, record.seq, baseline_seq,
			     record.seq > baseline_seq ? "(after submit)" : "(before submit)");
			found++;
		}
	}
	if (found == 0) {
		LOGF("\t STALLDUMP:     never written by the GPU (history seq now %" PRIu64
		     ", baseline %" PRIu64 ")\n",
		     g_guest_write_seq.load(), baseline_seq);
	}
}

static bool GuestGpuWriteSatisfied(uint64_t address, uint64_t since_seq, uint64_t ref,
                                   uint64_t mask, uint32_t func) {
	Common::LockGuard lock(g_guest_write_mutex);
	for (const auto& record: g_guest_writes) {
		if (record.seq > since_seq && record.address == address &&
		    TestWaitRegMemValue(record.value, ref, mask, func)) {
			return true;
		}
	}
	return false;
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

bool LabelTraceEnabled() {
	// KYTY_LABEL_TRACE=1 logs every GPU label write and wait with its queue, for sync debugging.
	static const bool enabled = std::getenv("KYTY_LABEL_TRACE") != nullptr;
	return enabled;
}

// Indirect arguments can come from a damaged packet; reading them must not fault the emulator.
static bool GuestRangeCommitted(const void* address, size_t size) {
	auto*       cursor = static_cast<const uint8_t*>(address);
	const auto* end    = cursor + size;
	while (cursor < end) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(cursor, &info, sizeof(info)) == 0 || info.State != MEM_COMMIT) {
			return false;
		}
		cursor = static_cast<const uint8_t*>(info.BaseAddress) + info.RegionSize;
	}
	return true;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	// Bits above the operation select cache and engine behavior the host does not model.
	wait_op &= 1u;
	(void)wait_op;

	(void)poll;
	const auto wait_address = reinterpret_cast<uint64_t>(addr);
	SetGpuPhase("wait_reg_mem", wait_address);
	// Read through the backing alias: a direct read of a GPU-owned page would fault and
	// download the GPU copy over the value a fence write placed here at parse time.
	T current = 0;
	if (!Libs::LibKernel::Memory::TryReadBacking(wait_address, &current, sizeof(T))) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(addr, &info, sizeof(info)) == 0 || info.State != MEM_COMMIT) {
			// A wait on unmapped memory comes from a damaged packet; nothing can release it.
			LOGF("\t WAITREGMEM: unmapped address 0x%016" PRIx64 " skipped\n", wait_address);
			g_current_execution->ClearWait();
			return;
		}
		current = *addr;
	}
	if (g_current_execution->TakeForcedPass(wait_address)) {
		LOGF("\t DEADLOCK BREAKER: wait on 0x%010" PRIx64 " released from the last GPU write\n",
		     wait_address);
		g_current_execution->ClearWait();
		return;
	}
	if (TestWaitRegMemValue(current, ref, mask, func)) {
		g_current_execution->ClearWait();
		return;
	}
	// The queue was descheduled while another one pulsed this address and overwrote it again.
	if (g_current_execution->AwaitedAddress() == wait_address &&
	    GuestGpuWriteSatisfied(wait_address, g_current_execution->WaitSeq(), ref, mask, func)) {
		g_current_execution->ClearWait();
		return;
	}
	// A write already in the GPU stream lands before this queue's following work executes.
	if (uint64_t promised = 0;
	    PendingGuestGpuWriteSatisfies(wait_address, ref, mask, func, promised)) {
		if (LabelTraceEnabled()) {
			LOGF("LABEL wait q=%d sub=%" PRIu64 " addr=0x%010" PRIx64
			     " satisfied by pending GPU write value=0x%" PRIx64 "\n",
			     QueueTag(), m_submit_id, wait_address, promised);
		}
		g_current_execution->ClearWait();
		return;
	}
	{
		// A wait that keeps failing while no GPU write lands anywhere is a deadlock: its release
		// was lost. Hardware never stalls like this, so after three seconds the wait passes.
		struct StalledWait {
			std::chrono::steady_clock::time_point since;
			uint64_t                              seq;
		};
		static std::map<std::pair<int, uint64_t>, StalledWait> stalled;
		const auto key  = std::make_pair(static_cast<int>(QueueTag()), wait_address);
		const auto now  = std::chrono::steady_clock::now();
		const auto seq  = CurrentGuestGpuWriteSeq();
		auto [entry, fresh] = stalled.try_emplace(key, StalledWait {now, seq});
		if (!fresh && entry->second.seq != seq) {
			entry->second = {now, seq};
		} else if (!fresh && now - entry->second.since > std::chrono::seconds(3)) {
			LOGF("\t STALLBREAK: q=%d wait on 0x%016" PRIx64 " released after 3 s without GPU progress\n",
			     QueueTag(), wait_address);
			stalled.erase(entry);
			g_current_execution->ClearWait();
			return;
		}
		if (stalled.size() > 256) {
			stalled.clear();
		}
	}
	if (g_current_execution->AwaitedAddress() != wait_address) {
		g_current_execution->BeginWait(wait_address, g_current_execution->BaselineSeq());
		if (LabelTraceEnabled()) {
			LOGF("LABEL wait q=%d sub=%" PRIu64 " addr=0x%010" PRIx64 " cur=0x%" PRIx64
			     " ref=0x%" PRIx64 " mask=0x%" PRIx64 " func=%" PRIu32 "\n",
			     QueueTag(), m_submit_id, wait_address, static_cast<uint64_t>(current),
			     static_cast<uint64_t>(ref), static_cast<uint64_t>(mask), func);
		}
	}
	g_current_execution->SetWaitOperands(ref, mask, func);
	{
		static std::atomic<uint32_t> wrm_log_count {0};
		RememberAwaitedFence(wait_address);
		const auto wrm_seen = wrm_log_count.fetch_add(1);
		if ((wrm_seen % 2048) == 0) {
			LOGF("\t WRMPOLL: addr=0x%016" PRIx64 " current=0x%016" PRIx64 " ref=0x%016" PRIx64
			     " mask=0x%016" PRIx64 " func=%" PRIu32 " size=%zu\n",
			     wait_address, static_cast<uint64_t>(current), static_cast<uint64_t>(ref),
			     static_cast<uint64_t>(mask), func, sizeof(T));
		}
		if (wrm_seen < 3) {
			LOGF("\t WAITREGMEM: addr=0x%016" PRIx64 " current=0x%08" PRIx64 " ref=0x%08" PRIx64
			     " mask=0x%08" PRIx64 " func=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(addr), static_cast<uint64_t>(current),
			     static_cast<uint64_t>(ref), static_cast<uint64_t>(mask), func);
			Pm4TraceDump();
			const auto* words = reinterpret_cast<const uint32_t*>(
			    reinterpret_cast<uintptr_t>(addr) & ~uintptr_t {0x3f});
			for (uint32_t i = 0; i < 16; i++) {
				LOGF("\t   mem[%02" PRIu32 "] 0x%016" PRIx64 " = 0x%08" PRIx32 "%s\n", i,
				     reinterpret_cast<uint64_t>(words + i), words[i],
				     (words + i) == reinterpret_cast<const uint32_t*>(addr) ? "  <== awaited" : "");
			}
		}
		SuspendPm4("WaitRegMem");
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;
	if (LabelTraceEnabled()) {
		LOGF("LABEL write_data q=%d sub=%" PRIu64 " dst=0x%010" PRIx64 " dw=%" PRIu32
		     " value=0x%08" PRIx32 " was=0x%08" PRIx32 "\n",
		     QueueTag(), m_submit_id, reinterpret_cast<uint64_t>(dst), dw_num, src[0], dst[0]);
	}
	if (IsAwaitedFence(reinterpret_cast<uint64_t>(dst))) {
		static std::atomic<uint32_t> reset_log_count {0};
		if (reset_log_count.fetch_add(1) < 64) {
			LOGF("\t FENCERESET: write_data dst=0x%016" PRIx64 " value=0x%08" PRIx32
			     " while another queue waits on it\n",
			     reinterpret_cast<uint64_t>(dst), src[0]);
		}
	}
	{
		static std::atomic<uint32_t> wd_log_count {0};
		if (wd_log_count.fetch_add(1) < 0) {
			LOGF("\t WRITETRACE: write_data dst=0x%016" PRIx64 " dw=%" PRIu32
			     " control=0x%08" PRIx32 " dst_sel=%" PRIu32 " one_address=%d value=0x%08" PRIx32
			     " 0x%08" PRIx32 " was=0x%08" PRIx32 " 0x%08" PRIx32 "\n",
			     reinterpret_cast<uint64_t>(dst), dw_num, write_control, dst_sel,
			     write_one_address ? 1 : 0, src[0], dw_num > 1 ? src[1] : 0, dst[0],
			     dw_num > 1 ? dst[1] : 0);
		}
	}

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	if (write_one_address) {
		for (uint32_t i = 0; i < dw_num; i++) {
			dst[0] = src[i];
		}
	} else {
		memcpy(dst, src, static_cast<size_t>(dw_num) * sizeof(uint32_t));
	}
	CheckGuestWatch(reinterpret_cast<uint64_t>(dst),
	                write_one_address ? sizeof(uint32_t)
	                                  : static_cast<uint64_t>(dw_num) * sizeof(uint32_t),
	                "write_data");
	if (dw_num == 1 || write_one_address) {
		NoteGuestGpuWriteValue(reinterpret_cast<uint64_t>(dst), src[dw_num - 1]);
	} else {
		NoteGuestGpuWrite(reinterpret_cast<uint64_t>(dst));
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	{
		static std::atomic<uint32_t> dma_log_count {0};
		if (dma_log_count.fetch_add(1) < 0) {
			LOGF("\t WRITETRACE: dma dst=0x%016" PRIx64 " bytes=%" PRIu32 " dst_sel=%u\n",
			     dst_address_or_offset, num_bytes, static_cast<unsigned>(dst_sel));
		}
	}
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	CheckGuestWatch(dst_address_or_offset, num_bytes, "dma_data");
	if (IsAwaitedFence(dst_address_or_offset)) {
		static std::atomic<uint32_t> reset_log_count {0};
		if (reset_log_count.fetch_add(1) < 64) {
			LOGF("\t FENCERESET: dma dst=0x%016" PRIx64 " src_sel=%u bytes=%" PRIu32
			     " while another queue waits on it\n",
			     dst_address_or_offset, static_cast<unsigned>(src_sel), num_bytes);
		}
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		// Selectors the hardware does not define come from damaged packets: skip the copy.
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 16) {
			LOGF("dmaData with destination selector 0x%02" PRIx8 " skipped\n", dst_sel);
		}
		return;
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2 && !dst_gds &&
	    (num_bytes == sizeof(uint32_t) || num_bytes == sizeof(uint64_t)) &&
	    !buffer_cache.IsRegionRegistered(dst_address_or_offset, num_bytes)) {
		// A one-word immediate fill outside any GPU buffer is a fence write. Going through the
		// buffer cache would make the label page GPU-owned, and every later host read of a
		// label in that page would drain the GPU and overwrite values other fence writes
		// placed there at parse time. Fills inside GPU buffers (counters) stay on the GPU.
		// The value lands at end of pipe, after the work recorded before it: the guest reads
		// these fences to learn the GPU is done with memory, so an early write lets it reuse
		// buffers the GPU has not consumed yet.
		const auto value = static_cast<uint32_t>(src_address_or_offset_or_immediate);
		for (uint32_t word = 0; word < num_bytes / 4u; word++) {
			Sync::WriteAtEndOfPipe32(m_submit_id, CurrentBuffer(),
			                         reinterpret_cast<uint32_t*>(dst_address_or_offset) + word,
			                         value);
		}
		return;
	}
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		if (!dst_gds) {
			if (num_bytes == sizeof(uint32_t) || num_bytes == sizeof(uint64_t)) {
				const auto pending = NotePendingGuestGpuWrite(
				    dst_address_or_offset, src_address_or_offset_or_immediate & 0xffffffffu,
				    num_bytes);
				m_renderer.GetCommandScheduler().DeferPriorityOperation(
				    [pending] { ResolvePendingGuestGpuWrite(pending); });
			}
			NoteGuestGpuWrite(dst_address_or_offset);
		}
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds) || (src_gds && dst_gds)) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 16) {
			LOGF("dmaData with source selector 0x%02" PRIx8 " (dst gds=%d) skipped\n", src_sel,
			     dst_gds ? 1 : 0);
		}
		return;
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
	if (!dst_gds) {
		NoteGuestGpuWrite(dst_address_or_offset);
	}
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	submission.submit_seq = CurrentGuestGpuWriteSeq();
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	LOGF("\t GWAIT: gpu idle wait thread=%d\n", Common::Thread::GetThreadIdUnique());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

// Prints, for a scheduler that can no longer run anything, where every submission is suspended
// and which queued submission would release it. Throttled, because the stall is polled.
std::atomic<uint64_t> g_stall_watch_address {0};

void GuestGpu::ReportQueueStall(GuestGpu& gpu) {
	for (uint32_t queue_id = 1; queue_id < QueueCount; queue_id++) {
		for (const auto& entry: gpu.m_queues[queue_id]) {
			if (entry.blocked && entry.command_execution.AwaitedAddress() != 0) {
				g_stall_watch_address.store(entry.command_execution.AwaitedAddress());
			}
		}
	}

	static std::atomic<uint32_t> poll_count {0};
	static std::atomic<uint32_t> report_count {0};
	if ((poll_count.fetch_add(1) % 512) != 0 || report_count.fetch_add(1) >= 2000) {
		return;
	}

	{
		auto&    scheduler = gpu.m_renderer.GetCommandScheduler();
		size_t   depth       = 0;
		uint64_t head_tick   = 0;
		bool     active      = false;
		uint64_t active_tick = 0;
		scheduler.ReportPriorityQueue(&depth, &head_tick, &active, &active_tick);
		LOGF("\t STALLDUMP: ---- every queue is blocked ---- current_tick=%" PRIu64
		     " gpu_tick=%" PRIu64 " eop_queue=%zu head_tick=%" PRIu64 " running=%d run_tick=%" PRIu64
		     "\n",
		     scheduler.CurrentTick(), scheduler.KnownGpuTick(), depth, head_tick, active ? 1 : 0,
		     active_tick);
	}
	for (uint32_t queue_id = 0; queue_id < QueueCount; queue_id++) {
		uint32_t index = 0;
		for (const auto& entry: gpu.m_queues[queue_id]) {
			const auto  awaited = entry.command_execution.AwaitedAddress();
			const auto* reason  = entry.command_execution.SuspendReason();
			LOGF("\t STALLDUMP: queue=%" PRIu32 " sub#%" PRIu32 " blocked=%d started=%d reason=%s"
			     " cursor=0x%05" PRIx32 " dw=0x%05zx awaited=0x%016" PRIx64 " value=0x%08" PRIx32
			     "\n",
			     queue_id, index, entry.blocked ? 1 : 0, entry.started ? 1 : 0,
			     reason != nullptr ? reason : "-", entry.command_execution.SuspendOffset(),
			     entry.commands.size(), awaited,
			     awaited != 0 ? *reinterpret_cast<const uint32_t*>(awaited) : 0);
			if (awaited != 0) {
				LOGF("\t STALLDUMP:     ref=0x%016" PRIx64 " mask=0x%016" PRIx64 " func=%" PRIu32
				     " submit_seq=%" PRIu64 "\n",
				     entry.command_execution.WaitRef(), entry.command_execution.WaitMask(),
				     entry.command_execution.WaitFunc(), entry.submit_seq);
				DescribeGuestGpuWrites(awaited, entry.command_execution.BaselineSeq());
				const char* promised_origin = "?";
				const char* promised_packet = "?";
				if (WasFencePromised(awaited, &promised_origin, &promised_packet)) {
					LOGF("\t STALLDUMP:     a submitted stream promised this write: yes (we lost "
					     "it), by %s in a %s stream\n",
					     promised_packet, promised_origin);
				} else {
					LOGF("\t STALLDUMP:     a submitted stream promised this write: no (guest "
					     "never submitted)\n");
				}
			}
			if (awaited != 0) {
				// A RELEASE_MEM or WRITE_DATA stores the address as a little-endian pair, so the
				// pair locates every packet that could release this wait.
				const auto lo = static_cast<uint32_t>(awaited);
				const auto hi = static_cast<uint32_t>(awaited >> 32u);
				for (uint32_t other = 0; other < QueueCount; other++) {
					uint32_t other_index = 0;
					for (const auto& candidate: gpu.m_queues[other]) {
						const auto& words = candidate.commands;
						for (size_t at = 0; at + 1 < words.size(); at++) {
							if (words[at] != lo || words[at + 1] != hi) {
								continue;
							}
							LOGF("\t STALLDUMP:     referenced by queue=%" PRIu32 " sub#%" PRIu32
							     " at dw=0x%05zx cursor=0x%05" PRIx32 " header=0x%08" PRIx32 "\n",
							     other, other_index, at,
							     candidate.command_execution.SuspendOffset(),
							     at >= 3 ? words[at - 3] : 0);
						}
						other_index++;
					}
				}
			}
			index++;
		}
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = true;
	g_gpu_state  = gpu;
#ifdef _WIN32
	g_gpu_os_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif

	uint64_t last_stall_flush_tick = 0;

	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		bool                         stall_flush    = false;
		SetGpuPhase("select");
		{
			// Recorded work must reach the GPU before this thread sleeps: the guest may be
			// waiting for an end-of-pipe event inside it before it submits anything else.
			auto& scheduler = gpu->m_renderer.GetCommandScheduler();
			// Flush skips a buffer with nothing recorded, so it is safe to ask every time.
			if (gpu->m_submission_count == 0 && gpu->m_commands.empty() && scheduler.Active() &&
			    gpu->m_gfx_cp != nullptr) {
				SetGpuPhase("idle_flush");
				gpu->m_gfx_cp->BufferFlush();
			}
		}
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				// Submissions on one queue retire in order: the guest fences frames through the
				// same labels, so running a later submission first would leave a stale value
				// behind. Only a different queue may run while this one's front is blocked.
				int          selected_queue = -1;
				const size_t selected_index = 0;
				// Compute queues run first: they run concurrently on hardware and block on graphics
				// labels anyway, while a lagging compute queue lets the guest recycle its fences.
				const auto runnable = [&](uint32_t id) {
					const auto& q = gpu->m_queues[id];
					return !q.empty() &&
					       (!q.front().blocked || q.front().blocked_seq != CurrentGuestGpuWriteSeq());
				};
				for (uint32_t offset = 0; offset < ComputeQueueCount && selected_queue < 0; offset++) {
					const auto id = 1 + (gpu->m_next_queue + offset) % ComputeQueueCount;
					if (runnable(id)) {
						selected_queue = static_cast<int>(id);
					}
				}
				if (selected_queue < 0 && runnable(0)) {
					selected_queue = 0;
				}
				static uint32_t stall_rounds = 0;
				if (selected_queue >= 0) {
					stall_rounds = 0;
				}
				if (selected_queue < 0 && ++stall_rounds >= 200) {
					// Stalled for a while with nothing left to flush: a wait whose fence the GPU
					// already released, and the guest then reset, is let through.
					Common::LockGuard write_lock(g_guest_write_mutex);
					for (uint32_t id = 0; id < QueueCount && selected_queue < 0; id++) {
						auto& queue = gpu->m_queues[id];
						if (queue.empty()) {
							continue;
						}
						auto&      execution = queue.front().command_execution;
						const auto address   = execution.AwaitedAddress();
						const auto found     = g_last_gpu_writes.find(address);
						if (address != 0 && found != g_last_gpu_writes.end() &&
						    TestWaitRegMemValue(found->second, execution.WaitRef(),
						                        execution.WaitMask(), execution.WaitFunc())) {
							execution.ForceWaitPass();
							queue.front().blocked = false;
							selected_queue        = static_cast<int>(id);
							stall_rounds          = 0;
						}
					}
				}
				static auto stall_since = std::chrono::steady_clock::now();
				if (selected_queue >= 0 || stall_rounds < 200) {
					stall_since = std::chrono::steady_clock::now();
				} else if (std::chrono::steady_clock::now() - stall_since > std::chrono::seconds(3)) {
					// Every queue has been blocked for seconds: a compute wait on a fence no stream ever
					// wrote lost its release. Hardware never deadlocks here, so let that wait through.
					Common::LockGuard write_lock(g_guest_write_mutex);
					for (uint32_t id = 1; id < QueueCount && selected_queue < 0; id++) {
						auto& queue = gpu->m_queues[id];
						if (queue.empty()) {
							continue;
						}
						auto&      execution = queue.front().command_execution;
						const auto address   = execution.AwaitedAddress();
						uint64_t promised = 0;
						if (address != 0 &&
						    !PendingGuestGpuWriteSatisfies(address, execution.WaitRef(), execution.WaitMask(),
						                                   execution.WaitFunc(), promised)) {
							LOGF("\t STALLBREAK: queue=%u forcing wait on unreleased 0x%016" PRIx64 "\n", id, address);
							execution.ForceWaitPass();
							queue.front().blocked = false;
							selected_queue        = static_cast<int>(id);
							stall_rounds          = 0;
							stall_since           = std::chrono::steady_clock::now();
						}
					}
				}
				if (selected_queue < 0) {
					ReportQueueStall(*gpu);
					gpu->m_processing = false;
					// Every queue waits on a fence. The end-of-pipe write that releases it may
					// still sit in a command buffer this thread has not submitted, and the
					// operation performing that write waits on the GPU timeline directly, so
					// nothing else will push the buffer out. Retire it before idling.
					stall_flush = true;
					for (auto& queue: gpu->m_queues) {
						for (auto& entry: queue) {
							entry.blocked = false;
						}
					}
				} else {
					static std::atomic<uint32_t> select_log_count {0};
					if ((select_log_count.fetch_add(1) % 4096) == 0) {
						LOGF("\t SELECT: queue=%d next=%" PRIu32 " depth=%zu\n", selected_queue,
						     gpu->m_next_queue, gpu->m_queues[static_cast<uint32_t>(selected_queue)].size());
					}
					auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
					submission  = std::move(queue[selected_index]);
					queue.erase(queue.begin() + static_cast<std::ptrdiff_t>(selected_index));
					gpu->m_submission_count--;
					if (selected_queue > 0) {
						gpu->m_next_queue = static_cast<uint32_t>(selected_queue) % ComputeQueueCount;
					}
					gpu->m_processing = true;
					has_submission    = true;
				}
			}
		}
		if (stall_flush) {
			SetGpuPhase("stall_flush");
			auto&      scheduler = gpu->m_renderer.GetCommandScheduler();
			const auto tick      = scheduler.CurrentTick();
			static std::atomic<uint32_t> flush_log_count {0};
			const auto                   flush_seen = flush_log_count.fetch_add(1);
			if ((flush_seen % 512) == 0) {
				LOGF("\t STALLLOOP: tick=%" PRIu64 " last=%" PRIu64 " active=%d\n", tick,
				     last_stall_flush_tick, scheduler.Active() ? 1 : 0);
			}
			if (scheduler.Active() && tick != last_stall_flush_tick) {
				last_stall_flush_tick = tick;
				const auto watch  = g_stall_watch_address.load();
				const auto before = watch != 0 ? *reinterpret_cast<const uint32_t*>(watch) : 0;
				scheduler.Finish();
				const auto after = watch != 0 ? *reinterpret_cast<const uint32_t*>(watch) : 0;
				if (before != after || (flush_seen % 512) == 0) {
					LOGF("\t STALLWATCH: addr=0x%016" PRIx64 " before=0x%08" PRIx32
					     " after=0x%08" PRIx32 "\n",
					     watch, before, after);
				}
			} else {
				Common::Thread::SleepMicro(1000);
			}
			continue;
		}

		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			// A readback may create buffers, which records commands; bind a context when idle.
			auto& scheduler = gpu->m_renderer.GetCommandScheduler();
			if (!scheduler.Active() && gpu->m_gfx_cp != nullptr) {
				scheduler.Begin(gpu->m_gfx_cp->GetCtx(), gpu->m_gfx_cp->GetUcfg(), gpu->m_gfx_cp->GetShCtx());
			}
			SetGpuPhase("command");
			command();

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		SetGpuPhase("process", submission.queue_id);
		const bool complete = gpu->Process(submission);

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			static std::atomic<uint32_t> blocked_log_count {0};
			if (blocked_log_count.fetch_add(1) < 60) {
				const auto* cmd_reason = submission.command_execution.SuspendReason();
				const auto* ce_reason  = submission.constant_execution.SuspendReason();
				LOGF("\t BLOCKED: queue_id=%" PRIu32 " type=%d cmd_reason=%s ce_reason=%s\n",
				     submission.queue_id, static_cast<int>(submission.type),
				     cmd_reason != nullptr ? cmd_reason : "-",
				     ce_reason != nullptr ? ce_reason : "-");
			}
			submission.blocked     = true;
			submission.blocked_seq = CurrentGuestGpuWriteSeq();
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				for (auto& entry: queue) {
					entry.blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

bool GuestGpu::Process(Submission& submission) {
	Common::WaitTrace::Scope process_scope(Common::WaitTrace::Kind::GpuProcess);

	const bool first_slice = !submission.started;
	auto& cp = GetProcessor(submission.queue_id);

	if (first_slice && submission.reset_processor) {
		cp.Reset();
	}

	if (first_slice) {
		submission.started = true;
		submission.command_execution.SetBaselineSeq(submission.submit_seq);
		submission.constant_execution.SetBaselineSeq(submission.submit_seq);
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
		cp.QueueReservedFlips(submission.reserved_flips);
	}

	cp.BufferInit();
	cp.SetStreamOrigin(submission.commands.data(), submission.guest_origin);
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			{
				static std::atomic<uint32_t> parse_log_count {0};
				if (parse_log_count.fetch_add(1) < 64) {
					LOGF("\t ORDER: graphics parse done submit_id=%" PRIu64 " complete=%d\n",
					     m_submit_id, complete ? 1 : 0);
				}
			}
			if (progressed) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::Compute: {
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			if (submission.command_execution.MadeProgress()) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			m_renderer.PrefetchReadbacks();
			m_renderer.RunGarbageCollector();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
		if (FrameCapture::Active()) {
			FrameCapture::NoteMarker(fmt::format("submit 0x{:010x} dw={}",
			                                     reinterpret_cast<uint64_t>(commands.data()),
			                                     commands.size()));
		}
	}
	if (LabelTraceEnabled() && execution.m_suspended && execution.m_suspend_words != 0) {
		uint64_t   words = 0;
		const auto sum   = RemainderChecksum(execution, words);
		if (sum != execution.m_suspend_checksum || words != execution.m_suspend_words) {
			const auto& cursor = execution.m_buffer_stack.back();
			LOGF("LABEL STREAM CHANGED while suspended (%s): sub=%" PRIu64 " buffer=0x%016" PRIx64
			     " offset=%u words=%" PRIu64 "->%" PRIu64 "\n",
			     execution.m_suspend_reason != nullptr ? execution.m_suspend_reason : "-", m_submit_id,
			     reinterpret_cast<uint64_t>(cursor.commands.data()), cursor.offset_dw,
			     execution.m_suspend_words, words);
		}
		execution.m_suspend_words = 0;
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	ProcessPm4(execution);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

uint64_t CommandProcessor::RemainderChecksum(const Pm4Execution& execution, uint64_t& words) {
	words = 0;
	if (execution.m_buffer_stack.empty()) {
		return 0;
	}
	const auto& cursor = execution.m_buffer_stack.back();
	uint64_t    sum    = 0;
	for (size_t i = cursor.offset_dw; i < cursor.commands.size() && words < 65536; i++, words++) {
		sum = sum * 1099511628211ull + cursor.commands[i];
	}
	return sum;
}

void CommandProcessor::SuspendPm4(const char* reason) {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspend_reason = reason;
	g_current_execution->m_suspended      = true;
	if (LabelTraceEnabled()) {
		g_current_execution->m_suspend_checksum =
		    RemainderChecksum(*g_current_execution, g_current_execution->m_suspend_words);
	}
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	while (!execution.m_buffer_stack.empty()) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			if (FrameCapture::Active()) {
				FrameCapture::NoteMarker(fmt::format("end 0x{:010x} dw={}",
				                                     reinterpret_cast<uint64_t>(cursor.commands.data()),
				                                     cursor.commands.size()));
			}
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		// Only type 3 carries a body. Type 0/1/2 are header-only padding, skipped one dword at a
		// time exactly as the disassembler in pm4.cpp does.
		if ((packet_header >> 30u) != 3u) {
			if (packet_header != 0x80000000u) {
				static std::atomic<uint32_t> skip_log_count {0};
				if (skip_log_count.fetch_add(1) < 64) {
					LOGF("\t skipping type %" PRIu32 " packet at 0x%05" PRIx32
					     ": cmd_id = 0x%08" PRIx32 "\n",
					     packet_header >> 30u, total_dw - remaining_dw, packet_header);
				}
			}
			if (packet_header != 0x80000000u) {
				cursor.in_data = true;
			}
			Pm4TraceRecord(total_dw - remaining_dw, packet_header, 1);
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		// Inside an inline data block (constants a later SET_SH_REG points a V# at), animated
		// floats such as -2.25 (0xc0101000, a NOP) decode as known packets and swallow the rest
		// of the buffer, which made the Uncharted menu UI blink. A header met while stepping
		// through data counts as a packet only if the packet after it is known too, or it ends
		// exactly at the end of the buffer. (Requiring two failed: data, SET_SH_REG, DRAW, data
		// is a real sequence.)
		if (cursor.in_data) {
			const auto known_at = [&](uint32_t at, uint32_t& len) {
				if (at >= total_dw) {
					return false;
				}
				const auto header = cursor.commands[at];
				if ((header >> 30u) != 3u || g_cp_op_func[(header >> 8u) & 0xffu] == nullptr) {
					return false;
				}
				len = KYTY_PM4_LEN(header);
				return len <= total_dw - at;
			};
			const auto at    = cursor.offset_dw;
			uint32_t   len0  = 0;
			uint32_t   len1  = 0;
			bool       valid = known_at(at, len0);
			if (valid && at + len0 != total_dw) {
				valid = known_at(at + len0, len1);
			}
			// Register writes decoded from data would corrupt render state; a real one names a
			// small register offset (plus an index field in the top bits).
			const auto op0 = opcode;
			if (valid && (op0 == Pm4::IT_SET_CONTEXT_REG || op0 == Pm4::IT_SET_SH_REG ||
			              op0 == Pm4::IT_SET_UCONFIG_REG || op0 == Pm4::IT_SET_UCONFIG_REG_INDEX ||
			              op0 == Pm4::IT_SET_CONFIG_REG)) {
				valid = len0 >= 3 && (cursor.commands[at + 1] & 0x0fff0000u) == 0 &&
				        (cursor.commands[at + 1] & 0xffffu) < 0x1000u;
			}
			if (valid) {
				if (static std::atomic<uint32_t> log_count {0}; log_count.fetch_add(1) < 400) {
					LOGF("PM4DATA accepted header=0x%08" PRIx32 " op=0x%02" PRIx32 " len=%" PRIu32
					     " body0=0x%08" PRIx32 " next=0x%08" PRIx32 "\n",
					     packet_header, opcode, len0, len0 > 1 ? cursor.commands[at + 1] : 0u,
					     at + len0 < total_dw ? cursor.commands[at + len0] : 0u);
				}
			}
			if (!valid) {
				if (static std::atomic<uint32_t> log_count {0};
				    len0 != 0 && log_count.fetch_add(1) < 2000) {
					LOGF("PM4DATA rejected header=0x%08" PRIx32 " op=0x%02" PRIx32 " len=%" PRIu32
					     " at=0x%05" PRIx32 "/%" PRIu32 " next=0x%08" PRIx32 "\n",
					     packet_header, opcode, len0, at, total_dw,
					     at + len0 < total_dw ? cursor.commands[at + len0] : 0u);
				}
				Pm4TraceRecord(total_dw - remaining_dw, packet_header, 1);
				cursor.offset_dw++;
				execution.m_made_progress = true;
				continue;
			}
			cursor.in_data = false;
		}

		// A submitted size may end mid-packet. Handlers read their body unconditionally, so the
		// declared length has to be checked before dispatch or they read past the buffer. Type 3
		// always spans at least two dwords, so a lone trailing header is truncated by definition.
		if (KYTY_PM4_LEN(packet_header) > remaining_dw) {
			static std::atomic<uint32_t> truncated_log_count {0};
			if (truncated_log_count.fetch_add(1) < 16) {
				LOGF("\t truncated packet at 0x%05" PRIx32 ": cmd_id = 0x%08" PRIx32
				     ", header_dw = %" PRIu32 ", remaining_dw = %" PRIu32 " buffer=0x%016" PRIx64
				     " dw=%" PRIu32 "\n",
				     total_dw - remaining_dw, packet_header, KYTY_PM4_LEN(packet_header),
				     remaining_dw, reinterpret_cast<uint64_t>(cursor.commands.data()), total_dw);
				LOGF("\t   partial:");
				for (uint32_t i = 0; i < std::min<uint32_t>(remaining_dw, 24); i++) {
					LOGF(" %08" PRIx32, packet[i]);
				}
				LOGF("\n");
				for (uint32_t i = 0; i + 4 < remaining_dw; i++) {
					if (packet[i] == 0xc004105cu) {
						LOGF("\t   DROPPED FLIP inside abandoned tail at +%" PRIu32 " arg=%" PRId64
						     "\n",
						     i,
						     static_cast<int64_t>(packet[i + 4] |
						                          (static_cast<uint64_t>(packet[i + 5]) << 32u)));
					}
				}
				Pm4TraceDump();
			}
			cursor.offset_dw          = cursor.commands.size();
			execution.m_made_progress = true;
			continue;
		}

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		// Command buffers carry inline constant blocks (for example the 32-dword blocks a later
		// SET_SH_REG points a V# at), and those are stepped over one dword at a time like any
		// other non-type-3 dword. A float in them such as 0xc09b3332 (-4.85) looks like a type-3
		// header; step over it too instead of dropping the rest of the buffer. Past a handful per
		// buffer, the buffer is more likely stale and is dropped below.
		constexpr uint32_t kMaxUnknownHeadersPerBuffer = 16;
		if (handler == nullptr && cursor.unknown_headers < kMaxUnknownHeadersPerBuffer &&
		    (execution.m_buffer_stack.size() > 1 ||
		     packet - (total_dw - remaining_dw) != m_stream_copy)) {
			static std::atomic<uint32_t> data_log_count {0};
			if (data_log_count.fetch_add(1) < 32) {
				LOGF("PM4: unknown header 0x%08" PRIx32 " in buffer 0x%016" PRIx64
				     " at dw 0x%05" PRIx32 "/%" PRIu32 "; stepping over it as inline data\n",
				     packet_header, reinterpret_cast<uint64_t>(packet - (total_dw - remaining_dw)),
				     total_dw - remaining_dw, total_dw);
			}
			cursor.unknown_headers++;
			cursor.in_data = true;
			Pm4TraceRecord(total_dw - remaining_dw, packet_header, 1);
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		if (handler == nullptr && (execution.m_buffer_stack.size() > 1 ||
		                           packet - (total_dw - remaining_dw) != m_stream_copy)) {
			// An indirect buffer is read live from guest memory; the game may already have
			// reused it. Skip the rest of it rather than stop the emulator.
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 32) {
				LOGF("PM4: unknown packet 0x%08" PRIx32 " in indirect buffer 0x%016" PRIx64
				     " at dw 0x%05" PRIx32 "/%" PRIu32 "; skipping the rest of the buffer\n",
				     packet_header, reinterpret_cast<uint64_t>(packet - (total_dw - remaining_dw)),
				     total_dw - remaining_dw, total_dw);
			}
			if (static std::atomic<uint32_t> walk_log_count {0}; walk_log_count.fetch_add(1) < 2) {
				// Re-walk the buffer from its start to show which packet led the parser here.
				const auto  offset = total_dw - remaining_dw;
				auto* const base   = packet - offset;
				uint32_t raw_run = 0;
				for (uint32_t at = 0; at < offset;) {
					const auto header = base[at];
					if ((header >> 30u) != 3u) {
						raw_run++;
						at++;
						continue;
					}
					if (raw_run != 0) {
						LOGF("PM4WALK   (%" PRIu32 " non-type-3 dwords)\n", raw_run);
						raw_run = 0;
					}
					const auto len = KYTY_PM4_LEN(header);
					LOGF("PM4WALK +0x%05" PRIx32 " header=0x%08" PRIx32 " op=0x%02" PRIx32
					     " len=%" PRIu32 " :",
					     at, header, (header >> 8u) & 0xffu, len);
					for (uint32_t i = 1; i < std::min<uint32_t>(len, 10); i++) {
						LOGF(" %08" PRIx32, base[at + i]);
					}
					LOGF("\n");
					at += len;
				}
				if (raw_run != 0) {
					LOGF("PM4WALK   (%" PRIu32 " non-type-3 dwords)\n", raw_run);
				}
				for (uint32_t i = offset; i < std::min<uint32_t>(total_dw, offset + 48); i += 8) {
					LOGF("PM4WALK tail +0x%05" PRIx32 ":", i);
					for (uint32_t j = i; j < std::min<uint32_t>(total_dw, i + 8); j++) {
						LOGF(" %08" PRIx32, base[j]);
					}
					LOGF("\n");
				}
			}
			execution.m_buffer_stack.pop_back();
			execution.m_made_progress = true;
			continue;
		}
		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			KYTY_PM4_FATAL_LOG("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			                   ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			                   reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			                   reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > kPm4DumpBefore ? offset - kPm4DumpBefore : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + kPm4DumpAfter);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				KYTY_PM4_FATAL_LOG("\t%05" PRIx32 "%s %08" PRIx32 "\n", i,
				                   (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		uint32_t packet_dw = 0;
		{
			Common::WaitTrace::Scope handler_scope(Common::WaitTrace::Kind::GpuHandler);
			packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		}
		if (packet_dw > remaining_dw) {
			// The handler already read past the end; name the packet before dying.
			const auto offset = total_dw - remaining_dw;
			KYTY_PM4_FATAL_LOG("PM4 packet overran the command buffer: op=0x%02" PRIx32
			                   ", cmd_id=0x%08" PRIx32 ", header_dw=%" PRIu32
			                   ", consumed_dw=%" PRIu32 ", remaining_dw=%" PRIu32
			                   ", num_dw=%" PRIu32 ", offset=0x%05" PRIx32
			                   ", data=0x%016" PRIx64 "\n",
			                   opcode, packet_header, KYTY_PM4_LEN(packet_header), packet_dw,
			                   remaining_dw, total_dw, offset,
			                   reinterpret_cast<uint64_t>(packet - offset));
			const auto  dump_begin = (offset > kPm4DumpBefore ? offset - kPm4DumpBefore : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + kPm4DumpAfter);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				KYTY_PM4_FATAL_LOG("\t%05" PRIx32 "%s %08" PRIx32 "\n", i,
				                   (i == offset ? ":" : " "), base[i]);
			}
			EXIT("PM4 packet overran the command buffer\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32
			     "\n",
			     offset, packet_header);
		}
		if (execution.m_suspended) {
			return;
		}
		Pm4TraceRecord(total_dw - remaining_dw, packet_header, packet_dw);
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			if (FrameCapture::Active()) {
				FrameCapture::NoteMarker(fmt::format(
				    "{} 0x{:010x} dw={} from 0x{:010x}+0x{:x}", execution.m_chain ? "chain" : "call",
				    reinterpret_cast<uint64_t>(execution.m_next_buffer.data()),
				    execution.m_next_buffer.size(), reinterpret_cast<uint64_t>(cursor.commands.data()),
				    cursor.offset_dw));
			}
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances = num_instances;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4("OcclusionQuery");
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			if (wait_op != 0) {
				BufferFlushAndWait();
			}
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = *reinterpret_cast<const volatile uint64_t*>(address);
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = m_num_instances;
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	DrawIndex({.index_count = index_count, .index_addr = index_addr});
}

// Stale CPU copies of GPU-written draw arguments can hold any value; a draw of billions of
// vertices ran for seconds and tripped the driver timeout (device lost).
static bool PlausibleIndirectDraw(uint64_t count, uint64_t instances, const char* what) {
	if (instances <= (uint64_t {1} << 16u) &&
	    count * std::max<uint64_t>(instances, 1u) <= (uint64_t {1} << 26u)) {
		return true;
	}
	static std::atomic<uint32_t> log_count {0};
	if (log_count.fetch_add(1) < 32) {
		LOGF("%s: skipped implausible draw count=%" PRIu64 " instances=%" PRIu64 "\n", what, count,
		     instances);
	}
	return false;
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	const auto* args_addr =
	    reinterpret_cast<const void*>(m_draw_indirect_args_base_addr + data_offset);

	if (!indexed) {
		DrawIndirectArgs args {};
		if (!Libs::LibKernel::Memory::ReadGpuArgs(reinterpret_cast<uint64_t>(args_addr),
	                                                     &args, sizeof(args))) {
		if (!GuestRangeCommitted(args_addr, sizeof(args))) {
			LOGF("\t warning: indirect args at %p are not mapped, draw skipped\n", args_addr);
			return;
		}
		std::memcpy(&args, args_addr, sizeof(args));
	}
		if (args.instance_count != 1u || args.start_vertex_location != 0u ||
		    args.start_instance_location != 0u) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1) < 64) {
				LOGF("\t warning: partial DrawIndirect args: vertex_count=%" PRIu32
				     ", instance_count=%" PRIu32 ", start_vertex=%" PRIu32
				     ", start_instance=%" PRIu32 "\n",
				     args.vertex_count_per_instance, args.instance_count,
				     args.start_vertex_location, args.start_instance_location);
			}
		}
		if (!PlausibleIndirectDraw(args.vertex_count_per_instance, args.instance_count, "DrawIndirect")) {
			return;
		}
		m_num_instances = args.instance_count;
		DrawIndexAuto({.vertex_count   = args.vertex_count_per_instance,
		               .instance_count = args.instance_count,
		               .first_vertex   = args.start_vertex_location,
		               .first_instance = args.start_instance_location,
		               .offset_source  = DrawOffsetSource::IndirectArgs});
		return;
	}

	DrawIndexedIndirectArgs args {};
	// Uncharted's GPU culling writes these counts every frame; the CPU copy is stale until a
	// readback, so scene draws ran with old or zero counts (burned-out lighting, wasted work).
	// Such draws read their counts on the GPU. KYTY_NO_GPU_DRAW_INDIRECT=1 restores host reads.
	static const bool no_gpu_draw_indirect = std::getenv("KYTY_NO_GPU_DRAW_INDIRECT") != nullptr;
	if (!no_gpu_draw_indirect && m_index_type_and_size != 2 && m_index_buffer_size != 0 &&
	    m_ucfg.GetPrimType() != Prospero::PrimitiveType::kQuadListLegacy &&
	    Libs::LibKernel::Memory::IsGpuWrittenRange(reinterpret_cast<uint64_t>(args_addr),
	                                               sizeof(args))) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 8) {
			LOGF("DrawIndexIndirect: GPU-written counts at %p, drawing from the GPU buffer\n",
			     args_addr);
		}
		m_num_instances = 1;
		DrawIndex({.index_count    = m_index_buffer_size,
		           .index_addr     = reinterpret_cast<const void*>(m_index_base_addr),
		           .instance_count = 1,
		           .offset_source  = DrawOffsetSource::IndirectArgs,
		           .gpu_args_addr  = reinterpret_cast<uint64_t>(args_addr)});
		return;
	}
	if (!Libs::LibKernel::Memory::ReadGpuArgs(reinterpret_cast<uint64_t>(args_addr),
	                                                     &args, sizeof(args))) {
		if (!GuestRangeCommitted(args_addr, sizeof(args))) {
			LOGF("\t warning: indirect args at %p are not mapped, draw skipped\n", args_addr);
			return;
		}
		std::memcpy(&args, args_addr, sizeof(args));
	}
	if (args.base_vertex_location != 0u || args.start_instance_location != 0u) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 64) {
			LOGF("\t warning: partial DrawIndexIndirect args: index_count=%" PRIu32
			     ", instance_count=%" PRIu32 ", start_index=%" PRIu32 ", base_vertex=%" PRIu32
			     ", start_instance=%" PRIu32 "\n",
			     args.index_count_per_instance, args.instance_count, args.start_index_location,
			     args.base_vertex_location, args.start_instance_location);
		}
	}

	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(args.start_index_location) * index_size);

	const uint32_t index_count =
	    (m_index_buffer_size != 0 ? std::min(args.index_count_per_instance, m_index_buffer_size)
	                              : args.index_count_per_instance);
	if (GraphicsRunDebugDumpEnabled() && index_count != args.index_count_per_instance) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("\t DrawIndexIndirect: clamped index_count from %" PRIu32 " to %" PRIu32
			     " using INDEX_BUFFER_SIZE\n",
			     args.index_count_per_instance, index_count);
		}
	}

	if (!PlausibleIndirectDraw(index_count, args.instance_count, "DrawIndexIndirect")) {
		return;
	}
	m_num_instances = args.instance_count;
	DrawIndex({.index_count    = index_count,
	           .index_addr     = index_addr,
	           .instance_count = args.instance_count,
	           .base_vertex    = static_cast<int32_t>(args.base_vertex_location),
	           .first_instance = args.start_instance_location,
	           .offset_source  = DrawOffsetSource::IndirectArgs});
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, uint32_t draw_initiator,
                                         bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		uint32_t count_value = 0;
		(void)Libs::LibKernel::Memory::ReadGpuArgs(reinterpret_cast<uint64_t>(count_addr),
		                                            &count_value, sizeof(count_value));
		draw_count = count_value;
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}

	if (draw_count == 0) {
		return;
	}

	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);

	uint64_t index_size = 0;
	if (indexed) {
		switch (m_index_type_and_size) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
		}
	}

	for (uint32_t i = 0; i < draw_count; i++) {
		const auto args_addr = m_draw_indirect_args_base_addr + data_offset +
		                       static_cast<uint64_t>(i) * stride_in_bytes;

		if (!indexed) {
			DrawIndirectArgs args_value {};
			(void)Libs::LibKernel::Memory::ReadGpuArgs(args_addr, &args_value, sizeof(args_value));
			const auto* args = &args_value;
			if (!PlausibleIndirectDraw(args->vertex_count_per_instance, args->instance_count,
			                           "DrawIndirectMulti")) {
				continue;
			}
			m_num_instances = args->instance_count;
			DrawIndexAuto({.vertex_count   = args->vertex_count_per_instance,
			               .instance_count = args->instance_count,
			               .first_vertex   = args->start_vertex_location,
			               .first_instance = args->start_instance_location,
			               .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		DrawIndexedIndirectArgs args_value {};
		(void)Libs::LibKernel::Memory::ReadGpuArgs(args_addr, &args_value, sizeof(args_value));
		const auto* args = &args_value;

		auto* index_addr = reinterpret_cast<const void*>(
		    m_index_base_addr + static_cast<uint64_t>(args->start_index_location) * index_size);

		const uint32_t index_count =
		    (m_index_buffer_size != 0
		         ? std::min(args->index_count_per_instance, m_index_buffer_size)
		         : args->index_count_per_instance);
		if (GraphicsRunDebugDumpEnabled() && index_count != args->index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args->index_count_per_instance, index_count);
			}
		}

		if (!PlausibleIndirectDraw(index_count, args->instance_count, "DrawIndexIndirectMulti")) {
			continue;
		}
		m_num_instances = args->instance_count;
		DrawIndex({.index_count    = index_count,
		           .index_addr     = index_addr,
		           .instance_count = args->instance_count,
		           .base_vertex    = static_cast<int32_t>(args->base_vertex_location),
		           .first_instance = args->start_instance_location,
		           .offset_source  = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));

	uint32_t frame_num = 0;
	// uint32_t local_x   = 1;
	// uint32_t local_y   = 1;
	// uint32_t local_z   = 1;

	{
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
				const auto& cs = m_sh_ctx.GetCs().cs_regs;
				const auto& oa = m_ucfg.GetGdsOaCounter(m_ucfg.GetGdsOaState().GetIndex());
				LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
				     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
				     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
				     "\n",
				     frame_num, m_submit_id, thread_group_x, thread_group_y, thread_group_z,
				     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
				     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
				     cs.data_addr, m_ucfg.GetGdsOaState().GetIndex(),
				     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
				     oa.GetSpaceAvailable());
			}
		}

		const auto& cs = m_sh_ctx.GetCs().cs_regs;
		// local_x        = std::max(cs.num_thread_x, 1u);
		// local_y        = std::max(cs.num_thread_y, 1u);
		// local_z        = std::max(cs.num_thread_z, 1u);
		m_renderer.GetRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
		                                              thread_group_y, thread_group_z, mode);
	}

	/*constexpr uint32_t DispatchInitiatorUseThreadDimensions = 1u << 5u;
	auto               group_count = [](uint32_t threads, uint32_t group_size) {
	    return (threads == 0
	                ? 0u
	                : (threads + std::max(group_size, 1u) - 1u) / std::max(group_size, 1u));
	};

	auto groups_x = thread_group_x;
	auto groups_y = thread_group_y;
	auto groups_z = thread_group_z;
	if ((mode & DispatchInitiatorUseThreadDimensions) != 0) {
	    groups_x = group_count(thread_group_x, local_x);
	    groups_y = group_count(thread_group_y, local_y);
	    groups_z = group_count(thread_group_z, local_z);
	}

	const uint64_t invocations =
	    static_cast<uint64_t>(groups_x) * groups_y * groups_z * local_x * local_y * local_z;
	if (invocations != 0) {
	    BufferFlushAndWait();
	}*/
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	struct DispatchIndirectArgs {
		uint32_t thread_group_x;
		uint32_t thread_group_y;
		uint32_t thread_group_z;
	};

	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	NoteIndirectArgsAddress(args_addr);
	DispatchIndirectArgs args {};
	// Group counts a GPU pass wrote are stale in the CPU copy; dispatch them from the GPU buffer
	// through the clamped path (bad counts are clamped instead of hanging the device).
	// Opt-in (KYTY_GPU_DISPATCH_INDIRECT=1): in Uncharted it tripped the game's frame-marker
	// assert, stalled loading for tens of seconds and ended in null-pointer crashes.
	static const bool gpu_dispatch = std::getenv("KYTY_GPU_DISPATCH_INDIRECT") != nullptr;
	if (gpu_dispatch && (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) == 0 &&
	    Libs::LibKernel::Memory::IsGpuWrittenRange(args_addr, sizeof(args))) {
		m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
		m_renderer.GetRenderExecutor().Dispatch(m_submit_id, CurrentBuffer(), 1, 1, 1, mode,
		                                        args_addr);
		return;
	}
	if (Libs::LibKernel::Memory::ReadGpuArgs(args_addr, &args, sizeof(args))) {
		// A zero count in guest memory is usually one a GPU pass wrote through a raw pointer the
		// tracker never saw (Uncharted's per-material tile lighting: 12 of 19 passes per frame
		// never ran and the frame stayed white). The GPU's copy decides.
		// Default: the GPU's copy is read here (a drain, served from the peek cache until any
		// write is recorded). KYTY_ZERO_ARGS_GPU=1 dispatches from the GPU buffer instead; every
		// run in that mode showed an over-exposed blob instead of the selector's cave.
		static const bool zero_gpu  = std::getenv("KYTY_ZERO_ARGS_GPU") != nullptr;
		static const bool zero_peek = !zero_gpu;
		const bool        zero =
		    args.thread_group_x == 0 || args.thread_group_y == 0 || args.thread_group_z == 0;
		if (zero && zero_gpu && !zero_peek &&
		    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) == 0) {
			m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
			// Outputs stay eligible for readback: loading threads wait on them (stalls without).
			m_renderer.GetRenderExecutor().Dispatch(m_submit_id, CurrentBuffer(), 1, 1, 1, mode,
			                                        args_addr);
			return;
		}
		if (zero_peek && zero) {
			DispatchIndirectArgs gpu {};
			const bool peeked = Libs::LibKernel::Memory::PeekGpuCopy(args_addr, &gpu, sizeof(gpu));
			if (peeked) {
				args = gpu;
			}
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1) < 64) {
				LOGF("ZEROARGS 0x%016" PRIx64 " peeked=%d gpu=%ux%ux%u writer=%016" PRIx64 "\n",
				     args_addr, peeked ? 1 : 0, gpu.thread_group_x, gpu.thread_group_y,
				     gpu.thread_group_z, Libs::Graphics::FindGpuWriter(args_addr));
			}
		}
		const auto groups =
		    uint64_t {args.thread_group_x} * args.thread_group_y * args.thread_group_z;
		// Garbage counts: the clamped GPU path allows 131072 groups; a stale CPU copy with
		// millions of groups ran for seconds and tripped the driver timeout (device lost when
		// the Uncharted selector loads).
		if (groups > (uint64_t {1} << 17u)) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1) < 32) {
				LOGF("DispatchIndirect: skipped implausible %ux%ux%u groups at 0x%016" PRIx64 "\n",
				     args.thread_group_x, args.thread_group_y, args.thread_group_z, args_addr);
			}
			return;
		}
		DispatchDirect(args.thread_group_x, args.thread_group_y, args.thread_group_z, mode);
		return;
	}

	// The GPU-side path read stale argument words and hung the GPU in the intro video; it stays
	// opt-in (KYTY_HOST_INDIRECT_DISPATCH=1) until that is understood.
	static const bool host_indirect = std::getenv("KYTY_HOST_INDIRECT_DISPATCH") != nullptr;
	if (host_indirect && (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) == 0) {
		m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
		m_renderer.GetRenderExecutor().DispatchIndirect(m_submit_id, CurrentBuffer(), args_addr,
		                                                mode);
		return;
	}

	if (!GuestRangeCommitted(reinterpret_cast<const void*>(args_addr), sizeof(args))) {
		LOGF("\t warning: dispatch indirect args at 0x%016" PRIx64 " are not mapped, dispatch skipped\n", args_addr);
		return;
	}
	std::memcpy(&args, reinterpret_cast<const void*>(args_addr), sizeof(args));
	DispatchDirect(args.thread_group_x, args.thread_group_y, args.thread_group_z, mode);
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	if (args.instance_count == 0) {
		args.instance_count = m_num_instances;
	}
	m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	BufferFlush();

	Common::WaitTrace::Scope flip_scope(Common::WaitTrace::Kind::GpuFlipWait);

	m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
	                                      static_cast<int>(display_buffer_index));
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);
	// A damaged packet can name unmapped memory (the host store crashed the emulator). Only the
	// store is skipped: the fence is still recorded and its interrupt still fires. Returning
	// early here dropped completions the game waits on and left the intro at 1 fps.
	// Fences live in memory the direct mapping may not show as committed (GPU-owned pages are
	// read and written through the backing alias, as WaitRegMem does), so the store goes
	// through the alias first and falls back to the direct mapping only when it is committed.
	const auto dst_address  = reinterpret_cast<uint64_t>(dst_gpu_addr);
	const auto store_fence  = [&](const void* data, size_t bytes) {
		if (dst_gpu_addr != nullptr) {
			Libs::LibKernel::Memory::NoteLabelStore(dst_address, bytes);
			Libs::LibKernel::Memory::ReadLogGpuToCpu("eop", dst_address, data, bytes, 0);
		}
		if (dst_gpu_addr == nullptr ||
		    Libs::LibKernel::Memory::TryWriteBacking(dst_address, data, bytes)) {
			return;
		}
		if (Libs::LibKernel::Memory::IsCommittedRange(dst_address, bytes)) {
			std::memcpy(dst_gpu_addr, data, bytes);
			return;
		}
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 16) {
			LOGF("WriteAtEndOfPipe: store to unmapped 0x%016" PRIx64 " skipped, event kept\n",
			     dst_address);
		}
	};
	T probe {};
	const bool dst_writable =
	    dst_gpu_addr == nullptr ||
	    Libs::LibKernel::Memory::TryReadBacking(dst_address, &probe, sizeof(T)) ||
	    Libs::LibKernel::Memory::IsCommittedRange(dst_address, sizeof(T));

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				Sync::TriggerEopEventAtEndOfPipe(command, m_interrupt_event_id,
				                                 interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	auto write32 = [&](bool with_writeback) {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		store_fence(&data, sizeof(data));

		if (with_interrupt) {
			if (with_writeback) {
				Sync::WriteAtEndOfPipeWithInterruptWriteBack32(m_submit_id, command, dst,
				                                               data, m_interrupt_event_id,
				                                               interrupt_context_id);
			} else {
				Sync::WriteAtEndOfPipeWithInterrupt32(m_submit_id, command, dst, data,
				                                      m_interrupt_event_id, interrupt_context_id);
			}
		} else if (with_writeback) {
			Sync::WriteAtEndOfPipeWithWriteBack32(m_submit_id, command, dst, data);
		} else {
			Sync::WriteAtEndOfPipe32(m_submit_id, command, dst, data);
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					if (dst_writable) {
						SynchronizeGpu();
						Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst,
						              value & 0xffffu, value >> 16u);
						Sync::WriteAtEndOfPipeGds32(m_submit_id, command, dst, value & 0xffffu,
						                            value >> 16u);
					}
					if (with_interrupt) {
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32(false);
				return;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00: write32(false); return;
						case 0x38: write32(true); return;
						default: break;
					}
				}
			} else {
				if (event_write_source == 0x04) {
					value = Sync::ReadReferenceClock();
				}
				auto write64 = [&](bool with_writeback) {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					store_fence(&value, sizeof(value));

					if (with_interrupt) {
						if (with_writeback) {
							Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
							    m_submit_id, command, dst, value, m_interrupt_event_id,
							    interrupt_context_id);
						} else {
							Sync::WriteAtEndOfPipeWithInterrupt64(m_submit_id, command, dst,
							                                      value, m_interrupt_event_id,
							                                      interrupt_context_id);
						}
					} else if (with_writeback) {
						Sync::WriteAtEndOfPipeWithWriteBack64(m_submit_id, command, dst,
						                                      value);
					} else {
						Sync::WriteAtEndOfPipe64(m_submit_id, command, dst, value);
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64(false);
									return;
								}
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) {
									write64(false);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64(false);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05) ||
								    (event_index == 0x00)) {
									write64(true);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64(true);
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}

	EXIT("unknown event type\n");
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::EmitGlobalBarrier() {
	Common::WaitTrace::Scope barrier_scope(Common::WaitTrace::Kind::GpuBarrier);
	GetScheduler().CheckActive();

	Common::LockGuard lock(m_renderer.GetMutex());
	CurrentBuffer().RequestGlobalBarrier();
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id, interrupt_context_id);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: EmitGlobalBarrier(); break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

uint64_t CommandProcessor::TakeFlipRequest(CommandBuffer& command, bool reserved) {
	// A flip reserved at submit time already counts as pending; only attach the command buffer.
	if (reserved && !m_reserved_flips.empty()) {
		const auto id = m_reserved_flips.front();
		m_reserved_flips.pop_front();
		if (id != 0) {
			command.GetContext().GetVideoOut().PrepareFlip(id, command);
			return id;
		}
	}
	return Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                 m_flip.flip_arg);
}

void CommandProcessor::Flip(bool reserved) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	auto& command = CurrentBuffer();
	auto  request = TakeFlipRequest(command, reserved);
	Sync::WriteAtEndOfPipeOnlyFlip(m_submit_id, command, m_flip.handle, m_flip.index,
	                               m_flip.flip_mode, m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithFlip32(m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr),
	                                 value, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                 m_flip.flip_arg, request);
	GetScheduler().Flush();
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithInterruptWriteBackFlip32(
	    m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr), value, m_flip.handle,
	    m_flip.index, m_flip.flip_mode, m_flip.flip_arg, request, m_interrupt_event_id);
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().Flush();
	m_renderer.GetVideoOut().CompleteFlip(request_id);
}

void CommandProcessor::SynchronizeGpu() {
	GetScheduler().Finish();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

} // namespace Libs::Graphics
