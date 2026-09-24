#include "common/threads.h"

#include "common/assert.h"
#include "common/timer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>             // IWYU pragma: keep
#include <cinttypes>
#include <condition_variable> // IWYU pragma: keep
#include <cstdio>
#include <cstdlib>
#include <mutex>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS && KYTY_COMPILER == KYTY_COMPILER_CLANG
#define KYTY_WIN_CS
#endif

// macOS has no clock_nanosleep.
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS && !defined(__APPLE__)
#define KYTY_POSIX_HIGH_RES_SLEEP
#include <ctime>
#endif

#include <sstream>
#include <string>
#include <thread>

#ifdef KYTY_WIN_CS
#include <windows.h> // IWYU pragma: keep
#include <tlhelp32.h>
// IWYU pragma: no_include <winbase.h>
constexpr DWORD    KYTY_CS_SPIN_COUNT          = 4000;
constexpr uint64_t KYTY_SLEEP_SPIN_LIMIT_100NS = 500; // 50 us

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

static void SleepHighResolution100ns(uint64_t units_100ns) {
	if (units_100ns == 0) {
		return;
	}

	// Keep spinning only where a kernel transition is
	// likely to cost more than the requested delay; ordinary millisecond sleeps use the
	// per-thread high-resolution waitable timer below.
	if (units_100ns <= KYTY_SLEEP_SPIN_LIMIT_100NS) {
		LARGE_INTEGER frequency {};
		LARGE_INTEGER start {};
		if (QueryPerformanceFrequency(&frequency) != 0 && QueryPerformanceCounter(&start) != 0 &&
		    frequency.QuadPart > 0) {
			const auto wait_ticks =
			    static_cast<LONGLONG>((static_cast<long double>(units_100ns) *
			                           static_cast<long double>(frequency.QuadPart)) /
			                          10000000.0L);
			const auto    deadline = start.QuadPart + std::max<LONGLONG>(wait_ticks, 1);
			LARGE_INTEGER now {};
			do {
				if (QueryPerformanceCounter(&now) == 0) {
					break;
				}
				YieldProcessor();
			} while (now.QuadPart < deadline);
			return;
		}
	}

	thread_local HANDLE timer = CreateWaitableTimerExW(
	    nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
	if (timer == nullptr) {
		thread_local HANDLE fallback_timer = CreateWaitableTimerW(nullptr, TRUE, nullptr);
		timer                              = fallback_timer;
	}

	if (timer != nullptr) {
		LARGE_INTEGER due_time {};
		due_time.QuadPart = -static_cast<LONGLONG>(units_100ns);
		if (SetWaitableTimerEx(timer, &due_time, 0, nullptr, nullptr, nullptr, 0) != 0) {
			WaitForSingleObject(timer, INFINITE);
			return;
		}
	}

	std::this_thread::sleep_for(std::chrono::nanoseconds(units_100ns * 100));
}

// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <synchapi.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <__mutex_base>
// IWYU pragma: no_include <__threading_support>
// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <winerror.h>

using InitializeConditionVariable_func_t = /*WINBASEAPI*/ VOID WINAPI (*)(PCONDITION_VARIABLE);
using WakeConditionVariable_func_t    = /*WINBASEAPI*/ VOID       WINAPI (*)(PCONDITION_VARIABLE);
using WakeAllConditionVariable_func_t = /*WINBASEAPI*/ VOID    WINAPI (*)(PCONDITION_VARIABLE);
using SleepConditionVariableCS_func_t = /*WINBASEAPI*/ BOOL    WINAPI (*)(PCONDITION_VARIABLE,
                                                                          PCRITICAL_SECTION, DWORD);

static InitializeConditionVariable_func_t ResolveInitializeConditionVariable() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<InitializeConditionVariable_func_t>(
		    GetProcAddress(h, "InitializeConditionVariable"));
	}
	return nullptr;
}
static WakeConditionVariable_func_t ResolveWakeConditionVariable() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<WakeConditionVariable_func_t>(
		    GetProcAddress(h, "WakeConditionVariable"));
	}
	return nullptr;
}
static WakeAllConditionVariable_func_t ResolveWakeAllConditionVariable() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<WakeAllConditionVariable_func_t>(
		    GetProcAddress(h, "WakeAllConditionVariable"));
	}
	return nullptr;
}
static SleepConditionVariableCS_func_t ResolveSleepConditionVariableCS() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<SleepConditionVariableCS_func_t>(
		    GetProcAddress(h, "SleepConditionVariableCS"));
	}
	return nullptr;
}

#endif

#ifdef KYTY_POSIX_HIGH_RES_SLEEP
// Spin for very short waits; use an absolute deadline for longer waits.
static void SleepHighResolutionNanos(uint64_t nanos) {
	if (nanos == 0) {
		return;
	}

	constexpr uint64_t NANOS_PER_SEC = 1000000000;
	constexpr uint64_t SPIN_LIMIT_NS = 50000; // below this a context switch dominates

	timespec deadline {};
	if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0) {
		std::this_thread::sleep_for(std::chrono::nanoseconds(nanos));
		return;
	}

	auto target_nsec = static_cast<uint64_t>(deadline.tv_nsec) + nanos;
	deadline.tv_sec += static_cast<time_t>(target_nsec / NANOS_PER_SEC);
	deadline.tv_nsec = static_cast<long>(target_nsec % NANOS_PER_SEC);

	if (nanos <= SPIN_LIMIT_NS) {
		timespec now {};
		do {
			if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
				return;
			}
		} while (now.tv_sec < deadline.tv_sec ||
		         (now.tv_sec == deadline.tv_sec && now.tv_nsec < deadline.tv_nsec));
		return;
	}

	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr) == EINTR) {
	}
}
#endif

namespace Common {

using thread_id_t = std::thread::id;

struct MutexPrivate {
#ifdef KYTY_WIN_CS
	MutexPrivate() { InitializeCriticalSectionAndSpinCount(&m_cs, KYTY_CS_SPIN_COUNT); }
	~MutexPrivate() { DeleteCriticalSection(&m_cs); }
	KYTY_CLASS_NO_COPY(MutexPrivate);
	CRITICAL_SECTION m_cs {};
#else
	std::recursive_mutex m_mutex;
#endif
};

struct CondVarPrivate {
#ifdef KYTY_WIN_CS
	CondVarPrivate() {
		static auto func = ResolveInitializeConditionVariable();
		EXIT_NOT_IMPLEMENTED(func == nullptr);
		func(&m_cv);
	}
	~CondVarPrivate() = default;
	KYTY_CLASS_NO_COPY(CondVarPrivate);
	CONDITION_VARIABLE m_cv {};
#else
	std::condition_variable_any m_cv;
#endif
};

static wait_poll_func_t g_cond_wait_poll_callback = nullptr;

struct ThreadPrivate {
	ThreadPrivate(thread_func_t f, void* a): func(f), arg(a), m_thread(&Run, this) {}

	static void Run(ThreadPrivate* t) {
		t->unique_id = Thread::GetThreadIdUnique();
		t->started   = true;
		t->func(t->arg);
	}

	thread_func_t    func;
	void*            arg;
	std::atomic_bool finished    = false;
	std::atomic_bool auto_delete = false;
	std::atomic_bool started     = false;
	int              unique_id   = 0;
	std::thread      m_thread;
};

static thread_id_t      g_main_thread;
static int              g_main_thread_int;
static std::atomic<int> g_thread_counter = 0;

void InitializeThreads() {
	g_main_thread     = std::this_thread::get_id();
	g_main_thread_int = Thread::GetThreadIdUnique();
	WaitTrace::Initialize();
}

Thread::Thread(thread_func_t func, void* arg)
    : m_thread(std::make_unique<ThreadPrivate>(func, arg)) {
	while (!m_thread->started) {
		Common::Thread::SleepMicro(1000);
	}
}

Thread::~Thread() {
	EXIT_IF(!m_thread->finished && !m_thread->auto_delete);

	m_thread.reset();
}

void Thread::Join() {
	EXIT_IF(m_thread->finished || m_thread->auto_delete);

	m_thread->m_thread.join();

	m_thread->finished = true;
}

void Thread::Detach() {
	EXIT_IF(m_thread->finished || m_thread->auto_delete);

	m_thread->auto_delete = true;
	m_thread->m_thread.detach();
}

void Thread::SleepMicro(uint32_t micros) {
#ifdef KYTY_WIN_CS
	SleepHighResolution100ns(static_cast<uint64_t>(micros) * 10);
#elif defined(KYTY_POSIX_HIGH_RES_SLEEP)
	SleepHighResolutionNanos(static_cast<uint64_t>(micros) * 1000);
#else
	std::this_thread::sleep_for(std::chrono::microseconds(micros));
#endif
}

void Thread::SleepNano(uint64_t nanos) {
#ifdef KYTY_WIN_CS
	SleepHighResolution100ns((nanos + 99) / 100);
#elif defined(KYTY_POSIX_HIGH_RES_SLEEP)
	SleepHighResolutionNanos(nanos);
#else
	std::this_thread::sleep_for(std::chrono::nanoseconds(nanos));
#endif
}

bool Thread::IsMainThread() {
	return g_main_thread == std::this_thread::get_id();
}

std::string Thread::GetId() const {
	std::stringstream ss;
	ss << m_thread->m_thread.get_id();
	return ss.str();
}

int Thread::GetUniqueId() const {
	return m_thread->unique_id;
}

std::string Thread::GetThreadId() {
	std::stringstream ss;
	ss << std::this_thread::get_id();
	return ss.str();
}

Mutex::Mutex(): m_mutex(std::make_unique<MutexPrivate>()) {}

Mutex::~Mutex() {
	m_mutex.reset();
}

void Mutex::Lock() {
#ifdef KYTY_WIN_CS
	EnterCriticalSection(&m_mutex->m_cs);
#else
	m_mutex->m_mutex.lock();
#endif
}

void Mutex::Unlock() {
#ifdef KYTY_WIN_CS
	LeaveCriticalSection(&m_mutex->m_cs);
#else
	m_mutex->m_mutex.unlock();
#endif
}

bool Mutex::TryLock() {
#ifdef KYTY_WIN_CS
	return (TryEnterCriticalSection(&m_mutex->m_cs) != 0);
#else
	return m_mutex->m_mutex.try_lock();
#endif
}

CondVar::CondVar(): m_cond_var(std::make_unique<CondVarPrivate>()) {}

CondVar::~CondVar() {
	m_cond_var.reset();
}

void CondVar::Wait(Mutex* mutex) {
#ifndef KYTY_WIN_CS
	std::unique_lock<std::recursive_mutex> cpp_lock(mutex->m_mutex->m_mutex, std::adopt_lock_t());
#endif
	auto poll_callback = [&] {
		auto* callback = g_cond_wait_poll_callback;
		if (callback == nullptr) {
			return;
		}
#if defined(KYTY_WIN_CS)
		LeaveCriticalSection(&mutex->m_mutex->m_cs);
		callback();
		EnterCriticalSection(&mutex->m_mutex->m_cs);
#else
		cpp_lock.unlock();
		callback();
		cpp_lock.lock();
#endif
	};
#ifdef KYTY_WIN_CS
	static auto func = ResolveSleepConditionVariableCS();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	if (g_cond_wait_poll_callback == nullptr) {
		func(&m_cond_var->m_cv, &mutex->m_mutex->m_cs, INFINITE);
	} else {
		if (func(&m_cond_var->m_cv, &mutex->m_mutex->m_cs, 10) == 0 &&
		    GetLastError() == ERROR_TIMEOUT) {
			poll_callback();
		}
	}
#else
	if (g_cond_wait_poll_callback == nullptr) {
		m_cond_var->m_cv.wait(cpp_lock);
	} else {
		if (m_cond_var->m_cv.wait_for(cpp_lock, std::chrono::microseconds(10000)) ==
		    std::cv_status::timeout) {
			poll_callback();
		}
	}
	cpp_lock.release();
#endif
}

void CondVar::SetWaitPollCallback(wait_poll_func_t callback) {
	g_cond_wait_poll_callback = callback;
}

namespace WaitTrace {
namespace {

constexpr int MaxThreads = 256;

struct Slot {
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Kind::Count)> ticks {};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Kind::Count)> counts {};
	std::atomic_bool                                                    used {false};
	std::atomic<uint32_t>                                               os_id {0};
};

std::array<Slot, MaxThreads> g_slots {};

const char* KindName(Kind kind) {
	switch (kind) {
		case Kind::EventFlag: return "evflag";
		case Kind::Semaphore: return "sema";
		case Kind::Equeue: return "equeue";
		case Kind::CondVar: return "cond";
		case Kind::Mutex: return "mutex";
		case Kind::Sleep: return "sleep";
		case Kind::GpuProcess: return "gpu_process";
		case Kind::GpuSubmit: return "gpu_submit";
		case Kind::GpuDraw: return "draw";
		case Kind::GpuDispatch: return "dispatch";
		case Kind::GpuBarrier: return "barrier";
		case Kind::GpuFlipWait: return "flipwait";
		case Kind::GpuRenderLock: return "renderlock";
		case Kind::GpuGarbage: return "gc";
		case Kind::GpuHandler: return "handler";
		case Kind::GpuReadback: return "readback";
		case Kind::GpuTickWait: return "tickwait";
		case Kind::GpuPipeline: return "pipeline";
		case Kind::GpuBindings: return "bindings";
		case Kind::GpuCommit: return "commit";
		case Kind::GpuRenderBegin: return "renderbegin";
		default: return "?";
	}
}

bool     g_enabled = false;
uint64_t g_qpc_frequency = 1;

constexpr size_t CallerFrames  = 4;
constexpr size_t CallerEntries = 96;

struct CallerEntry {
	std::array<uint64_t, CallerFrames> frames {};
	uint64_t                           count = 0;
};

std::mutex                                                                     g_caller_mutex;
std::array<std::array<CallerEntry, CallerEntries>, static_cast<size_t>(Kind::Count)> g_callers {};

// Counting has to be near free or it changes the thing being measured, so this is a performance
// counter read plus a relaxed add on a cached slot, with no allocation and no id lookup.
thread_local Slot* t_slot = nullptr;

Slot* LocalSlot() {
	if (t_slot == nullptr) {
		const auto id = Thread::GetThreadIdUnique();
		if (id < 0 || id >= MaxThreads) {
			return nullptr;
		}
		t_slot = &g_slots[static_cast<size_t>(id)];
		t_slot->used.store(true, std::memory_order_relaxed);
#ifdef KYTY_WIN_CS
		t_slot->os_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
	}
	return t_slot;
}

} // namespace

void Initialize() {
	g_enabled       = std::getenv("KYTY_WAIT_TRACE") != nullptr;
	const auto freq = Timer::QueryPerformanceFrequency();
	g_qpc_frequency = (freq != 0 ? freq : 1);
}

bool Enabled() {
	return g_enabled;
}

void Note(Kind kind, uint64_t ticks, uint64_t count) {
	auto* slot = LocalSlot();
	if (slot == nullptr) {
		return;
	}
	slot->ticks[static_cast<size_t>(kind)].fetch_add(ticks, std::memory_order_relaxed);
	slot->counts[static_cast<size_t>(kind)].fetch_add(count, std::memory_order_relaxed);
}

Scope::Scope(Kind kind): m_kind(kind) {
	if (g_enabled) {
		m_start = Timer::QueryPerformanceCounter();
	}
}

Scope::~Scope() {
	if (m_start != 0) {
		Note(m_kind, Timer::QueryPerformanceCounter() - m_start);
	}
}

uint32_t HottestOsThread(Kind kind) {
	uint32_t best_os    = 0;
	uint64_t best_ticks = 0;
	for (auto& slot: g_slots) {
		const auto ticks = slot.ticks[static_cast<size_t>(kind)].load(std::memory_order_relaxed);
		if (ticks > best_ticks) {
			best_ticks = ticks;
			best_os    = slot.os_id.load(std::memory_order_relaxed);
		}
	}
	return best_os;
}

// A thread that is blocked all second is idle and uninteresting. What matters is the thread that
// is awake, so rows are ordered by how much of the second each one spent NOT waiting.
void DumpThreads(FILE* out) {
#ifdef _WIN32
	// Game code is mapped at a fixed base, so its return addresses stand out on each stack.
	constexpr uint64_t GameBegin = 0x900000000ull;
	constexpr uint64_t GameEnd   = 0x904000000ull;
	const auto         self      = GetCurrentThreadId();
	HANDLE             snapshot  = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	THREADENTRY32      entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) {
			continue;
		}
		HANDLE thread = OpenThread(THREAD_ALL_ACCESS, FALSE, entry.th32ThreadID);
		if (thread == nullptr) {
			continue;
		}
		if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
			CloseHandle(thread);
			continue;
		}
		CONTEXT context {};
		context.ContextFlags = CONTEXT_FULL;
		if (GetThreadContext(thread, &context) != 0) {
			std::fprintf(out, "tid %lu rip=%llx:", entry.th32ThreadID,
			             static_cast<unsigned long long>(context.Rip));
			const auto* stack = reinterpret_cast<const uint64_t*>(context.Rsp);
			int         found = 0;
			for (int slot = 0; slot < 4096 && found < 20; slot++) {
				uint64_t value = 0;
				if (ReadProcessMemory(GetCurrentProcess(), stack + slot, &value, sizeof(value),
				                      nullptr) == 0) {
					break;
				}
				if (value >= GameBegin && value < GameEnd) {
					std::fprintf(out, " %llx", static_cast<unsigned long long>(value - GameBegin));
					found++;
				}
			}
			std::fprintf(out, "\n");
		}
		ResumeThread(thread);
		CloseHandle(thread);
	}
	CloseHandle(snapshot);
	std::fflush(out);
#else
	(void)out;
#endif
}

void PrintHostStack(const char* label) {
#ifdef _WIN32
	void*      raw[24] {};
	const auto captured = RtlCaptureStackBackTrace(1, 24, raw, nullptr);
	const auto base     = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
	std::printf("%s stack:", label);
	for (unsigned f = 0; f < captured; f++) {
		const auto address = reinterpret_cast<uint64_t>(raw[f]);
		if (address >= base && address - base < 0x4000000ull) {
			std::printf(" %llx", static_cast<unsigned long long>(address - base));
		}
	}
	std::printf("\n");
	std::fflush(stdout);
#else
	std::printf("%s\n", label);
#endif
}

void NoteCaller(Kind kind) {
	if (!g_enabled) {
		return;
	}
#ifdef _WIN32
	void*      raw[12] {};
	const auto captured = RtlCaptureStackBackTrace(1, 12, raw, nullptr);
	const auto base     = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
	std::array<uint64_t, CallerFrames> frames {};
	size_t                             kept = 0;
	for (unsigned f = 0; f < captured && kept < CallerFrames; f++) {
		const auto address = reinterpret_cast<uint64_t>(raw[f]);
		if (address >= base && address - base < 0x4000000ull) {
			frames[kept++] = address - base;
		}
	}
	std::lock_guard lock(g_caller_mutex);
	auto&           table = g_callers[static_cast<size_t>(kind)];
	for (auto& entry: table) {
		if (entry.count != 0 && entry.frames == frames) {
			entry.count++;
			return;
		}
		if (entry.count == 0) {
			entry.frames = frames;
			entry.count  = 1;
			return;
		}
	}
#else
	(void)kind;
#endif
}

void Report(double fps, uint64_t frame) {
	if (!g_enabled) {
		return;
	}
	const auto to_ms = [](uint64_t ticks) {
		return static_cast<double>(ticks) * 1000.0 / static_cast<double>(g_qpc_frequency);
	};

	struct Row {
		int    id         = 0;
		double blocked_ms = 0.0;
	};
	std::array<Row, MaxThreads> rows {};
	int                         used = 0;
	for (int id = 0; id < MaxThreads; id++) {
		auto& slot = g_slots[static_cast<size_t>(id)];
		if (!slot.used.load(std::memory_order_relaxed)) {
			continue;
		}
		double   blocked = 0.0;
		uint64_t events  = 0;
		for (size_t kind = 0; kind < static_cast<size_t>(Kind::Count); kind++) {
			events += slot.counts[kind].load(std::memory_order_relaxed);
			if (kind < static_cast<size_t>(Kind::GpuProcess)) {
				blocked += to_ms(slot.ticks[kind].load(std::memory_order_relaxed));
			}
		}
		// A thread blocked for the whole second never leaves its wait and so records nothing;
		// it is idle, not busy, and would otherwise sort to the top as "awake".
		if (events == 0) {
			continue;
		}
		rows[static_cast<size_t>(used)] = {id, blocked};
		used++;
	}
	std::sort(rows.begin(), rows.begin() + used,
	          [](const Row& a, const Row& b) { return a.blocked_ms < b.blocked_ms; });

	std::printf("WAITTRACE fps=%.1f frame=%" PRIu64 " (rows sorted by busiest)\n", fps, frame);
	const int shown = (used < 12 ? used : 12);
	for (int row = 0; row < shown; row++) {
		const auto  id   = rows[static_cast<size_t>(row)].id;
		auto&       slot = g_slots[static_cast<size_t>(id)];
		std::printf("  t%-3d os=%-6u awake~%6.1fms", id, slot.os_id.load(std::memory_order_relaxed),
		            1000.0 - rows[static_cast<size_t>(row)].blocked_ms);
		for (size_t kind = 0; kind < static_cast<size_t>(Kind::Count); kind++) {
			const auto ticks = slot.ticks[kind].load(std::memory_order_relaxed);
			const auto n     = slot.counts[kind].load(std::memory_order_relaxed);
			if (n != 0) {
				std::printf("  %s=%.1fms/%" PRIu64, KindName(static_cast<Kind>(kind)), to_ms(ticks),
				            n);
			}
		}
		std::printf("\n");
	}
	{
		std::lock_guard lock(g_caller_mutex);
		for (size_t kind = 0; kind < static_cast<size_t>(Kind::Count); kind++) {
			auto& table = g_callers[kind];
			std::sort(table.begin(), table.end(),
			          [](const CallerEntry& a, const CallerEntry& b) { return a.count > b.count; });
			if (table[0].count == 0) {
				continue;
			}
			std::printf("  CALLERS %s:", KindName(static_cast<Kind>(kind)));
			for (size_t i = 0; i < 6 && table[i].count != 0; i++) {
				std::printf(" %" PRIu64 "x[", table[i].count);
				for (const auto rva: table[i].frames) {
					std::printf("%" PRIx64 ",", rva);
				}
				std::printf("]");
			}
			std::printf("\n");
			for (auto& entry: table) {
				entry = {};
			}
		}
	}
	std::fflush(stdout);
	for (auto& slot: g_slots) {
		for (size_t kind = 0; kind < static_cast<size_t>(Kind::Count); kind++) {
			slot.ticks[kind].store(0, std::memory_order_relaxed);
			slot.counts[kind].store(0, std::memory_order_relaxed);
		}
	}
}

} // namespace WaitTrace

bool CondVar::WaitFor(Mutex* mutex, uint32_t micros) {
	bool ok = false;
#ifndef KYTY_WIN_CS
	std::unique_lock<std::recursive_mutex> cpp_lock(mutex->m_mutex->m_mutex, std::adopt_lock_t());
#endif
#ifdef KYTY_WIN_CS
	static auto func = ResolveSleepConditionVariableCS();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	ok = !(func(&m_cond_var->m_cv, &mutex->m_mutex->m_cs, (micros < 1000 ? 1 : micros / 1000)) ==
	           0 &&
	       GetLastError() == ERROR_TIMEOUT);
#else
	ok = (m_cond_var->m_cv.wait_for(cpp_lock, std::chrono::microseconds(micros)) ==
	      std::cv_status::no_timeout);
	cpp_lock.release();
#endif
	return ok;
}

void CondVar::Signal() {
#ifdef KYTY_WIN_CS
	static auto func = ResolveWakeConditionVariable();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	func(&m_cond_var->m_cv);
#else
	m_cond_var->m_cv.notify_one();
#endif
}

void CondVar::SignalAll() {
#ifdef KYTY_WIN_CS
	static auto func = ResolveWakeAllConditionVariable();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	func(&m_cond_var->m_cv);
#else
	m_cond_var->m_cv.notify_all();
#endif
}

int Thread::GetThreadIdUnique() {
	static thread_local int tid = ++g_thread_counter;
	return tid;
}

} // namespace Common
