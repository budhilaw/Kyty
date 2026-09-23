#ifndef KYTY_COMMON_THREADS_H_
#define KYTY_COMMON_THREADS_H_

#include "common/common.h"

#include <memory>
#include <string>

namespace Common {

void InitializeThreads();

// KYTY_WAIT_TRACE=1 accounts, per guest thread, the wall time spent blocked in each kind of
// kernel wait and the work the GPU thread does, so a frame's cost can be attributed to the
// hand-off that actually pays for it. Reported once a second next to the frame rate.
namespace WaitTrace {

enum class Kind
{
	EventFlag,
	Semaphore,
	Equeue,
	CondVar,
	Mutex,
	Sleep,
	GpuProcess,
	GpuSubmit,
	GpuDraw,
	GpuDispatch,
	GpuBarrier,
	GpuFlipWait,
	GpuRenderLock,
	GpuGarbage,
	GpuHandler,
	GpuReadback,
	GpuTickWait,
	GpuPipeline,
	GpuBindings,
	GpuCommit,
	GpuRenderBegin,
	Count
};

void               Initialize();
[[nodiscard]] bool Enabled();
void               Note(Kind kind, uint64_t nanoseconds, uint64_t count = 1);
void               Report(double fps, uint64_t frame);
// OS id of the thread that has spent the most time in this kind, for targeting a profiler.
[[nodiscard]] uint32_t HottestOsThread(Kind kind);
// Records the in-module return addresses of the caller so the report can say who triggers a kind.
void NoteCaller(Kind kind);

// Accumulates one blocked region; does nothing at all when the trace is off.
class Scope {
public:
	explicit Scope(Kind kind);
	~Scope();
	KYTY_CLASS_NO_COPY(Scope);

private:
	Kind     m_kind;
	uint64_t m_start = 0;
};

} // namespace WaitTrace

using thread_func_t    = void (*)(void*);
using wait_poll_func_t = void (*)();

struct ThreadPrivate;
struct MutexPrivate;
struct CondVarPrivate;

class Thread {
public:
	Thread(thread_func_t func, void* arg);
	~Thread();

	void Join();
	void Detach();

	// Once a thread has finished, the id may be reused by another thread.
	[[nodiscard]] std::string GetId() const;

	// The id is unique and can't be reused by another thread.
	[[nodiscard]] int GetUniqueId() const;

	static void SleepMicro(uint32_t micros);
	static void SleepNano(uint64_t nanos);
	static bool IsMainThread();

	// Get current thread id
	// Once a thread has finished, the id may be reused by another thread.
	static std::string GetThreadId();

	// Get current thread id
	// The id is unique and can't be reused by another thread.
	static int GetThreadIdUnique();

	KYTY_CLASS_NO_COPY(Thread);

private:
	std::unique_ptr<ThreadPrivate> m_thread;
};

class Mutex {
public:
	Mutex();
	~Mutex();

	void Lock();
	void Unlock();
	bool TryLock();

	friend class CondVar;

	KYTY_CLASS_NO_COPY(Mutex);

private:
	std::unique_ptr<MutexPrivate> m_mutex;
};

class CondVar {
public:
	CondVar();
	~CondVar();

	void Wait(Mutex* mutex);
	bool WaitFor(Mutex* mutex, uint32_t micros);
	void Signal();
	void SignalAll();

	static void SetWaitPollCallback(wait_poll_func_t callback);

	KYTY_CLASS_NO_COPY(CondVar);

private:
	std::unique_ptr<CondVarPrivate> m_cond_var;
};

class LockGuard {
public:
	using mutex_type = Mutex;

	// NOLINTNEXTLINE(google-runtime-references)
	explicit LockGuard(mutex_type& m): m_mutex(m) { m_mutex.Lock(); }

	~LockGuard() { m_mutex.Unlock(); }

	KYTY_CLASS_NO_COPY(LockGuard);

private:
	mutex_type& m_mutex;
};

} // namespace Common

#endif /* KYTY_COMMON_THREADS_H_ */
