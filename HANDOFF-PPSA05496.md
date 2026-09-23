# Handoff: getting PPSA05496 (LEGO Builder's Journey, Unity/AGC) to the main menu on KytyPS5

Branch: `feat/depth-to-color-copy` (base `main`). Last commit: `ebfc633`. Everything below
"Uncommitted work" is in the working tree only.

## 1. Goal and current status

Goal: boot `[PS5DUMPID] PPSA05496 Builder's Journey (03.002.583)` until the main menu renders.

Status: **the game boots and renders its title screen** ("BUILDER'S JOURNEY" logo art) at ~27 fps,
running for minutes, thousands of command buffers, ~2300 presented frames. The remaining blocker
is a deterministic crash while the main menu scene loads. Section 8 has the current state; the
sections below it are the earlier investigation and are still accurate history.

## 2. Build / run (exact commands, PowerShell)

Environment script (creates MSVC + Ninja + Vulkan + Qt env), copied to `scratchpad-kyty-env.ps1` in the repo root:
`C:\Users\Ericsson\AppData\Local\Temp\claude\C--Users-Ericsson-Dev-Personal-Other-KytyPS5\af78625a-79a3-4933-a159-3782c85e56ff\scratchpad\kyty-env.ps1`
(copy it somewhere permanent; it is in a per-session temp dir).

```powershell
Get-Process kyty_emulator -EA SilentlyContinue | Stop-Process -Force   # MUST kill first, see trap below
& "<path>\kyty-env.ps1" | Out-Null
cmake --build "C:\Users\Ericsson\Dev\Personal\Other\KytyPS5\_Build\windows" --target launcher --parallel 2>&1 | Select-String "error|FAILED"
cmake --install "C:\Users\Ericsson\Dev\Personal\Other\KytyPS5\_Build\windows" --prefix "C:\Users\Ericsson\Dev\Personal\Other\KytyPS5\_Build\windows\install" 2>&1 | Select-String "kyty_emulator.exe"
```

Run headless with the log to a file (the log is the primary tool; `_err.txt` is often empty):

```powershell
$exe="C:\Users\Ericsson\Dev\Personal\Other\KytyPS5\_Build\windows\install\kyty_emulator.exe"
$g="C:\Users\Ericsson\Downloads\[PS5DUMPID] PPSA05496 Builder's Journey (03.002.583)"
Set-Location (Split-Path $exe); [System.IO.File]::Delete("_kyty.txt")
$p = Start-Process -FilePath $exe -ArgumentList @('--game',"`"$g`"",'--printf-direction','File','--printf-output-file','_kyty.txt') -PassThru
Start-Sleep 60; $p.MainWindowTitle; Stop-Process -Id $p.Id -Force
```

Trap: `cmake --install` silently skips `kyty_emulator.exe` if an instance is still running (file
lock). Always kill first and check the install output contains `kyty_emulator.exe`.

Trap: perl/sed one-liners that write C string literals turn `\t`/`\n` into real characters.
Use the Edit tool for anything containing escapes.

## 3. Root causes found and fixed (chronological)

### 3.1 EOP fence writes were never performed (committed `ebfc633`)
`Sync::RecordEndOfPipeWrite` only recorded debug info. Now it defers the memory write via
`CommandScheduler::DeferPriorityOperation`, gated on the Vulkan master timeline tick. Without it
every `WAIT_REG_MEM` on a RELEASE_MEM label deadlocked.

### 3.2 The AGC driver must snapshot command buffers at submit (uncommitted, `src/libs/agc.cpp`)
Evidence: Unity recycles DCB chunk memory *within the same frame*, right after
`sceAgcDriverSubmitDcb` returns, with no kernel wait in between (traced via GWAIT/GSIGNAL logs:
the worker thread 52 re-recorded chunks 1,21..26 while 20 chunks were still queued). Executing
from borrowed guest memory produced "truncated packet" errors (new stream overwrote old one),
dropped RELEASE_MEM/flip packets and cross-queue fence deadlocks.

Fix: `submit_dcb`/`submit_acb` call `snapshot_command_stream()`, which copies the PM4 stream AND
every table referenced by `SET_{CONTEXT,SH,UCONFIG}_REG_INDIRECT` (opcodes 0x9f/0x63/0x64,
layout `addr_lo|flags, addr_hi, 0x80000000, N` with N (offset,value) pairs) into one owned
`std::vector`, repointing the copied packets at the copied tables. Ownership travels in
`Submission::owned_commands` (`GuestGpu::Submit/SubmitCompute` gained
`std::vector<uint32_t> owned_commands, const uint32_t* guest_origin` params).
`GuestGpu::Submit` itself still borrows memory, so the maintainer's `GpuCommandLane`
"borrowed graphics commands" contract test is untouched (the copy lives in the driver layer).
Verified: 10,559 indirect-patch calls logged, none targeting a pending (un-resubmitted) chunk.

### 3.3 Per-slice render-target growth was O(n^2) (uncommitted, textureCache/image)
Unity clears every slice of a 2D-array render target (up to ~130 layers x 9 mips) with one draw
each, and each draw's view exposed one more layer. `TextureCache::ExpandImage` re-created and
copied the whole image per draw (15-70 ms each, 1.8-3.9 s per 1.5 KB chunk). This slowness is
what made every race above visible.

Fix: `Image` got `capacity_layers` (physical VkImage array layers) and `GrowLayers()` (remaps
`backing.subresource_states`, new layers start `eUndefined`). `ExpandImage` grows in place when
only the layer count changes and capacity allows; otherwise it reallocates with capacity
`max(requested, 2*old)` capped at 512. Result: max submission time 3883 ms -> 375 ms, draws
over 2 ms 344 -> 22.

### 3.4 Null buffer descriptors (uncommitted, `pipeline/descriptors.cpp`)
Unity binds placeholder V#s (`00000000 00fff000 0a500000 00000000`, dst_sel all zero, base
outside the address space). `FindBuffers` now binds the null buffer when `fields[3]==0` or the
base is unreadable instead of `ClampRangeSize` EXITing.

### 3.5 EOP clock-counter writes wrote 0 (uncommitted, `sync.cpp`)
`WriteAtEndOfPipeClockCounter[WithWriteBack]` passed `0` instead of `value`. Fixed.

### 3.6 Diagnostics made informative (uncommitted)
`depthRenderTarget.cpp` "unsupported depth register state" now prints all fields (two sites).
`DepthFatal` logs to the file too (committed `9912774`).

### 3.7 Tried and REVERTED
- Submit-time pre-execution of small `DMA_DATA`/`WRITE_DATA` ("CP-immediate" writes). Wrong:
  the CP executes in order, and Unity's chunks contain `WAIT_REG_MEM(label==1)` followed by
  `WRITE_DATA(label<-0)`; pre-executing the reset before the wait deadlocked the graphics queue.
  Removed. With the parser now fast, parse-time in-order writes are correct.
- Scheduler skip-ahead (run first non-blocked entry of a queue instead of the front) is still in
  `GuestGpu::ThreadRun`. It is probably harmless now but violates in-order queue semantics; it
  should be reverted to front-only once the current blocker is understood.

## 4. Uncommitted work, classified

Keep (real fixes): agc.cpp `snapshot_command_stream` + owned submissions (graphicsRun.h/.cpp,
commandProcessor.h `SetStreamOrigin/LiveGuestPacket`), image.h/.cpp `capacity_layers`/`GrowLayers`,
textureCache.cpp/h `ExpandImage` growth + `InsertImage(info, capacity)`, descriptors.cpp null
descriptors, sync.cpp clock value, depthRenderTarget.cpp detailed fatals.

Strip before committing (diagnostics; user said heavy logging is fine while not production, but
these should not land upstream): all `LOGF` tagged `GWAIT:`, `GSIGNAL:`, `PATCH:`, `DCB:`,
`DCBWALK:`, `DCBFLIP:`, `SUBMITQ:`, `EXEC:`, `EXECTIME:`, `ORDER:`, `EOPQ:`, `EOPPATH:`,
`RELEASEMEM:`, `WRITETRACE:`, `WRMADDR:`, `WAITREGMEM:`, `BLOCKED:`, `QUEUESTALL/FENCESCAN/FLIPSCAN`,
`Pm4Trace`, `INDIRECTCX:`, `VBRANGE:`, `VBTABLE:`, `BUFDESC:` (keep the null-binding logic),
`DRAWTIME/DRAWPHASE/PREPPHASE/RTPHASE/TCPHASE/XTCPHASE/XTCPATH/EXPAND/PIPETIME`, `EQEVENT:`,
`FLIP:` lines in videoOut.cpp, the `GWAIT` logs in kernel/{semaphore,eventFlag,pthread}.cpp,
and the `PIPETIME`/`DrawTimeLogger` timers in renderDraw.cpp/pipelineCache.cpp.
Also revert the scheduler skip-ahead in `GuestGpu::ThreadRun` (see 3.7).

Commit rules from the user: single-line commit messages, no Co-Authored-By trailer, author
Ericsson Budhilaw. Do not stop to ask about committing/pushing/filing issues; keep fixing until
the main menu renders. Don't over-comment code.

## 5. Facts about the game's GPU usage (verified from logs)

- Chunks are 32 KB regions (`begin` stride 0x8030); commands at the bottom (<= 6 KB per
  submit), indirect register tables and per-draw V# tables in the upper area. The whole region
  is recycled after submit (LIFO). Copying only the dwords is not enough; tables must be copied
  (done). Per-draw V# tables (user-data pointer, `fetch_buffer_reg`) are read at draw time from
  guest memory; they were garbage only when parsing lagged by seconds, so the speedup in 3.3 is
  what keeps them valid. If `VBTABLE: unreadable V#` reappears, parsing is lagging again.
- Sync pattern per "SyncTG" block: `DMA_DATA(label<-0)`, `RELEASE_MEM(label<-1 at EOP)`; the
  CPU spins on labels in user space (invisible to kernel-wait logs). Only ONE RELEASE_MEM with an
  interrupt (`int_sel=2`) per run so far (frame 0 end, dst 0x21025d4d0).
- GPU-side flips: Kyty NOP `0xc004105c` (R_FLIP), `flip_arg` = INT64_MIN + frame index.
  Flip completion triggers `TriggerInterrupt(event_id=0)` which is delivered to BOTH graphics
  equeues the game registered: eq 2 "UnityFTMFlipQueue" (thread 50) and eq 3 "EOP QUEUE"
  (thread 53 "Gfx Task Executor"), both `AgcDriverAddEqEvent(id=0)`.
- Threads: 3 main, 49 "Suspend Point Checker" (prints "Wanted to force a call to
  sce::Agc::suspendPoint..." when the main loop stalls > 3 s), 50 UnityEOPThread,
  51 GfxFlipThread (polls eq 4 "Flip Event Queue GfxDeviceAgc" every 500 ms), 52
  UnityGfxDeviceWorker (records chunks), 53 Gfx Task Executor (submits DCB/ACB, sets eventflag
  0x1003c68f8d0 that wakes 52 after an EOP event), 18-30 Job.Workers (eventflag 0x1003c68fb70),
  89-93 FMOD threads (ignore).
- `AgcDriverSubmitDcb` returns OK; the game never calls an unresolved stub; the two unresolved
  `Agc_v1` imports of eboot (K2mciNVxUCE, GPbUp9jXQa8) are never called.

## 6. Current blocker (where to continue)

After flip 3 (frame 1 presented) the CPU chain is:
- main (3): `GWAIT sema=0x1003c348d80 inf` (the sema address changes per run; it is the one the
  main thread waits on last). Historically it is signaled by the worker (52).
- worker (52): `GWAIT eventflag=0x1003c68f8d0 pattern=1 inf`, set only by executor (53) after
  it consumed an EOP event (it did so at 7385 and 61221 in earlier logs).
- executor (53): after the flip-3 EOP event it does NOT set the flag again and makes no further
  kernel calls (its last logged action was a `timo=0` poll then `timo=inf` wait on EOP QUEUE
  before the event; after the event nothing is logged). Either it is spinning in user space on
  a label, or it consumed the event and is waiting on something we do not log (pthread mutex,
  rwlock, nanosleep loop).

Hypotheses to test next, in order:
1. Executor waits for a *second* EOP event or a specific `context_id`/`udata`. Frame 0 produced
   one `int_sel=2` RELEASE_MEM (id 0, ctx 0) plus the flip EOP; frame 1 produced only the flip
   EOP. Check whether frame 1's chunks contain an `EVENT_WRITE_EOP`/`RELEASE_MEM` with interrupt
   that we ignore (grep the copied streams for opcode 0x47/0x49 and NOP R_RELEASE_MEM with
   `interrupt` bits 1/4 - `CpOpReleaseMem` only accepts int_sel 0/2/4; log any `EVENT_WRITE_EOP`).
2. Executor spins on a label written by `WriteAtEndOfPipeClockCounter` (a GPU timestamp; the
   game compares against `submitProcessTimeCounter`). This was writing 0 until 3.5; verify in the
   next run whether behaviour changed (the run in this handoff was made AFTER that fix and still
   stalls, so probably not it, but check the `EndOfPipe Signal!!!` addresses against what 53
   reads).
3. Add logging to `PthreadMutexLock`/`PthreadRwlock*`/`KernelNanosleep` (nanosleep is already
   logged as `GWAIT: nanosleep`) to see where 53 actually blocks, and log `KernelClearEventFlag`
   and `KernelPollEventFlag` (poll is used for non-blocking checks).
4. `VideoOut` flip status: the game reads `flipPendingNum`, `currentBuffer`,
   `submitProcessTimeCounter`; make sure `GetFlipStatus` after GPU flips reports the flip as
   completed (`gcQueueNum == 0`, `flipPendingNum == 0`, `count` increments).
5. Only after the stall is solved: strip diagnostics (section 4), revert scheduler skip-ahead,
   run the test binaries in `_Build/windows` (esp. `shader_recompiler_compute_tests.exe`
   GpuCommandLane and texture cache tests), commit in logical units.

## 7. Useful log queries (bash on the install dir)

```bash
grep -n "FLIP: presented\|--- Error\|--- Fatal" _kyty.txt
grep -n "BLOCKED:\|WAITREGMEM:" _kyty.txt | tail        # GPU queue blocks and awaited fences
grep -n "EXEC: submit_id\|DCB: seq=" _kyty.txt           # submit vs execute order
grep "thread=53$\|thread_id = 53$" _kyty.txt | tail     # per-thread kernel waits/signals
grep -n "EQEVENT:\|Equeue wait received" _kyty.txt      # GPU interrupt -> equeue delivery
grep -n "VBTABLE\|BUFDESC\|truncated packet" _kyty.txt  # memory-recycle symptoms
```
`scratchpad/regname.sh cx 0x31c` (session temp dir) resolves register offsets via Mesa gfx103.json.

## 8. Session 2 (Opus): from 3 frames to the title screen

### 8.1 Root cause of the original hang: out-of-order fence writes
The scheduler ran the first *non-blocked* submission in a queue instead of the front. Unity fences
consecutive frames through the **same** label, so running a later submission first left a stale
value: the log showed `write_data 0x21025cb10 <- 2` followed by `<- 1`, and the guest then span
forever on `cmp [0x21025cb10], 2`.

Fix: `GuestGpu::ThreadRun` selects only `q.front()`; a blocked front no longer lets later
submissions on the *same* queue overtake it. Other queues may still run. This alone took the run
from 34 command buffers / 3 flips to **2852 command buffers / 24+ flips**.

How it was found: a watchdog (`StartSpinWatchdog` in `src/main.cpp`) samples every thread's CPU
time every 10 s and, for any thread burning >50% of a core, suspends it and logs RIP/RSP/registers
plus 64 bytes of code around RIP. It showed one guest thread spinning at eboot+0x13b25f0 in a
2-instruction loop; disassembling the logged bytes gave `mov rsi,[rdx]; cmp rsi,rcx; jne` and the
registers gave the label address and expected value. **Keep this watchdog** — it is the fastest way
to turn "it hangs" into "it spins on address X waiting for Y".

### 8.2 Other fixes this session
- **Depth MIPID**: `DB_DEPTH_VIEW.MIPID` may be non-zero while `DB_Z_INFO.MAXMIP == 0`. A
  single-level Z surface is fully described by its base address and `DB_DEPTH_SIZE_XY`, so a stale
  MIPID selects nothing and is now ignored instead of fatal (`depthRenderTarget.cpp`).
- **`AgcSuspendPoint` must not drain the GPU**: it called `GuestGpu::Done()` →
  `WaitForIdle()`. `sce::Agc::suspendPoint` is the TRC R4089 watchdog call the game is *forced* to
  make from a helper thread; blocking it on GPU idle deadlocks against any queue waiting on a fence
  that the calling thread still has to submit. Added `GuestGpu::SuspendPoint()` (marks the frame
  boundary only). `Done()` keeps the idle barrier because the `GpuCommandLane` test uses it.
- **Stall flush**: when every queue is blocked, the GPU thread now retires the recorded command
  buffer (`CommandScheduler::Finish()`). The priority-operation thread calls `m_master.Wait(tick)`
  *directly*, bypassing `CommandScheduler::Wait`, so a deferred end-of-pipe write whose tick belongs
  to a still-unsubmitted buffer could never land.
- **Fence pulses are latched** (`NoteGuestGpuWrite` / `GuestGpuWriteSatisfied` in graphicsRun.cpp):
  a hardware ring polls its wait address continuously and sees a value even if a later packet
  overwrites it; a queue here only polls while scheduled. Every GPU-side write to guest memory
  (WRITE_DATA, DMA fill/copy, deferred EOP write) is recorded in a 512-entry ring with a sequence
  number, and a failing wait also passes if some recorded write since the submission's baseline
  satisfied it. The baseline is the write sequence at `Enqueue` time, because a real ring starts
  executing at submission, not when our scheduler gets round to it.
- **Blocked is re-checked on any write**: `Submission::blocked_seq` records the write sequence when
  a wait failed; the selector treats a blocked front as eligible again as soon as the global write
  sequence moves. Prevents a satisfied wait from being skipped indefinitely (observed: a submission
  stuck with `value=1 ref=1 mask=0xffffffff func=3`, i.e. already satisfied).
- **`WriteAtEndOfPipeClockCounter[WithWriteBack]` wrote 0** instead of the timestamp.

### 8.3 IMPORTANT: a diagnostic of mine caused a crash
To find where a thread blocked I replaced `PthreadMutexLock`'s blocking `NativeMutexLock(m, nullptr)`
with a 500 ms timed probe plus a fallback. That changed locking semantics and produced IL2CPP object
corruption (a garbage `klass` pointer). **Reverted.** Do not instrument guest mutex acquisition this
way; use the spin watchdog instead.

### 8.4 Current blocker
A **deterministic** access violation while the main menu scene loads, identical registers on every
run:

```
pc = Il2CppUserAssemblies.prx + 0x115f8fb   (module base 0x0980000000)
faulting insn: call qword ptr [rax+0x2C8]     (virtual call)
rdi = r14 = 0x21380add0   (the object)
rax = [r14] = 0x202ca4011  <- odd/unaligned, so not a valid class pointer
target read from [rax+0x2c8] = 0xffffffffffffffff
```
Preceding guest activity is ordinary scene loading: 2 MB direct-memory allocations
(`memory_type=12`) mapped ReadWrite/shared, and graphics pipeline creation. No unresolved import
stub is ever called, and no predicated packet is skipped.

Because the registers are identical run to run, this is a logic bug, not a race: the object at
0x21380add0 is built the same way every time and its first field is wrong. Suggested next steps:
1. Determine whether 0x21380add0 is written by the GPU: `NoteGuestGpuWrite` already records every
   GPU-side write; add a check that logs when a recorded destination falls inside a known IL2CPP
   heap range, or simply log any GPU write to 0x21380a000-0x21380b000.
2. Walk back from the faulting function: the guest stack in the crash dump has return addresses into
   Il2Cpp (`+0x13a27c9`, `+0x24e005`, `+0x24df5e`) and one into the emulator
   (`Posix::pthread_mutex_unlock+6`). Mapping those Il2Cpp offsets to the calling method would name
   the C# subsystem being initialised.
3. The value 0x202ca4011 is odd; check whether some emulated call returns a value the game stores as
   an object header (an allocator or a handle returning a tagged/aligned pointer we get wrong).

### 8.5 Diagnostics currently in the tree (strip before upstreaming)
`STALLDUMP`/`STALLLOOP`/`STALLWATCH`/`SELECT` (graphicsRun.cpp), `WRMPOLL`, `FENCERESET`,
`WAITREGMEM`, `BLOCKED`, `Pm4Trace`, `SPIN` watchdog (main.cpp), `GWAIT`/`GSIGNAL`
(kernel/{semaphore,eventFlag,pthread}.cpp), `EQEVENT` (sync.cpp/renderContext.cpp),
`reserve-fixed replace` (kernel/memory.cpp), `DRAWTIME`/`PREPPHASE`/`RTPHASE`/`XTCPHASE`/`EXPAND`/
`PIPETIME`, `DCB`/`DCBWALK`/`DCBFLIP`/`SUBMITQ` (agc.cpp), `VBTABLE`/`VBRANGE`/`BUFDESC`.
The *logic* in `BUFDESC` (binding a null buffer for a descriptor with `fields[3]==0` or an unreadable
base) is a real fix and must stay; only its log should go.

Keep: the spin watchdog is worth upstreaming behind a config flag.

## 9. Session 2 continued: title screen reached, two remaining failure modes

Verified by capturing the emulator window (script:
`scratchpad/capture.ps1`, run it with **Windows PowerShell**, not pwsh — PowerShell 7 lacks
System.Drawing): the game renders the **BUILDER'S JOURNEY title screen** (logo + brick diorama) at
**25-27 fps**, frame counter climbing past 2400. Measured properly by sampling the window title's
frame counter: 144 frames in 5 s = 28.8 fps. The `fps:` number in the title bar goes stale when the
game stalls, so always diff the frame counter instead of reading `fps:`.

### 9.1 How far the game gets
- Mounts `savedata0`, reads `savedata.bin` (544 bytes) successfully.
- Mounts savedata once per chapter and stats the chapter thumbnails
  (`CM_Beach_Clean.png`, `CM_Hike_Landscape.png`, `CM_Home_House.png`, `CM_Skate.png`,
  `CM_Factory.png`, `CM_Goop.png`, `CM_Home_Car.png`, `CM_Beach_Boat.png`) — that is main-menu
  content being prepared, so the game is past pure engine init.
- Loads plugins PSNCommon, PSNCore, SaveData, Share, libfmod, libfmodstudio, PS5Util,
  Il2CppUserAssemblies. All `sceSysmoduleLoadModule` calls return 0.
- **Never imports or calls any pad library** (no `Pad_v1` in any module's imports, 0 pad calls), so
  it has not reached the input stage. Don't chase controller support yet.
- No unresolved import stub is ever called; no draws are skipped (only 32 benign zero-sized
  dispatches); no predicated packets are skipped.

### 9.2 Two failure modes, roughly 50/50 per run
**(a) Deterministic IL2CPP crash** — identical registers every time:
```
pc  = Il2CppUserAssemblies.prx + 0x115f8fb        (module base 0x0980000000)
insn: call qword ptr [rax+0x2C8]                   (virtual call)
rdi = r14 = 0x21380add0                            (the object)
rax = [r14] = 0x202ca4011                          (odd/unaligned -> not a class pointer)
[rax+0x2c8] = 0xffffffffffffffff                   (the call target)
```
The fault handler now also dumps 64 bytes at rdi/rax/r12 (`runtimeLinker.cpp`), so one crashing run
will show whether that object is freed/poisoned or merely mis-initialised. Ruled out: the texture
cache in-place layer growth (disabling it does not change the crash).

**(b) Compute-queue stall.** Latest, cleanest instance:
```
STALLDUMP: queue=25 sub#0 blocked=1 cursor=0x00006 dw=0x00153 awaited=0x226dbaf50 value=0
           ref=1 mask=0xffffffff func=3 submit_seq=1792
           never written by the GPU (history seq now 1792, baseline 1792)
```
`history seq == baseline` means **no GPU write of any kind has happened since that ACB was
submitted**. The graphics queue is empty, so the only queued work is this blocked ACB, and the
guest will not submit the DCB that would signal it because thread 53 (`Gfx Task Executor`) sleeps on
`Equeue wait: EOP QUEUE, timo=inf` waiting for an end-of-pipe interrupt that only new GPU work can
raise. Interrupt delivery itself is healthy (1695 triggers, every one delivered to both eq 2 and
eq 3 with result=0, and thread 53's final wait comes *after* the last trigger, i.e. it consumed it).

So the cycle is: ACB waits on a fence -> no GPU work -> no EOP interrupt -> guest never submits the
DCB that writes the fence. The open question is why that fence is never written: either the guest
expects a value left over from an earlier frame that we reset, or a DCB submission is being dropped.
Next step: log every `submit_dcb`/`submit_acb` with queue id and dword count *and* the fence
addresses each contains, then check whether the DCB that writes `awaited` was ever submitted at all.

### 9.3 Useful technique
`ReportQueueStall` (graphicsRun.cpp) prints, for every queued submission: blocked flag, suspend
reason, cursor, awaited address with its live value and the wait's ref/mask/func, which other
queued submission references that address, and the GPU write history for it. It is throttled
(1 report per 512 polls, 2000 max). This is what turned "it hangs" into the analysis above — keep it.

## 10. The IL2CPP crash is a single corrupted bit (key finding)

The fault handler now dumps memory at rdi/rax/r12. For the crashing object:

```
mem@rdi (0x21380add0):        <- the C# object, and it is otherwise HEALTHY
 11 40 ca 02 02 00 00 00      [0x00] klass   = 0x202ca4011   <-- odd, must be 0x202ca4010
 00 00 00 00 00 00 00 00      [0x08] monitor = 0             <-- correct for a live object
 50 bf cc 36 02 00 00 00      [0x10] = 0x236ccbf50           <-- plausible heap pointers
 80 01 ce 1e 02 00 00 00      [0x18] = 0x21ece0180
 00 c0 d9 29 02 00 00 00      [0x20] = 0x229d9c000
```

So the object is not freed or poisoned: only **bit 0 of its class pointer** is set. Every
downstream value follows from reading the class one byte off:
`mov rsi,[rax+0x2d0]` returns `0xd00000000202aee3`, which is exactly the correct qword shifted by
one byte, and `call [rax+0x2c8]` then fetches a misaligned target that happens to be
`0xffffffffffffffff`.

Ruled out so far: **none of the PM4 write paths touch that address.** A watchpoint
(`CheckGuestWatch`, enabled with the `KYTY_WATCH_ADDR=0x21380add0` environment variable, hooked into
`CommandProcessor::WriteData`, `CommandProcessor::DmaData` and `Sync::RecordEndOfPipeWrite`) reports
**0 hits** across a full run. Also ruled out: the texture-cache in-place layer growth.

Remaining suspects, in order:
1. Buffer/texture cache write-back into guest memory (a stale GPU copy restored over CPU-written
   bytes). Hook `CheckGuestWatch` into the download/write-back paths
   (`bufferCache.cpp:537` memcpy is already covered via DmaData; the texture download path around
   `TextureCache::BuildDownload` / `TrackImageDownload` is not).
2. `ReplaceFixedRangeWithReserved` in `kernel/memory.cpp` — it unmaps and re-reserves Direct ranges
   and restores backings on failure; a restore could revert a byte. It runs shortly before several
   crashes.
3. The guest itself (would mean we returned bad data from an emulated call much earlier).

Note the address and value are identical on every run, so any hypothesis can be tested in one run
with the watchpoint.

## 11. Current run outcomes (as of end of session 2)

Per run, roughly: title screen renders at 25-29 fps, then either
- the IL2CPP crash above (dominant lately), or
- the compute-queue stall from section 9.2.

`STALLDUMP` now also prints, for a blocked wait, whether any submitted stream ever promised that
write (`NotePromisedFenceWrite`, recorded while snapshotting each DCB/ACB) and the scheduler's
`current_tick`/`gpu_tick`/end-of-pipe queue depth. On the last captured stall the answer was
"promised: yes" with `history seq == baseline` (no GPU write at all since that ACB was submitted),
which is why the end-of-pipe queue state was added — the next run that stalls will show whether a
deferred write is parked on a tick that was never submitted.

## 12. Controller support, and why input is not the blocker

Kyty **already implements the pad**: `src/libs/libPad.cpp` registers `LIB_VERSION("Pad", 1, ...)`
(so imports would appear as `Pad_v1`), `src/libs/controller.cpp` is a full SDL GameController
backend (buttons, sticks, touchpad, accel/gyro), and `src/graphics/presentation/window/hostInput.cpp`
maps configurable keyboard/mouse bindings onto pad buttons by name (Cross, Circle, Square,
Triangle, ...), sharing the player-1 pad with the gamepad.

PPSA05496 **never imports or calls any of it**: zero `Pad_v1` imports across every loaded module and
zero `Controller::` symbol resolutions in a full run. So no amount of input work can advance this
game right now - it has not reached the stage where it opens the pad. Revisit input only after the
game gets past loading.

## 13. GPU-assisted validation: compute shader LDS data race (likely root cause area)

Running with `--gpu-assisted-validation true` produced a real shader error within 10 s:

```
vkCmdDispatch(): A data race was detected on the shared memory variable "lds_dwords"
in local invocation index 0 while performing a load or store operation.
The other access in this race was at: OpAtomicOr
Stage = Compute.  Global invocation ID (x, y, z) = (7040, 0, 0)
Compute Dispatch Index 2, Shader Module internal ID 50
```

Context that makes this important:
- This host is **wave32 only**: the log prints
  `Vulkan subgroup: default=32 min=32 max=32 stages=0x3fff size_control=false wave64=false`.
- PS5 compute shaders are **wave64**. The recompiler emulates them by giving each host invocation
  two logical lanes: `SpirvEmitter.cpp:324` sets `state.lane_count = 2` when
  `program.wave_size == 64 && host_subgroup_size == 32`.
- `s_barrier` *is* translated correctly (`Translator::S_BARRIER` -> `IR::ValueOpcode::Barrier` ->
  `OpControlBarrier` with workgroup scope and `AcquireRelease | WorkgroupMemory` semantics), so the
  race is not a dropped barrier - it is LDS ordering that wave64 hardware provides implicitly and
  the two-lanes-per-invocation emulation does not.

Why this is the best remaining lead for both failure modes:
- Wrong compute results would explain why the game's loading never finishes even in runs that
  neither crash nor stall.
- It also explains the corruption in section 10 that **no CPU-side watchpoint could catch**. Every
  emulator path that writes guest memory was instrumented and reported **zero hits** on the
  corrupted address: `CommandProcessor::WriteData`, `CommandProcessor::DmaData`,
  `Sync::RecordEndOfPipeWrite`, and `LibKernel::Memory::WriteBacking` (which covers both the buffer
  cache and texture cache read-back paths). The four `reserve-fixed replace` ranges in that run also
  do not contain the address. That leaves the GPU itself writing guest memory - i.e. a shader
  storing through an address computed from wave-dependent data, which is exactly what a wave64/wave32
  mismatch would corrupt, and matches a single stray low byte rather than a bulk overwrite.

Suggested next steps:
1. Re-run with `--gpu-assisted-validation true` and collect *all* races/OOB reports, not just the
   first (the emulator currently treats a validation error as fatal at
   `vulkanWindow.cpp:838` - make it log-and-continue for triage).
2. Audit the `lane_count == 2` path for LDS and cross-lane ops, and add explicit workgroup memory
   barriers where wave64 hardware would have implicit ordering.
3. If an AMD RDNA GPU is available, test there: `wave64=true` would skip the emulation entirely and
   would confirm or eliminate this whole class in one run.

## 14. BLUEPRINT: from title screen to in-game (read this first if you are continuing)

### 14.1 The proven root cause (not a lead any more)

The GPU-assisted-validation race is reproducible and fully characterised. Re-run with
`--graphics-debug-dump true --gpu-assisted-validation true` and the emulator exits at the same spot
(`Compute Dispatch Index 2`, `OpAtomicOr %1689` vs `OpLoad %1729`). Disassembling every dumped
compute module (`spirv-dis _Shaders/*_cs_*.spv`) shows exactly one contains `OpAtomicOr`:

    _Shaders/0028_new_shader_cs_250f003662660c85.spv   (GCN hash 0x250f003662660c85, 186 instr)

Its SPIR-V has `OpExecutionMode %main LocalSize 32 1 1` (so it is the `lane_count == 2` path:
64 guest threads on 32 host invocations), `lds_dwords` is 128 dwords, and the module contains
**no `OpControlBarrier` at all**. The original RDNA2 code (`_Shaders/original/...250f...bin`,
scanned by encoding) has:

    word   8  ds_write_b32       zero the LDS bitmask
    word  13  s_waitcnt lgkmcnt(0)
    ...       4 unrolled iterations of buffer loads guarded by s_cbranch_execz ...
    word 258  ds_or_b32          each lane ORs its bit into the LDS bitmask
    word 267  ds_read_b32        each lane reads the whole bitmask back
    word 273  s_waitcnt lgkmcnt(0)
    word 276  s_endpgm
    (no s_barrier anywhere)

On real hardware this is correct without `s_barrier`: LDS instructions issued by one wave execute
in order for the whole wave, so every lane's `ds_or_b32` has completed before any lane's
`ds_read_b32` issues. The recompiler maps that wave onto 32 host invocations and emits nothing
between the atomic and the load (`EmitWaitcnt = EmitVoid`, `spirvEmitterInstructions.h:151`);
Vulkan gives no intra-subgroup ordering, so invocation 0's load races the other invocations' ORs.
The loaded value (`%1747`) is then `OpStore`d to a storage buffer (`%buffers[1]`, at `%1769`),
i.e. the shader's *output* is a racy bitmask that the CPU/next dispatch consumes. That is why
the title screen renders but the scene load never completes: the wrong bits mean wrong culling /
wrong indirect work, in a game whose loading is GPU-driven.

This is a general bug in the wave emulation, not specific to this shader: any wave-relative
LDS pattern without `s_barrier` (very common in Unity / HDRP compute) is broken on wave32 hosts.

### 14.2 The fix (shader recompiler, ~40 lines)

Rule: **a guest wave's LDS operations must appear in order to every lane of that wave.** A guest
wave is exactly one host subgroup (`lane_count == 2` packs 64 lanes into 32 invocations; wave32
programs run on a 32-wide subgroup), so the ordering scope is `Subgroup`.

Implementation point: `spirvEmitterProgram.cpp`, `EmitBlock()` (line ~321). Before the
`for (half ...)` loop, i.e. at the instruction level where control flow is still wave-uniform:

```cpp
if (NeedsWaveLdsOrdering(ctx.state, inst)) {
    // Guest LDS ops execute in wave order; the host subgroup must observe the same order.
    ctx.state.builder.AddFunction(spv::OpControlBarrier,
        ConstantU32(ctx.state, spv::ScopeSubgroup), ConstantU32(ctx.state, spv::ScopeSubgroup),
        ConstantU32(ctx.state, spv::MemorySemanticsAcquireReleaseMask |
                               spv::MemorySemanticsWorkgroupMemoryMask));
}
```

`NeedsWaveLdsOrdering(state, inst)`:
- the program has a workgroup (`ShaderWorkgroupInput(...) != nullptr`, compute/mesh) - for other
  stages LDS is a `StorageClassFunction` private array and needs nothing;
- `inst` is an LDS access: opcodes `LoadShared*`, `WriteShared*`, `SharedAtomic*`
  (`ValueOpcodes.inc:244-269`) - or, more generally, `ctx.Memory(inst).kind == IR::ResourceKind::Lds`
  (how `EmitAtomic32` in `spirvEmitterMemory.cpp:898` already tells LDS from buffers);
- optional optimisation: only barrier before a *load/atomic* that follows a *store/atomic* since
  the last barrier - keep a `bool lds_dirty` in `EmitterState`, set on `WriteShared*` /
  `SharedAtomic*`, cleared by the emitted barrier and by `ValueOpcode::Barrier`. Start without it;
  add it only if a profile shows the barriers matter.

Why the placement is legal: the emitter guards every LDS op with a per-lane exec branch
(`OpBranchConditional %1659` where `%1659` is the lane's exec bit, see the dump around line 1855);
that branch is *non-uniform*, so the barrier must be emitted **before** the emitter enters the
instruction, which is what the `EmitBlock` hook does. Structured block branches are on subgroup
ballots of EXEC/VCC/SCC (`BranchCondition()`, `spirvEmitterProgram.cpp:110-122`), so the
instruction level is subgroup-uniform. The only early exit is `EmitKillIfBoolFalse` (`OpKill`,
line 16), which is pixel-only. `OpControlBarrier` with `Subgroup` execution scope is valid in
Vulkan compute, and NVIDIA lowers it to `bar.warp.sync` (cheap). The `GroupNonUniform`
capability is already required on the `lane_count == 2` path (`spirvEmitterModule.cpp:639`).

Do **not** implement it via `Waitcnt` instead: this shader's `ds_or` -> `ds_read` has no waitcnt
between them; the hardware guarantee is per-instruction, not per-waitcnt.

Do not gate it on `lane_count == 2` only: wave32 guest programs on a 32-wide host subgroup have
the same hole (Vulkan does not promise lockstep inside a subgroup either). Gate on "compute/mesh
stage with LDS", period. On AMD (native wave64) the barrier is a no-op in practice.

Test: `tests/ShaderRecompilerComputeTests.cpp` (`CompileCase(test, host_subgroup_size)`,
`TestCase::required_spirv`). Add a case: a wave64 program with `ds_or_b32` followed by
`ds_read_b32` and no `s_barrier`, compiled with `host_subgroup_size = 32`, requiring an
`OpControlBarrier` with execution scope Subgroup (constant 3) between `OpAtomicOr` and the LDS
`OpLoad`; and the same program at `host_subgroup_size = 64` (still required, see previous
paragraph) so the rule is documented by the test.

### 14.3 Verification ladder (each step is a pass/fail gate)

1. **Build + unit test** (section 2 commands, plus `ctest` for `ShaderRecompilerComputeTests`).
2. **GPU-AV must be clean for the same 10 s window**:
   `--gpu-assisted-validation true --graphics-debug-dump true`. Before that, make the debug
   messenger non-fatal so *all* remaining reports come out in one run: `vulkanWindow.cpp:837`
   `EXIT_COLOR(...)` -> keep the `LOGF_COLOR` and only `EXIT` when env `KYTY_VALIDATION_FATAL=1`
   is set. Run 3 minutes; grep `_kyty.txt` for `data race|out of bounds|Descriptor index`. Every
   remaining hit is a new bug to fix the same way (look at the module id, disassemble, map to the
   GCN pattern).
3. **Behavioural gate**: normal run (no validation), 120 s, fullscreen capture every 10 s with
   `scratchpad-capture.ps1 -Screen` (repo root) (Windows PowerShell, not pwsh) and `scratchpad-imagediff.ps1`. Success =
   the frame after the title screen changes to chapter-select / first level (the loading
   previously never completed). Watch `_kyty.txt` for the loading thread: previously it ended in
   either the IL2CPP fault (section 10, `rdi`/`rax` dump from `runtimeLinker.cpp`) or the
   compute-queue `STALLDUMP` (section 11). If neither appears within 120 s and frames keep
   changing, we are in.
4. **If the IL2CPP single-bit corruption still occurs** after 14.2: the only GPU->guest path not
   covered by the `KYTY_WATCH_ADDR` watch is the direct `std::memcpy` in `BufferCache::CopyBuffer`
   (`bufferCache.cpp:537`, guest-to-guest DMA fast path); image downloads (`textureCache.cpp:1909`)
   go through `WriteBacking` and *are* covered. Add `CheckGuestWatch(dst_vaddr, size,
   "buffer_copy")` at 537 and re-run with `KYTY_WATCH_ADDR=0x21380add0`. A hit there means a
   DMA_DATA / CP DMA with a bad destination - look at the PM4 packet that issued it
   (`pm4Handlers.cpp`, `DmaData`).
5. **If the compute-queue fence stall still occurs**: it is the *other* failure mode and is
   independent of the shader race. `STALLDUMP` reports `promised: yes (we lost it)` with
   `history seq == baseline`, i.e. the RELEASE_MEM was in the snapshot but its EOP write never ran.
   The EOP write is a `DeferPriorityOperation` gated on `m_master.Wait(tick)`; the stall_flush path
   in `GuestGpu::ThreadRun` calls `scheduler.Finish()` when every queue is blocked. Instrument
   `CommandScheduler::DeferPriorityOperation` to log `tick` vs `KnownGpuTick()` at enqueue and at
   completion, and `Finish()` entry/exit, then correlate with the STALLDUMP timestamp. The two
   candidates are (a) the compute slice was recorded into a command buffer that was never
   submitted to Vulkan (needs a `Flush()` after every ACB slice, not only on stall), or (b) the
   RELEASE_MEM handler took an "event type not handled" path and never reached
   `RecordEndOfPipeWrite` (log every RELEASE_MEM `event_type`/`data_sel` on the compute queue).

### 14.4 Cheap control experiment

Any AMD RDNA GPU reports `wave64=true`; there `lane_count == 1`, the subgroup is a real wave64
and this whole class disappears. One run on such a machine tells you whether anything *other*
than the wave emulation still blocks gameplay. Not required - 14.2 + 14.3 is the actual work.

### 14.5 Uncommitted-tree hygiene before the fix goes in

Section 6 lists which diagnostics to strip and which fixes to keep. Commit in this order, one
line each, no trailer: (1) AGC submit snapshot + in-order queue scheduler + non-blocking
`SuspendPoint`; (2) depth MIPID / null V# / EOP clock-counter fixes; (3) image layer growth;
(4) the LDS subgroup ordering fix + its test. Keep the env-gated non-fatal validation switch as a
real feature; strip the STALL*/GWAIT/GSIGNAL/SPIN logging.

## 15. Session 3: what was fixed, and two earlier conclusions that were WRONG

Read this before acting on sections 10-14: it corrects them with direct evidence.

### 15.1 Landed: the wave64 LDS ordering fix (section 14.2 implemented)

Implemented as specified in 14.2:
- `spirvEmitterInternal.h`: `SpirvRequirements::wave_lds_ordering`, plus `lds_pending_read` /
  `lds_pending_write` on `EmitterState`.
- `SpirvEmitter.cpp` (`AnalyzeProgramRequirements`): sets `wave_lds_ordering` when a compute/mesh
  program has a workgroup-LDS access that is not a pure read.
- `spirvEmitterProgram.cpp`: `NeedsWaveLdsOrdering()` + `EmitWaveLdsBarrier()`, called from
  `EmitBlock()` before the per-half loop. Hazard tracking is exact inside a block (RAW, WAR, WAW
  on LDS) and resets conservatively at each block boundary, because a predecessor or a back edge
  may have executed any access. `ValueOpcode::Barrier` (a real `s_barrier`) clears the pending state.

The emitted barrier is `OpControlBarrier Subgroup Workgroup (AcquireRelease|WorkgroupMemory)`.
Execution scope is **Subgroup**, not Workgroup, because a host workgroup can contain several guest
waves (`spirvEmitterModule.cpp:679` packs 64 guest threads into 32 invocations, so a 128-thread
guest workgroup is two subgroups) and those waves diverge at IR-block granularity. A workgroup-scope
barrier there would be a hang, not a fix.

Verified in the emitted SPIR-V: compute module `0x250f003662660c85` went from **0** barriers to 3,
with one correctly placed between the racing `OpAtomicOr` and the LDS `OpLoad`, at the IR-block top
level (uniform control flow), not inside the per-lane exec guard.

Placement is safe on both emission paths: structured block terminators branch on subgroup ballots of
EXEC/VCC/SCC (`BranchCondition`, `spirvEmitterProgram.cpp:110-122`), and the dispatcher fallback's
`pc` is either a ballot-derived `OpSelect` or a scalar (wave-uniform) indirect target
(`EmitDispatcherNextPc`). So an IR block is entered by all invocations of a subgroup or none.

### 15.2 GPU-assisted validation still reports these races - it is a validator limitation

After the fix, GPU-AV still reports `lds_dwords` races at the same instruction pairs. That is a
false positive: **GPU-AV only clears its shared-memory tracking on a workgroup-scope
`OpControlBarrier`**. Evidence from one run with all LDS modules dumped and disassembled:

| module | workgroup barriers | subgroup barriers | flagged by GPU-AV |
|---|---|---|---|
| `3113889d` | 1 | 27 | no |
| `37a6cefc` | 2 | 6 | no |
| `250f0036` | 0 | 3 | **yes** |
| `7529b484` | 0 | 5 | **yes** |

Every flagged module has only subgroup-scope barriers; every unflagged LDS module contains at least
one real `s_barrier`. In `7529b484` three subgroup barriers sit between the two accesses GPU-AV
pairs up. Do not "fix" this by widening the scope to Workgroup - see 15.1 for why that deadlocks.

### 15.3 WRONG in section 10/13: the IL2CPP corruption is not GPU corruption

Section 13 concluded the stray low bit in the klass pointer had to come from the GPU, because no
CPU-side write path hit it. That is false. A hardware data breakpoint (`KYTY_HW_WATCH`, below)
catches the writer directly, and it is **the game itself**:

```
HWWATCH: hit address=0x21380add0 rip=0x980246209 value=0x0000000202ca4011 os_thread=11340
   code before rip:  ... 4c 89 31   41 80 0e 01     <- mov [rcx],r14 ; or byte ptr [r14],1
HWWATCH: hit address=0x21380add0 rip=0x980245928 value=0x0000000202ca4010 os_thread=11340
```

`or byte ptr [r14], 1` - a non-atomic mark bit that Unity sets and clears around an object-graph
traversal. Both RIPs are inside `Il2CppUserAssemblies.prx` (base `0x980000000`). The preceding
instructions push the object onto a work list, i.e. a mark-and-sweep. Right before it, the main
thread signals a burst of semaphores that wake threads 4-16, which are named
`AssetGarbageCollectorHelper`: this is **Unity's asset garbage collector**, not the GPU.

The crash is a cross-thread race on that mark bit, confirmed by thread ids:
- marker = `os_thread=11340`, the **main thread** (no `Pthread run begin` record).
- faulter = `os_thread=21380` = guest thread id 89, one of many unnamed threads sharing entry
  `0x9801fb830` in `Il2CppUserAssemblies.prx`, i.e. an **IL2CPP managed thread**.
- In a crashing run the last watch hit is the *set* with no matching clear: the fault lands inside
  the marked window, and `mem@rdi` still shows `11` in the fault dump.

So a managed thread reads `klass` while the asset GC has bit 0 set, then virtual-calls through the
tagged pointer. On real hardware that window is not observable by managed code. The emulator bug is
therefore whatever lets a managed thread run during Unity's asset GC - **not** a stray GPU write.

Ruled out while chasing this (all verified, do not redo): the pthread mutex implementation
(`NativeMutexLock/Unlock`, owner tracking and recursion are correct), the condvar implementation
(per-thread `cond_sequence` predicate, no spurious-wake bug, correct unlock ordering), the
semaphore implementation (`KernelSemaPrivate::Signal/Wait` - count persists, no lost signal), and
the clock (`KernelGetProcessTimeUsNative` - calibrated invariant TSC, monotonic).

### 15.4 WRONG in section 12: the game DOES import libScePad

Section 12 said the game imports zero pad functions, based on `grep Pad` over the log returning 0.
That test is invalid: `grep VideoOut` over the same log also returns 0, and video out obviously
works - Kyty simply does not log a relocation line for every library.

The eboot's imported-module table contains `libScePad.prx libScePad`, between `libSceAudioIn` and
`libSceIme`:

```
$ dd if=eboot.bin bs=1 skip=30377100 count=900 | tr '\0' '\n'
  ... libSceAudioIn.prx libSceAudioIn libScePad.prx libScePad libSceIme.prx libSceIme ...
```

However, the game never calls `scePadOpen` in any run so far, and sending keystrokes (J=Cross,
L=Circle, Enter=Options per `hostInput.cpp:210`) to the focused window at the title screen changes
nothing. Input is not the current gate - the game has not yet reached input initialisation.

### 15.5 The actual blocker, precisely located

The game is **not** hung: it renders at ~26 fps, FMOD mixer/feeder/stream threads cycle, Job.Workers
cycle, the GfxFlipThread and the main thread trade semaphores every frame, and the asset GC runs to
completion. It has already opened `level0`, `level1` **and `level2`**, plus every FMOD bank.

One thread is stranded. `Loading.PreloadManager` (guest thread 56) blocks here and never runs again:

```
line  396742:  GWAIT: sema=0x1003d482090 done thread=56      <- main let it proceed
line  396743:  GWAIT: sema=0x1003d482720 need=1 timeout=inf thread=56
                  ... 3,817,039 more log lines / ~117 s, thread 56 never appears again ...
```

That semaphore has exactly 7 events in the whole run: the loader waits, **thread 3 (main) signals**,
loader wakes - twice - and then the third wait is never answered. This is Unity's PreloadManager
integration handshake: the loading thread hands work to the main thread and waits for the main
thread to finish integrating it. The main thread keeps rendering but stops driving that state
machine.

This is the thing to fix to reach gameplay. The asset GC completed immediately before (helper
threads 4/5/16 go quiet at lines ~396539-396552, the loader blocks at 396743), which is consistent
with the 15.3 race having left engine state wrong; the ~50% crash and this stall are plausibly the
same root cause with different timing.

Confirmed *not* the cause: FMOD (all five threads cycling), audio device (opened fine), the asset GC
helpers (they finish), image layer growth (`EXPAND` events continue normally after the stall), guest
file I/O (nothing is pending), unresolved import stubs (175 exist, **none is ever called**), and
Unity native-plugin `dlsym` misses (`UnityPluginLoad` etc. are optional and absent on every plugin).

Suggested next step: find why the main thread stops calling the preload integration. Add the guest
return address to the `GSIGNAL`/`GWAIT` logs to recover the engine code address that signalled
`0x1003d482720` the two times it worked, then breakpoint that address and see which frame stops
reaching it. The game's own `Debug.Log` output is not available: Kyty implements no
`sceKernelDebugOutText` and no `/devlog`, and the game stats `/devlog` at boot and stays silent.

### 15.6 New tooling added this session

- **`KYTY_HW_WATCH=<hex addr>`** (`main.cpp`, next to `StartSpinWatchdog`): arms an x86 data
  breakpoint (DR0, write, 8 bytes) on that 8-byte-aligned address in every thread, re-arming new
  threads every 200 ms, and logs `HWWATCH:` with the writing instruction's rip, the value written,
  the os thread, full registers, code bytes and stack. This is what settled 15.3, and it catches any
  writer regardless of path - far better than adding `CheckGuestWatch` call sites one at a time.
  Note: only *writes* are armed on purpose. Watching reads makes the handler's own read of the slot
  retrigger the breakpoint recursively and hard-crashes the process.
- **`KYTY_VALIDATION_NONFATAL=1`** (`vulkanWindow.cpp`): a Vulkan validation error is logged and
  execution continues, so one run reports every problem instead of stopping at the first. This took
  the GPU-AV run from 1 reported error to 20 and is what made the table in 15.2 possible.
- The fault handler now prints the faulting `os_thread`, which is what identified the racing reader.

### 15.7 Test suite state

`shader_recompiler_compute_tests` had two **pre-existing** failures from session 2's uncommitted
work; both are fixed and both are still uncommitted along with the rest of that work:

- `RenderExecutorStencilBindingDiscovery` - `FindBuffers` treated `descriptor.fields[3] == 0` as a
  placeholder descriptor and bound a null buffer. A legitimate V# can have dword3 zero (the test
  builds one), so a clamped range was lost. The base-readability probe already covers the real case
  the game hits, so the `fields[3] == 0` term was dropped.
- `CubeFaceStorageExpansion` - `Image::Image` set `backing.layers` from `create.arrayLayers`, which
  is now `max(guest layers, reserved capacity)`. Everything else treats `backing.layers` as the
  guest layer count (transitions, copy bounds), so an expanded image reported too many layers.
  `backing.layers` is now the guest count, capacity stays in `capacity_layers`, and `GrowLayers`
  updates it.

A third failure remains and is **not** from this session: the suite access-violates in the case
after `GraphicsBranchPathFinalVmExportDiscardsInactiveExec`. Verified pre-existing by rebuilding
with the LDS barrier disabled - it crashes at exactly the same point. The two fail-fast failures
above had been masking it, so nothing past them had ever run with this tree.

Committed this session (single-line, on `feat/depth-to-color-copy`):
- `3dc571c` Order a guest wave's LDS accesses with a subgroup barrier on wave32 hosts
- `9564166` Allow Vulkan validation errors to be non-fatal via KYTY_VALIDATION_NONFATAL

The rest of the working tree is still session 1/2's uncommitted work, now including the two test
fixes above, the `KYTY_HW_WATCH` watchpoint in `main.cpp` and the faulting-thread id in
`runtimeLinker.cpp`.

## 16. PERFORMANCE: why gameplay is capped at ~28 fps, and the plan to lift it (research only)

Gameplay reached in session 3 (commit `ab6a394`). Splash/title run at 60+ fps; in-game sits at
27-29 fps on an i5-12400F + RTX 3070. This section is research for the implementing session: it
states what is measured, what is ruled out, what the bottleneck must be, and a measurement-first
plan. Nothing here has been implemented.

### 16.1 Measurements (all from `_kyty_perf.txt`, gameplay, `KYTY_REPORT_FPS=1`)

`KYTY_REPORT_FPS=1` (new, `window.cpp`) prints `FPS: <n> frame=<m>` to stdout once a second. Use
it, not the window title: the title is set via `RunOnMainThread` and stalls while that thread is
busy, which is what made the earlier "frozen at 863/979 frames" readings wrong.

| experiment | fps | conclusion |
|---|---|---|
| default (1280x720, logging to file) | 28.1 | baseline |
| `--printf-direction Silent` | 27.4 | logging is not it (LOGF short-circuits when silent) |
| 640x360 | 27.5 | not GPU fill / rasterisation |
| 1920x1080 | 28.4 | same |
| `--vblank-frequency 240` | 27.8 | not vblank quantisation (the earlier "vblank 120 = 59.8" was measured on the splash, not gameplay - disregard it) |
| `--present-mode Immediate` | 37.5 avg / 60 max | mixed splash+gameplay window, unreliable |
| process CPU during gameplay | 113.8 % of one core = 9.5 % of 12 | not CPU-throughput bound |
| spin watchdog (>50 % of a core) | one host thread at 50-60 %, sampled inside system DLLs | no saturated thread |
| GPU timeline lag (`ORDER: priority op waiting tick=T current=C known_gpu=G`) | G is 1-3 ticks behind C, with ~24 ticks/frame | GPU keeps up; no backlog |
| per-frame readbacks / image expands / copies / GC runs | 0 | no GPU->CPU round trips per frame |
| `SetFlipRate` | always 0 (60 Hz) | the game asks for 60 |
| pipelines created during gameplay | 0 (last `PIPETIME` at 7 s) | no per-frame pipeline builds |
| slow-draw probes (`DRAWPHASE`/`PREPPHASE` > 2 ms) | ~0 per second | per-draw CPU work is cheap |

So: not GPU, not CPU throughput, not logging, not vsync, not pipeline/shader compiles, not
readbacks. The ~35 ms frame is **latency spent in hand-offs between threads inside the emulator**.
That is also why the 3070 looks idle: nothing is saturated, everything is waiting on something else.

### 16.2 What a gameplay frame consists of (counts per second at 28 fps, i.e. per frame = /28)

| event | per second | per frame |
|---|---|---|
| `sceAgcDriverSubmitDcb` (`SUBMITQ`) | ~617 | **~22 DCB submissions** |
| `AgcWaitRegMemPatchAddress` (`cmd/address` pairs) | ~11,500 | ~400 (cheap: no logging when silent, `PRINT_NAME` is off) |
| event-flag waits (`GWAIT: eventflag`) - 13 `Job.Worker` threads | ~3,800 | ~135 (about 10 per worker) |
| event-flag sets (`set=0xffffffffffffffff`, broadcast) | ~330, from main (3) and `UnityGfxDeviceWorker` (52) | ~12 broadcasts that wake all 13 workers |
| semaphore waits / signals | ~545 | ~19 hand-offs (main <-> GfxDeviceWorker <-> GfxFlipThread <-> Gfx Task Executor) |
| EOP queue waits | 58 (29 `timo=0` polls that time out + 29 `timo=inf` blocks) | 1 poll + 1 blocking wait for the previous frame's GPU fence |
| flip event waits (`GfxFlipThread`, `timo=500000`) | 29 | 1 |
| `EndOfPipe Signal` | 60 | 2 |
| `usleep(10000)` | ~180, only threads 72/79 (managed pollers) | irrelevant |
| yields | 2-5 | irrelevant |

Every one of those is a host mutex + condition-variable transition. On the PS5 they are ~1 us
syscalls on 7 dedicated cores; here each is tens of microseconds of Windows scheduling latency, and
many are on the frame's critical path (a broadcast that wakes 13 workers, then the main thread waits
for all of them; a semaphore ping-pong between four render threads; 22 serial submissions each of
which wakes the GuestGpu thread and ends in a `vkQueueSubmit`).

### 16.3 Code facts relevant to the bottleneck (verified, do not re-audit)

- `CommandProcessor::BufferFlush()` -> `CommandScheduler::Flush()` -> `Submit()` -> `vkQueueSubmit`
  is called **after every DCB/ACB submission that made progress** (`graphicsRun.cpp:1057,1083`),
  after every RELEASE_MEM data write (`pm4Handlers.cpp` `data_sel==1` path), and on every flip
  (`graphicsRun.cpp:2047,2067,2093,2108`). With ~22 submissions + 2 EOPs + 1 flip that is
  **~25 `vkQueueSubmit` per frame**, each one a timeline-semaphore signal under `queue_mutex`.
  The GPU timeline confirms this (~24 ticks per frame).
- `GuestGpu::ThreadRun` waits for work on `m_work_available` (a condvar; no polling) and processes
  submissions strictly in order (`q.front()` only - required, see 3.x). There is a `SleepMicro(1000)`
  only on the stall path (`graphicsRun.cpp:944`), not hot.
- The EOP write and the guest interrupt are `DeferPriorityOperation`s run by one priority thread
  that waits `m_master.Wait(tick)` = `vkWaitSemaphores` (blocking, not polling) - fine.
- Wait primitives: `KernelEventFlagPrivate::Wait` blocks on `m_cond_var.Wait` and `Set` does
  `SignalAll` under the same mutex - prompt wake, no lost wake-up. Semaphores, mutexes and condvars
  likewise notify promptly; the `SIGNAL_APC_POLL_MICROS = 10000` (10 ms) is only a fallback poll for
  signal delivery. `Common::Mutex` is a `CRITICAL_SECTION` with spin count; `CondVar` is a Win32
  condition variable. `CondVar::WaitFor` rounds to whole milliseconds (`micros/1000`, min 1) and
  `timeBeginPeriod` is never called, so any *timed* wait that expires has system-tick granularity.
- Guest thread affinity is stored only (`PthreadSetaffinity` -> `attr`), never applied to the host;
  priority maps to sched +2/0/-2. Neither is obviously harmful.
- `SleepMicro` uses `CREATE_WAITABLE_TIMER_HIGH_RESOLUTION` - accurate.
- Tracy is compiled in: `--profile`, client 0.14.1, port 8086, 16 files carry `KYTY_PROFILER_*`
  zones (scheduler, pipeline cache, draws). A capture needs the Tracy `tracy-capture` CLI or GUI.

### 16.4 Hypotheses, ranked

**H1 - hand-off latency chain (most likely).** The Unity frame is a dependency chain of hundreds of
cross-thread wake-ups (16.2). Windows wake-up latency of a blocked thread is ~10-50 us when the
target core is idle and far worse when it must be woken from a deep C-state or another thread has
to be preempted; with 13 workers woken by one broadcast and the main thread then waiting for the
slowest, each job-system phase costs the *worst* wake. ~12 broadcasts + ~19 semaphore hops + ~22
submission hops per frame at 0.5-1.5 ms each is exactly a 20-35 ms frame with idle CPUs.
Prediction: the timeline in 16.5 shows the main thread blocked in event-flag/semaphore waits for
most of the frame while workers are mostly idle.

**H2 - GuestGpu thread serialisation.** ~22 submissions per frame are processed one at a time by one
thread, each ending in a `vkQueueSubmit`; the guest's `SubmitDcb` returns immediately, but the
frame's fence (EOP) can only be signalled after the last of them has been recorded, submitted and
retired. If recording + 25 submits costs ~15-20 ms on that thread (it was the one thread at
50-60 %), the blocking EOP wait each frame is where the main thread loses the rest of its frame.
Prediction: the timeline shows the `Gfx Task Executor` EOP wait returning ~15+ ms after the last
`SubmitDcb`, and the GuestGpu thread busy for most of that interval.

H1 and H2 are not exclusive; the instrumentation below apportions the frame between them.

Ruled out (16.1): GPU throughput, resolution, logging, vsync/vblank, flip rate, pipeline compiles,
readbacks, per-draw executor cost, yields, usleeps.

### 16.5 Plan for the implementing session

Step 1 - measure before touching anything (half a day, all in-tree, cheap when silent):
1. Add a per-frame critical-path timeline, QPC timestamps, printed once per frame to stdout behind
   `KYTY_FRAME_TIMELINE=1` (same style as `KYTY_REPORT_FPS`): (a) main thread first `SubmitDcb` of
   the frame, (b) last `SubmitDcb`, (c) GuestGpu thread start/end of processing per submission
   (accumulate busy time), (d) `vkQueueSubmit` count, (e) GPU tick of the frame-ending EOP reached
   (`MasterSemaphore::Refresh`), (f) EOP delivered to guest (`TriggerInterrupt`), (g) guest EOP wait
   returned (`KernelWaitEqueue` on "EOP QUEUE"), (h) flip queued, presented, flip event delivered,
   (i) for each event-flag broadcast: time from `Set` to the last of the 13 workers resuming, and
   time until main resumes from its following wait. Output ~10 numbers per frame.
2. Take one Tracy capture of ~10 s of gameplay (`--profile`, `tracy-capture -o gameplay.tracy
   -a 127.0.0.1 -p 8086`, tool from github.com/wolfpld/tracy `capture/`). The zones already exist in
   the scheduler, pipeline cache and draw paths; add zones to `GuestGpu::ThreadRun` submission
   processing, `submit_dcb`, `KernelSetEventFlag`, `KernelWaitEventFlag`, `KernelSignalSema`,
   `KernelWaitSema`, `KernelWaitEqueue` if missing.
3. Decide from (1)/(2) whether the frame is main-thread-blocked-on-workers (H1) or main-blocked-on-
   EOP (H2), and how many ms each hand-off class costs.

Step 2 - fixes, in the order they are likely to pay, each measured with `KYTY_REPORT_FPS` before
and after (target: >30 sustained; realistic ceiling on this chain is 45-60):

A. **Batch Vulkan submissions** (H2, low risk, 2-6 ms): stop calling `BufferFlush()` after every
   submission that made progress (`graphicsRun.cpp:1057,1083`). Flush only when something needs the
   tick: an EOP/RELEASE_MEM write, a WAIT_REG_MEM that another queue depends on, a flip, a queue
   switch, or a command-buffer size threshold. The in-order fence semantics (3.x) are unaffected:
   the deferred EOP op still waits for its tick; it just gets a later, larger tick. Watch the
   compute-queue stall path: a compute WAIT_REG_MEM on a graphics fence needs the graphics work
   *submitted*, so keep a flush when a queue blocks (the `stall_flush` path already does this).
B. **Cut wake latency on the hot primitives** (H1, medium risk, potentially the big one): add a
   short bounded spin (e.g. up to 50-100 us, `_mm_pause`, only while the process has idle cores)
   before blocking in `KernelWaitEventFlag`, `KernelWaitSema` and `PthreadCondWait`, and make
   `KernelSetEventFlag` / `KernelSignalSema` wake without holding the mutex across the wake.
   Game engines' job systems assume sub-10 us hand-offs; this is the standard emulator answer.
   Measure 13-worker broadcast-to-all-resumed time before/after with the timeline from step 1.
C. **Raise the timer resolution once at startup** (`timeBeginPeriod(1)` or
   `NtSetTimerResolution`) so every `CondVar::WaitFor`/`SleepConditionVariableCS` timeout and every
   Win32 wait has 1 ms rather than 15.6 ms granularity. Cheap, safe, helps every timed wait.
D. **Process-level scheduling**: set the emulator process to `ABOVE_NORMAL_PRIORITY_CLASS` and map
   the guest's render/main thread priorities (`PthreadSetprio` +2) to `THREAD_PRIORITY_HIGHEST` on
   the host. Reduces preemption of the critical path by the 13 workers. Cheap; measure.
E. **Drop the 10 ms signal-poll wake-ups** (`SIGNAL_APC_POLL_MICROS`) from the infinite waits when
   no signal is pending (they are only needed for the GC's `sceKernelRaiseException`, which now
   also wakes the target explicitly via `PthreadWakeForSignal`). ~40 threads x 100 spurious
   wakes/s of CPU; small win, low risk.
F. Only if H2 dominates after A: overlap recording and submission (record submission N+1 while
   N's `vkQueueSubmit` happens on another thread) - larger change, not needed for 30 fps.

Do not: change the in-order queue policy (it fixed the fence bug in 3.x), widen the LDS barrier
scope (15.2), or touch `WaitForSignalDispatch`/the APC path (15.x, it is what reaches gameplay).

### 16.6 Second title: PPSA05684 fails at boot in the shader recompiler

`C:\Users\Ericsson\Downloads\[PS5DUMPID]-PPSA05684\PPSA05684-app0` exits with

```
shader resource tracking: hash=0xf9a0dc71344fec27 stage=compute pc=0x000002c8
GetBufferResource dword 0 is not a valid runtime value
  in src/graphics/shader/recompiler/ir/passes/ResourceTracking.cpp:168
```

after 16 compute shaders compiled; the run also called three stubbed imports (`Http_v1`,
`VideoOutVrrStatus_v1` x2, `VideoOutVrrPeg_v1`) which returned and did not stop it. This is the
resource-tracking pass refusing a V# whose dword 0 it cannot trace to a runtime value (an SRT /
user-data derived base it does not model). Not a GPU-driver problem. Start by dumping that shader
(`--graphics-debug-dump true`, `_Shaders/original/*cs_f9a0dc71344fec27*`), look at what feeds
`s_load` / the buffer descriptor at pc 0x2c8, and extend `ResourceTracking.cpp` to follow it (or
fall back to a runtime descriptor fetch as the pixel path does). Separate task from the fps work.

## 17. PERFORMANCE, executed: 28 -> 38 fps, and why the RTX 3070 is now the limit

Measured with `KYTY_REPORT_FPS=1` on gameplay (frame > 600), 75 s runs, i5-12400F + RTX 3070.

| state | gameplay avg fps |
|---|---|
| section 16 baseline | 27-28 |
| skip empty submits + lazy global barrier + 2 ms polled timeline wait + readback prefetch at flip | 34-36 |
| no GPU drains from CPU-side reads on the command thread (17.1) | 36-37 |
| RELEASE_MEM labels no longer submit one queue submission each (17.2) | 37-38 |
| CPU reads of retired GPU writes served from a second queue (17.3) | 37-39 |

### 17.1 Drains removed from the command-processor thread

Every read of a GPU-protected page from the command thread drained the whole queue. Sources found
with the `READFAULT` stack capture (`KYTY_WAIT_TRACE=1`, `RenderContext::HandleFault`):

- `NoteGuestGpuWrite` (stall diagnostic) re-read the DMA destination through the guest mapping
  after every `DMA_DATA`: ~145 drains/s. Now reads through the backing alias only when clean.
- `DispatchIndirect` read its argument triple from GPU-written memory: ~30 drains/s. Now
  `vkCmdDispatchIndirect` from the cached buffer (`RenderExecutor::DispatchIndirect`) when the
  arguments are GPU-dirty; `KYTY_NO_INDIRECT_DISPATCH=1` restores the old path.
- The per-descriptor `TryReadBacking` probe in `FindBuffers` touched the page; it now asks
  `RenderContext::IsMapped`.
- `CopyBuffer`/`UploadCopies` read guest memory through the protected mapping; they now read the
  backing alias (`KYTY_NO_BACKING_COPY=1` restores memcpy).

### 17.2 Label submissions

`CpOpReleaseMem` flushed after every 32-bit label write: 6800 of 7400 `vkQueueSubmit`/s (190 per
frame). A label without an interrupt is visible at once (the immediate write in `write32`) and
its end-of-pipe operation retires with the next natural submit, so it now only flushes once 16
draws/dispatches have accumulated (`CommandProcessor::BufferFlushIfBusy`, threshold
`KYTY_LABEL_FLUSH_WORK`, `KYTY_FLUSH_LABELS=1` restores per-label flushes). Submits: 7400 -> 1250/s.
Fully batching (never flushing) doubled the main thread's readback wait, hence the threshold.

### 17.3 Second-queue readback

`RBTICK` logging showed the GPU had already finished writing the counters the game reads every
frame (writer tick < executed tick) while `ReadMemory` still drained to the queue tail (~5 ms per
read, 40/s on the main thread). `BufferCache::TryImmediateReadback` copies the dirty ranges of the
faulting 64 KiB window on a second queue of the same family (`GraphicContext::readback_queue`,
created in `vulkanWindow.cpp` when the family has two queues) waiting only on the master timeline
value of the last writer (`m_gpu_write_ticks`, per 64 KiB block, recorded in `ObtainBuffer`).
Main-thread readback wait: 183 -> ~80 ms/s. `KYTY_NO_READBACK_QUEUE=1` disables it.

### 17.4 Why it stops at ~38 fps: the GPU

`nvidia-smi` during gameplay: 77-83 % utilization at 208 W. `KYTY_GPU_TIMING=1` (timestamp after
every draw/dispatch, `RenderContext::GpuTimer*`, printed as `GPUTIME` per shader hash) shows
~830 ms of GPU work per second: dispatch 600 ms, draw 230 ms. Three compute passes cost 2.0-2.6 ms
each per frame (hashes 83a66a913cffb783, 1f7c6bce476acaf9, f8918ae204ec6733; the first two are the
froxel volumetric passes, 39x22x1 and 39x22x64 groups). The game renders at native 3840x2160
(2300 render targets of that size per run); the video-out output mode has no 4K/1080p choice on
PS5, so the emulator cannot lower it.

`KYTY_PIPELINE_STATS=1` (VK_KHR_pipeline_executable_properties, printed as `PIPESTATS` at pipeline
creation) gives the cause: those shaders use **255 registers** (the hardware maximum), binaries of
188-255 KB against 12-14 KB of native code, and one spills 288 bytes of local memory per thread.
Occupancy is therefore ~8 warps per SM and the passes are latency bound. The root is structural:
the recompiler runs a 64-lane guest wave as `LocalSize 32` with two guest lanes per host thread
(`lane_count == 2`, `EmitBlock` emits every instruction twice), which doubles live state.

Tried and measured as no gain on GPU time (all removed or left off):
- dropping the per-dispatch `ShaderWriteHazardBarrier`/`ShaderAccessBarrier` (`KYTY_LIGHT_BARRIERS=1`,
  still available): dispatch time unchanged.
- skipping the software bounds check on raw buffer loads (robustBufferAccess2 is enabled): SPIR-V
  shrank but GPU time and registers did not change; code removed.

Next real step for fps is a different wave64 strategy in the recompiler (64 host threads per wave
with cross-subgroup exchange for wave-wide scalar decisions) or lower register pressure in the
emitted code. Both are recompiler projects, not tuning.

### 17.5 Diagnostics added (all env-gated, off by default)

`KYTY_REPORT_FPS`, `KYTY_WAIT_TRACE` (per-thread wait report, `READFAULT` with stacks, `CALLERS`
histograms for submits and tick waits), `KYTY_SAMPLE`/`KYTY_SAMPLE_DELAY`/`KYTY_SAMPLE_SECONDS`
(sampling profiler of the command thread; symbolize RVAs with the lld map), `KYTY_GPU_TIMING`,
`KYTY_PIPELINE_STATS`, `KYTY_HANG_DUMP` (prints the command thread's stack when no frame is
presented for 8 s). A boot-time hang (game stops submitting, Gfx thread times out on the flip event)
was seen twice in ~30 runs, before and after these changes; it never reproduced in 8 watched runs.
