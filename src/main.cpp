#include "common/common.h"
#include "common/dateTime.h"
#include "common/debug.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "emulator.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kytyGitVersion.h"

#include <atomic>
#include <cstring>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <tlhelp32.h>
#endif

#include <charconv>
#include <cstdio>
#include <string_view>
#include <vector>
#include <fmt/format.h>
#include <magic_enum.hpp>

using namespace Common;
using namespace Emulator;

static std::string GetBuildString() {
	Date date = Date::FromMacros(std::string(__DATE__));

#if KYTY_BUILD == KYTY_BUILD_DEBUG
	std::string type = "Debug";
#elif KYTY_BUILD == KYTY_BUILD_RELEASE
	std::string type = "Release";
#else
	std::string type = "????";
#endif

	std::string compiler = Debug::GetCompiler() + "-" + Debug::GetLinker();

	std::string str =
	    fmt::format("{}, {}, ver = {}, git = {}, date = {}", type.c_str(), compiler.c_str(),
	                KYTY_VERSION, KYTY_GIT_VERSION, date.ToString().c_str());

	return str;
}

static void PrintUsage() {
	::printf("%s\n", GetBuildString().c_str());
	::printf("kyty_emulator --game <dir|elf> [options]\n\n");
	::printf("Options:\n");
	::printf("  --game <dir|elf>                     Game directory or ELF to load.\n");
	::printf("  --game-patch <json>                  ETAHen cheat file.\n");
	::printf("  --screen-width <num>                 Window width. Default: 1280.\n");
	::printf("  --screen-height <num>                Window height. Default: 720.\n");
	::printf(
	    "  --user-name <name>                   Local user name (1-16 bytes). Default: Kyty.\n");
	::printf("  --user-id <num>                      Local user ID. Default: %d.\n",
	         Config::DEFAULT_USER_ID);
	::printf("  --mic <name>                        Capture from this microphone; omit for silence.\n");
	::printf(
	    "  --present-mode <value>               Fifo, Mailbox, or Immediate. Default: Mailbox.\n");
	::printf(
	    "  --gpu <index>                        Vulkan physical device index. Default: auto.\n");
	::printf("  --fullscreen                         Run in borderless desktop fullscreen.\n");
	::printf("  --vr                                 Enable the virtual VR headset.\n");
	::printf("  --amd-cpu                            Apply AMD CPU instruction patches.\n");
	::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
	::printf("  --console-language <0-29>            Console language. Default: 1 (English US).\n");
	::printf("  --vulkan-validation <true|false>     Enable Vulkan validation.\n");
	::printf("  --gpu-assisted-validation <t|f>      Bounds-check shader accesses on the GPU.\n"
	         "                                       Implies --vulkan-validation; very slow.\n");
	::printf("  --shader-validation <true|false>     Enable shader validation.\n");
	::printf("  --tessellation                      Draw tessellation patches; skipped by default.\n");
	::printf("  --shader-optimization-type <value>   None, Size, or Performance.\n");
	::printf("  --shader-log-direction <value>       Silent, Console, or File.\n");
	::printf("  --shader-log-folder <path>           Shader log output folder.\n");
	::printf("  --command-buffer-dump <true|false>   Enable command buffer dumps.\n");
	::printf("  --command-buffer-dump-folder <path>  Command buffer dump folder.\n");
	::printf("  --graphics-debug-dump <true|false>   Enable graphics debug dumps.\n");
	::printf("  --printf-direction <value>           Silent, Console, or File.\n");
	::printf("  --printf-output-file <path>          Guest printf output file.\n");
	::printf("  --profile                            Enable the Tracy profiler.\n");
	::printf("  --spirv-debug-printf <true|false>    Enable SPIR-V debug printf.\n");
	::printf(
	    "  --readback-linear-images <true|false> Read back writable linear images on submit.\n");
	::printf("  --playgo-hack                       Use the supplied PlayGo stub fallback.\n");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::printf("  --redzone                            Protect the guest SysV red zone.\n");
#endif
	::printf("  --keymap <Control=Input>             DualSense mapping; may be repeated.\n");
	::printf("  --rd                                 Enable RenderDoc capture.\n");
}

static bool NextArg(int argc, char* argv[], int& index, std::string& out) {
	if (index + 1 >= argc) {
		return false;
	}

	index++;
	out = argv[index];
	return true;
}

static bool ParseBool(const std::string& value, bool& out) {
	if (Common::EqualNoCase(value, "true") || value == "1" || Common::EqualNoCase(value, "yes") ||
	    Common::EqualNoCase(value, "on")) {
		out = true;
		return true;
	}

	if (Common::EqualNoCase(value, "false") || value == "0" || Common::EqualNoCase(value, "no") ||
	    Common::EqualNoCase(value, "off")) {
		out = false;
		return true;
	}

	return false;
}

template <typename E>
static bool ParseEnum(const std::string& value, E& out) {
	auto enum_value = magic_enum::enum_cast<E>(value.c_str());
	if (!enum_value.has_value()) {
		return false;
	}

	out = enum_value.value();
	return true;
}

static bool ParseConsoleLanguage(const std::string& value, uint32_t& out) {
	uint32_t language = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), language);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    language > Config::MAX_CONSOLE_LANGUAGE) {
		return false;
	}
	out = language;
	return true;
}

static bool ParseUserId(const std::string& value, int32_t& out) {
	int32_t user_id   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), user_id);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    !Config::IsConfiguredUserIdValid(user_id)) {
		return false;
	}
	out = user_id;
	return true;
}

static bool ParseArgs(int argc, char* argv[], RunOptions& options, bool& show_help) {
	show_help = false;

	for (int i = 1; i < argc; i++) {
		std::string arg = std::string(argv[i]);
		std::string value;

		if (arg == "--help" || arg == "-h") {
			show_help = true;
			continue;
		}

		if (arg == "--rd") {
			options.config.renderdoc_enabled = true;
			continue;
		}

		if (arg == "--fullscreen") {
			options.config.fullscreen_enabled = true;
			continue;
		}

		if (arg == "--vr") {
			options.config.vr_enabled = true;
			continue;
		}

		if (arg == "--amd-cpu") {
			options.config.amd_cpu_enabled = true;
			continue;
		}

		if (arg == "--playgo-hack") {
			options.config.playgo_hack_enabled = true;
			continue;
		}

		if (arg == "--tessellation") {
			options.config.tessellation_enabled = true;
			continue;
		}

		if (arg == "--profile") {
			options.config.profiler_enabled = true;
			continue;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (arg == "--redzone") {
			options.config.red_zone_protection_enabled = true;
			continue;
		}
#endif

		if (!arg.starts_with("--")) {
			::printf("game input must be provided with --game\n");
			return false;
		}

		if (!NextArg(argc, argv, i, value)) {
			::printf("missing value for %s\n", arg.c_str());
			return false;
		}

		if (arg == "--game") {
			if (!options.app0_dir.empty()) {
				::printf("--game can only be specified once\n");
				return false;
			}

			value = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (Common::File::IsDirectoryExisting(path)) {
				options.app0_dir = path;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::File::IsFileExisting(path)) {
				options.app0_dir = path.parent_path();

				if (options.app0_dir.empty()) {
					options.app0_dir = ".";
				}

				options.elf = std::filesystem::path("/app0") / path.filename();
			} else {
				::printf("--game must point to an existing directory or ELF: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--game-patch") {
			if (!options.game_patch.empty()) {
				::printf("--game-patch can only be specified once\n");
				return false;
			}
			value = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (!Common::File::IsFileExisting(path)) {
				::printf("--game-patch must point to an existing file: %s\n", value.c_str());
				return false;
			}
			options.game_patch = path;
		} else if (arg == "--screen-width") {
			options.config.screen_width = static_cast<uint32_t>(Common::ToInt32(value));
		} else if (arg == "--screen-height") {
			options.config.screen_height = static_cast<uint32_t>(Common::ToInt32(value));
		} else if (arg == "--user-name") {
			if (value.empty() || value.size() > Config::MAX_USER_NAME_LENGTH) {
				::printf("invalid user name: must contain 1-%zu bytes\n",
				         Config::MAX_USER_NAME_LENGTH);
				return false;
			}
			options.config.user_name = value;
		} else if (arg == "--user-id") {
			if (!ParseUserId(value, options.config.user_id)) {
				::printf("invalid user ID: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--mic") {
			options.config.audio_input_device = value;
		} else if (arg == "--present-mode") {
			if (!ParseEnum(value, options.config.present_mode)) {
				::printf("invalid present mode: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--gpu") {
			options.config.gpu_index = Common::ToInt32(value);
		} else if (arg == "--vblank-frequency") {
			const int32_t vblank_frequency = Common::ToInt32(value);
			options.config.vblank_frequency =
			    static_cast<uint32_t>(vblank_frequency < 0 ? 0 : vblank_frequency);
		} else if (arg == "--console-language") {
			if (!ParseConsoleLanguage(value, options.config.console_language)) {
				::printf("invalid console language: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--vulkan-validation") {
			if (!ParseBool(value, options.config.vulkan_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--gpu-assisted-validation") {
			if (!ParseBool(value, options.config.gpu_assisted_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-validation") {
			if (!ParseBool(value, options.config.shader_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-optimization-type") {
			if (!ParseEnum(value, options.config.shader_optimization_type)) {
				::printf("invalid shader optimization type: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-direction") {
			if (!ParseEnum(value, options.config.shader_log_direction)) {
				::printf("invalid shader log direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-folder") {
			options.config.shader_log_folder = Common::PathFromUtf8(value);
		} else if (arg == "--command-buffer-dump") {
			if (!ParseBool(value, options.config.command_buffer_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--command-buffer-dump-folder") {
			options.config.command_buffer_dump_folder = Common::PathFromUtf8(value);
		} else if (arg == "--graphics-debug-dump") {
			if (!ParseBool(value, options.config.graphics_debug_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--printf-direction") {
			if (!ParseEnum(value, options.config.printf_direction)) {
				::printf("invalid printf direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--printf-output-file") {
			options.config.printf_output_file = Common::PathFromUtf8(value);
		} else if (arg == "--spirv-debug-printf") {
			if (!ParseBool(value, options.config.spirv_debug_printf_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--readback-linear-images") {
			if (!ParseBool(value, options.config.readback_linear_images)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--keymap") {
			const auto split = value.find('=');
			if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
				::printf("invalid keymap: %s\n", value.c_str());
				return false;
			}
			options.config.keymap.push_back(value);
		} else {
			::printf("unknown option: %s\n", arg.c_str());
			return false;
		}
	}

	if (options.config.gpu_assisted_validation_enabled) {
		options.config.vulkan_validation_enabled = true;
	}

	return show_help || (!options.app0_dir.empty() && !options.elf.empty());
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// Reports any thread that burns CPU without making a kernel call, with the instruction pointer
// so a guest spin loop can be traced back to its module.
// KYTY_SAMPLE_GPU=1: samples the GPU command thread about every millisecond and prints the
// hottest emulator code (image-relative offsets, self and inclusive) every five seconds.
// KYTY_SAMPLE_ALL=1: every thread of the process. Every five seconds, for the busiest threads
// (by CPU time): the CPU share and where their samples landed (guest code, emulator functions,
// system modules), to tell whether the game's own threads limit the frame rate.
static void StartAllThreadSampler() {
	if (std::getenv("KYTY_SAMPLE_ALL") == nullptr) {
		return;
	}
	std::thread([] {
		const auto self_tid = GetCurrentThreadId();
		const auto module   = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
		struct ThreadStats {
			uint64_t                               cpu_start = 0;
			uint32_t                               samples   = 0;
			uint32_t                               guest     = 0;
			std::unordered_map<uint64_t, uint32_t> exe;     // emulator code, by RVA
			std::unordered_map<uint64_t, uint32_t> modules; // system modules, by base
			std::unordered_map<uint64_t, uint32_t> callers; // emulator caller of system code
		};
		std::unordered_map<DWORD, ThreadStats> stats;
		std::vector<DWORD>                     tids;
		auto last_list   = std::chrono::steady_clock::now() - std::chrono::seconds(1);
		auto last_report = std::chrono::steady_clock::now();
		const auto cpu_time = [](HANDLE thread) {
			FILETIME created {}, exited {}, kernel {}, user {};
			if (GetThreadTimes(thread, &created, &exited, &kernel, &user) == 0) {
				return uint64_t {0};
			}
			return ((uint64_t {kernel.dwHighDateTime} << 32u) | kernel.dwLowDateTime) +
			       ((uint64_t {user.dwHighDateTime} << 32u) | user.dwLowDateTime);
		};
		for (;;) {
			Sleep(2);
			const auto now = std::chrono::steady_clock::now();
			if (now - last_list > std::chrono::milliseconds(500)) {
				last_list = now;
				tids.clear();
				auto* snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
				if (snapshot != INVALID_HANDLE_VALUE) {
					THREADENTRY32 entry {};
					entry.dwSize = sizeof(entry);
					for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
						if (entry.th32OwnerProcessID == GetCurrentProcessId() && entry.th32ThreadID != self_tid) {
							tids.push_back(entry.th32ThreadID);
						}
					}
					CloseHandle(snapshot);
				}
			}
			for (const auto tid: tids) {
				auto* thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION,
				                          FALSE, tid);
				if (thread == nullptr) {
					continue;
				}
				auto& s = stats[tid];
				if (s.cpu_start == 0) {
					s.cpu_start = cpu_time(thread);
				}
				uint64_t frames[8] {};
				int      frame_count = 0;
				if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
					CONTEXT context {};
					context.ContextFlags = CONTEXT_FULL;
					if (GetThreadContext(thread, &context) != 0) {
						for (; frame_count < 8 && context.Rip != 0; frame_count++) {
							frames[frame_count] = context.Rip;
							DWORD64 image_base = 0;
							auto*   function   = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
							if (function == nullptr) {
								frame_count++;
								break;
							}
							void*   handler_data = nullptr;
							DWORD64 establisher  = 0;
							RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context,
							                 &handler_data, &establisher, nullptr);
						}
					}
					ResumeThread(thread);
				}
				CloseHandle(thread);
				if (frame_count == 0) {
					continue;
				}
				s.samples++;
				const auto rip = frames[0];
				if (rip >= module && rip < module + 0x4000000u) {
					s.exe[rip - module]++;
					continue;
				}
				HMODULE owner = nullptr;
				GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
				                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				                   reinterpret_cast<LPCSTR>(rip), &owner);
				if (owner == nullptr) {
					s.guest++;
					continue;
				}
				s.modules[reinterpret_cast<uint64_t>(owner)]++;
				for (int frame = 1; frame < frame_count; frame++) {
					if (frames[frame] >= module && frames[frame] < module + 0x4000000u) {
						s.callers[frames[frame] - module]++;
						break;
					}
				}
			}
			if (now - last_report < std::chrono::seconds(5)) {
				continue;
			}
			const auto elapsed_100ns =
			    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now - last_report).count()) * 10u;
			last_report = now;
			struct Row {
				DWORD  tid;
				double cpu;
			};
			std::vector<Row> rows;
			for (auto& [tid, s]: stats) {
				auto* thread = OpenThread(THREAD_QUERY_INFORMATION, FALSE, tid);
				if (thread == nullptr) {
					continue;
				}
				const auto cpu = cpu_time(thread);
				CloseHandle(thread);
				rows.push_back({tid, elapsed_100ns == 0 ? 0.0 : 100.0 * static_cast<double>(cpu - s.cpu_start) / elapsed_100ns});
				s.cpu_start = cpu;
			}
			std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.cpu > b.cpu; });
			const auto top = [](const std::unordered_map<uint64_t, uint32_t>& table, uint32_t total, size_t count,
			                    bool names) {
				std::vector<std::pair<uint64_t, uint32_t>> items(table.begin(), table.end());
				std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
				std::string text;
				for (size_t i = 0; i < items.size() && i < count; i++) {
					if (names) {
						char name[MAX_PATH] {};
						GetModuleFileNameA(reinterpret_cast<HMODULE>(items[i].first), name, MAX_PATH);
						const char* slash = std::strrchr(name, '\\');
						text += fmt::format(" {}={:.0f}%", slash != nullptr ? slash + 1 : name,
						                    100.0 * items[i].second / std::max(total, 1u));
					} else {
						text += fmt::format(" {:x}={:.0f}%", items[i].first, 100.0 * items[i].second / std::max(total, 1u));
					}
				}
				return text;
			};
			for (size_t i = 0; i < rows.size() && i < 8; i++) {
				auto& s = stats[rows[i].tid];
				std::printf("THREADSAMPLE tid=%lu cpu=%.0f%% samples=%u guest=%.0f%% |exe%s |mod%s |via%s\n",
				            static_cast<unsigned long>(rows[i].tid), rows[i].cpu, s.samples,
				            100.0 * s.guest / std::max(s.samples, 1u), top(s.exe, s.samples, 6, false).c_str(),
				            top(s.modules, s.samples, 4, true).c_str(), top(s.callers, s.samples, 4, false).c_str());
			}
			std::fflush(stdout);
			for (auto& [tid, s]: stats) {
				s.samples = 0;
				s.guest   = 0;
				s.exe.clear();
				s.modules.clear();
				s.callers.clear();
			}
		}
	}).detach();
}

static void StartGpuSampler() {
	StartAllThreadSampler();
	if (std::getenv("KYTY_SAMPLE_GPU") == nullptr) {
		return;
	}
	std::thread([] {
		const auto module = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
		std::unordered_map<uint64_t, uint32_t> self;
		std::unordered_map<uint64_t, uint32_t> inclusive;
		// Samples outside the emulator image: by module and by the emulator function that called out.
		std::unordered_map<uint64_t, uint32_t> ext_module;
		std::unordered_map<uint64_t, uint32_t> ext_caller;
		std::unordered_map<uint64_t, uint32_t> ext_caller2;
		std::unordered_map<uint64_t, uint32_t> parent;
		uint32_t samples = 0;
		auto     last    = std::chrono::steady_clock::now();
		for (;;) {
			Sleep(1);
			const auto tid = Libs::Graphics::GpuOsThreadId();
			if (tid == 0) {
				continue;
			}
			auto* gpu = OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME, FALSE, tid);
			if (gpu == nullptr) {
				continue;
			}
			// Nothing may allocate while the thread is suspended: it can be holding the heap
			// lock, and the sampler would then wait on it forever.
			uint64_t frames[16] {};
			int      frame_count = 0;
			if (SuspendThread(gpu) != static_cast<DWORD>(-1)) {
				CONTEXT context {};
				context.ContextFlags = CONTEXT_FULL;
				if (GetThreadContext(gpu, &context) != 0) {
					for (; frame_count < 16 && context.Rip != 0; frame_count++) {
						frames[frame_count] = context.Rip;
						DWORD64 image_base = 0;
						auto*   function   = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
						if (function == nullptr) {
							frame_count++;
							break;
						}
						void*   handler_data = nullptr;
						DWORD64 establisher  = 0;
						RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context,
						                 &handler_data, &establisher, nullptr);
					}
				}
				ResumeThread(gpu);
			}
			if (frame_count > 0) {
				samples++;
				std::unordered_set<uint64_t> seen;
				for (int frame = 0; frame < frame_count; frame++) {
					const auto rip = frames[frame];
					if (rip >= module && rip < module + 0x4000000u) {
						if (frame == 0) {
							self[rip - module]++;
						}
						if (seen.insert(rip - module).second) {
							inclusive[rip - module]++;
						}
					} else if (frame == 0) {
						self[0]++;
						HMODULE owner = nullptr;
						GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
						                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
						                   reinterpret_cast<LPCSTR>(rip), &owner);
						ext_module[reinterpret_cast<uint64_t>(owner)]++;
					}
				}
				// Caller of the sampled emulator function (who spins in a lock, who allocates).
				if (frame_count > 1 && frames[0] >= module && frames[0] < module + 0x4000000u &&
				    frames[1] >= module && frames[1] < module + 0x4000000u) {
					parent[frames[1] - module]++;
				}
				if (frame_count > 0 && !(frames[0] >= module && frames[0] < module + 0x4000000u)) {
					for (int frame = 1; frame < frame_count; frame++) {
						if (frames[frame] >= module && frames[frame] < module + 0x4000000u) {
							ext_caller[frames[frame] - module]++;
							for (int next = frame + 1; next < frame_count; next++) {
								if (frames[next] >= module && frames[next] < module + 0x4000000u) {
									ext_caller2[frames[next] - module]++;
									break;
								}
							}
							break;
						}
					}
				}
			}
			CloseHandle(gpu);
			if (std::chrono::steady_clock::now() - last > std::chrono::seconds(5) && samples > 0) {
				last = std::chrono::steady_clock::now();
				const auto top = [&](const std::unordered_map<uint64_t, uint32_t>& table,
				                     const char* name) {
					std::vector<std::pair<uint64_t, uint32_t>> rows(table.begin(), table.end());
					std::sort(rows.begin(), rows.end(),
					          [](const auto& a, const auto& b) { return a.second > b.second; });
					std::string line = fmt::format("GPUSAMPLE {} tid={} n={}:", name, tid, samples);
					for (size_t i = 0; i < rows.size() && i < 30; i++) {
						line += fmt::format(" {:x}={:.1f}%", rows[i].first,
						                    100.0 * rows[i].second / samples);
					}
					std::printf("%s\n", line.c_str());
				};
				top(self, "self");
				top(inclusive, "incl");
				top(ext_caller, "extcaller");
				top(ext_caller2, "extcaller2");
				top(parent, "parent");
				{
					std::string line = fmt::format("GPUSAMPLE extmod n={}:", samples);
					for (const auto& [base, count]: ext_module) {
						char name[MAX_PATH] {};
						if (base == 0 || GetModuleFileNameA(reinterpret_cast<HMODULE>(base), name, MAX_PATH) == 0) {
							std::snprintf(name, sizeof(name), "nounwind");
						}
						const char* slash = std::strrchr(name, '\\');
						line += fmt::format(" {}={:.1f}%", slash != nullptr ? slash + 1 : name, 100.0 * count / samples);
					}
					std::printf("%s\n", line.c_str());
				}
				ext_module.clear();
				ext_caller.clear();
				ext_caller2.clear();
				parent.clear();
				std::fflush(stdout);
				self.clear();
				inclusive.clear();
				samples = 0;
			}
		}
	}).detach();
}

static void StartSpinWatchdog() {
	std::thread([] {
		std::unordered_map<DWORD, uint64_t> previous;
		const auto                          process = GetCurrentProcessId();
		const auto                          self    = GetCurrentThreadId();
		uint64_t last_beat  = 0;
		uint32_t stuck_hits = 0;
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(10));
			// The GPU thread reports its phase; when the heartbeat stops, print where it sits.
			{
				const auto beat = Libs::Graphics::GpuHeartbeat();
				stuck_hits      = beat == last_beat && beat != 0 ? stuck_hits + 1 : 0;
				last_beat       = beat;
				if (stuck_hits != 0) {
					uint64_t    rip = 0;
					std::string stack;
					const auto  tid = Libs::Graphics::GpuOsThreadId();
					auto*      gpu  = tid != 0 ? OpenThread(THREAD_QUERY_INFORMATION | THREAD_GET_CONTEXT |
                                                          THREAD_SUSPEND_RESUME,
                                                      FALSE, tid)
					                            : nullptr;
					if (gpu != nullptr) {
						CONTEXT context {};
						context.ContextFlags = CONTEXT_CONTROL;
						context.ContextFlags = CONTEXT_FULL;
						if (SuspendThread(gpu) != static_cast<DWORD>(-1)) {
							if (GetThreadContext(gpu, &context) != 0) {
								rip = context.Rip;
								// Unwind the suspended thread with the x64 function tables.
								for (int frame = 0; frame < 40 && context.Rip != 0; frame++) {
									HMODULE module = nullptr;
									char    name[MAX_PATH] {};
									if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
									                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
									                       reinterpret_cast<LPCSTR>(context.Rip), &module) != 0) {
										GetModuleFileNameA(module, name, sizeof(name));
										const char* slash = std::strrchr(name, '\\');
										stack += fmt::format(" {}+{:x}", slash != nullptr ? slash + 1 : name,
										                     context.Rip - reinterpret_cast<uint64_t>(module));
									} else {
										stack += fmt::format(" {:x}", context.Rip);
									}
									DWORD64 image_base = 0;
									auto*   function   = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
									if (function == nullptr) {
										if (context.Rsp == 0 || IsBadReadPtr(reinterpret_cast<void*>(context.Rsp), 8)) {
											break;
										}
										context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
										context.Rsp += 8;
									} else {
										void*   handler_data = nullptr;
										DWORD64 establisher  = 0;
										RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function,
										                 &context, &handler_data, &establisher, nullptr);
									}
								}
							}
							ResumeThread(gpu);
						}
						CloseHandle(gpu);
					}
					const auto base = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
					LOGF("\t GPU THREAD STUCK: %" PRIu32 "0s phase=%s detail=0x%016" PRIx64
					     " rip=0x%016" PRIx64 " rva=0x%" PRIx64 "\n",
					     stuck_hits, Libs::Graphics::GpuPhase(), Libs::Graphics::GpuPhaseDetail(), rip,
					     rip >= base ? rip - base : 0);
					LOGF("\t GPU THREAD STACK:%s\n", stack.c_str());
					Libs::Graphics::PrintRecentShaders();
					Libs::Graphics::ReportSchedulerState();
				}
			}
			auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
			if (snapshot == INVALID_HANDLE_VALUE) {
				continue;
			}
			THREADENTRY32 entry {};
			entry.dwSize = sizeof(entry);
			if (Thread32First(snapshot, &entry) != 0) {
				do {
					if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self) {
						continue;
					}
					auto* handle = OpenThread(THREAD_QUERY_INFORMATION | THREAD_GET_CONTEXT |
					                              THREAD_SUSPEND_RESUME,
					                          FALSE, entry.th32ThreadID);
					if (handle == nullptr) {
						continue;
					}
					FILETIME creation {};
					FILETIME exited {};
					FILETIME kernel {};
					FILETIME user {};
					if (GetThreadTimes(handle, &creation, &exited, &kernel, &user) != 0) {
						const auto to_ticks = [](const FILETIME& time) {
							return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) |
							       time.dwLowDateTime;
						};
						const auto total = to_ticks(kernel) + to_ticks(user);
						const auto found = previous.find(entry.th32ThreadID);
						const auto delta = found == previous.end() ? 0 : total - found->second;
						previous[entry.th32ThreadID] = total;
						// Half the sample window spent running means a spin, not normal work.
						if (delta > 50000000ull) {
							CONTEXT context {};
							context.ContextFlags = CONTEXT_FULL;
							bool captured        = false;
							if (SuspendThread(handle) != static_cast<DWORD>(-1)) {
								captured = GetThreadContext(handle, &context) != 0;
								ResumeThread(handle);
							}
							if (!captured) {
								CloseHandle(handle);
								continue;
							}
							LOGF("\t SPIN: os_thread=%lu cpu_ms=%" PRIu64 " rip=0x%016" PRIx64
							     " rsp=0x%016" PRIx64 "\n",
							     entry.th32ThreadID, delta / 10000ull, context.Rip, context.Rsp);
							LOGF("\t SPIN:   rax=%016" PRIx64 " rbx=%016" PRIx64 " rcx=%016" PRIx64
							     " rdx=%016" PRIx64 "\n",
							     context.Rax, context.Rbx, context.Rcx, context.Rdx);
							LOGF("\t SPIN:   rsi=%016" PRIx64 " rdi=%016" PRIx64 " rbp=%016" PRIx64
							     " r8= %016" PRIx64 "\n",
							     context.Rsi, context.Rdi, context.Rbp, context.R8);
							LOGF("\t SPIN:   r9= %016" PRIx64 " r10=%016" PRIx64 " r11=%016" PRIx64
							     " r12=%016" PRIx64 "\n",
							     context.R9, context.R10, context.R11, context.R12);
							LOGF("\t SPIN:   r13=%016" PRIx64 " r14=%016" PRIx64 " r15=%016" PRIx64
							     "\n",
							     context.R13, context.R14, context.R15);
							const auto* code =
							    reinterpret_cast<const uint8_t*>(context.Rip - 0x20);
							LOGF("\t SPIN:   code@rip-0x20:");
							for (uint32_t offset = 0; offset < 0x40; offset++) {
								LOGF("%s%02x", offset == 0x20 ? " |" : " ", code[offset]);
							}
							LOGF("\n");
							const auto* stack = reinterpret_cast<const uint64_t*>(context.Rsp);
							LOGF("\t SPIN:   stack:");
							for (uint32_t slot = 0; slot < 8; slot++) {
								LOGF(" %016" PRIx64, stack[slot]);
							}
							LOGF("\n");
						}
					}
					CloseHandle(handle);
				} while (Thread32Next(snapshot, &entry) != 0);
			}
			CloseHandle(snapshot);
		}
	}).detach();
}

// KYTY_SAMPLE=<os thread id> samples that thread's instruction pointer every millisecond and
// prints the hottest addresses on exit, as offsets into the module so llvm-symbolizer can name
// them. Used to find where a saturated thread actually spends its time.
extern std::atomic<uint64_t> g_kyty_flip_counter;

// KYTY_HANG_DUMP=1: when no frame has been presented for eight seconds, print the wait
// report and where the command-processor thread is, a few times, so a hang can be read.
// KYTY_THREAD_DUMP=<seconds>: after that delay, print every thread's position in game code and
// the game return addresses found on its stack, to see where a stuck game waits.
static void StartThreadDump() {
	const char* text = std::getenv("KYTY_THREAD_DUMP");
	if (text == nullptr) {
		return;
	}
	const auto delay = std::strtoul(text, nullptr, 10);
	std::thread([delay] {
		constexpr uint64_t GameBegin = 0x900000000ull;
		constexpr uint64_t GameEnd   = 0x904000000ull;
		for (int round = 0; round < 3; round++) {
			std::this_thread::sleep_for(std::chrono::seconds(round == 0 ? delay : 5));
			const auto self     = GetCurrentThreadId();
			HANDLE     snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
			THREADENTRY32 entry {};
			entry.dwSize = sizeof(entry);
			printf("THREADDUMP round %d\n", round);
			for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
				if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) {
					continue;
				}
				HANDLE thread = OpenThread(THREAD_ALL_ACCESS, FALSE, entry.th32ThreadID);
				if (thread == nullptr || SuspendThread(thread) == static_cast<DWORD>(-1)) {
					if (thread != nullptr) {
						CloseHandle(thread);
					}
					continue;
				}
				CONTEXT context {};
				context.ContextFlags = CONTEXT_FULL;
				if (GetThreadContext(thread, &context) != 0) {
					printf("  tid %lu rip=%llx:", entry.th32ThreadID,
					       static_cast<unsigned long long>(context.Rip));
					const auto* stack = reinterpret_cast<const uint64_t*>(context.Rsp);
					int         found = 0;
					for (int slot = 0; slot < 4096 && found < 16; slot++) {
						uint64_t value = 0;
						if (ReadProcessMemory(GetCurrentProcess(), stack + slot, &value, sizeof(value),
						                      nullptr) == 0) {
							break;
						}
						if (value >= GameBegin && value < GameEnd) {
							printf(" %llx", static_cast<unsigned long long>(value - GameBegin));
							found++;
						}
					}
					printf("\n");
				}
				ResumeThread(thread);
				CloseHandle(thread);
			}
			CloseHandle(snapshot);
			fflush(stdout);
		}
	}).detach();
}

static void StartHangWatch() {
	if (std::getenv("KYTY_HANG_DUMP") == nullptr) {
		return;
	}
	std::thread([] {
		uint64_t last  = 0;
		int      stale = 0;
		int      dumps = 0;
		for (;;) {
			Common::Thread::SleepMicro(2000000);
			const auto now = g_kyty_flip_counter.load(std::memory_order_relaxed);
			if (now != last) {
				last  = now;
				stale = 0;
				continue;
			}
			if (++stale < 4 || dumps >= 3) {
				continue;
			}
			dumps++;
			printf("HANGDUMP flips=%" PRIu64 "\n", now);
			Common::WaitTrace::Report(0.0, now);
			const auto target =
			    Common::WaitTrace::HottestOsThread(Common::WaitTrace::Kind::GpuProcess);
			HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
			                               THREAD_QUERY_INFORMATION,
			                           FALSE, target);
			if (handle == nullptr) {
				printf("HANGDUMP cannot open gpu thread %lu\n", target);
				fflush(stdout);
				continue;
			}
			const auto base = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
			for (int sample = 0; sample < 6; sample++) {
				Common::Thread::SleepMicro(50000);
				if (SuspendThread(handle) == static_cast<DWORD>(-1)) {
					break;
				}
				CONTEXT context {};
				context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
				if (GetThreadContext(handle, &context) != 0) {
					char    rip_name[64] = "?";
					HMODULE rip_module   = nullptr;
					if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
					                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
					                       reinterpret_cast<LPCSTR>(context.Rip), &rip_module) != 0) {
						char full[MAX_PATH] = "";
						GetModuleFileNameA(rip_module, full, MAX_PATH);
						const char* slash = std::strrchr(full, '\\');
						std::snprintf(rip_name, sizeof(rip_name), "%s",
						              slash != nullptr ? slash + 1 : full);
					}
					printf("HANGDUMP gpu thread rip=%s+0x%" PRIx64 " stack:", rip_name,
					       rip_module != nullptr
					           ? context.Rip - reinterpret_cast<uint64_t>(rip_module)
					           : context.Rip);
					const auto* stack = reinterpret_cast<const uint64_t*>(context.Rsp);
					int         kept  = 0;
					for (int slot = 0; slot < 1024 && kept < 12; slot++) {
						uint64_t value = 0;
						if (ReadProcessMemory(GetCurrentProcess(), stack + slot, &value,
						                      sizeof(value), nullptr) == 0) {
							break;
						}
						if (value >= base && value < base + 0x2000000) {
							printf(" %" PRIx64, value - base);
							kept++;
						}
					}
					printf("\n");
				}
				ResumeThread(handle);
			}
			CloseHandle(handle);
			fflush(stdout);
		}
	}).detach();
}

static void StartSampler() {
	const char* text = std::getenv("KYTY_SAMPLE");
	if (text == nullptr) {
		return;
	}
	const auto requested = static_cast<DWORD>(std::strtoul(text, nullptr, 10));
	const auto delay     = [] {
		const char* value = std::getenv("KYTY_SAMPLE_DELAY");
		return value != nullptr ? std::strtoul(value, nullptr, 10) : 45ul;
	}();
	const auto seconds = [] {
		const char* value = std::getenv("KYTY_SAMPLE_SECONDS");
		return value != nullptr ? std::strtoul(value, nullptr, 10) : 20ul;
	}();

	std::thread([requested, delay, seconds] {
		std::this_thread::sleep_for(std::chrono::seconds(delay));

		// Without an explicit id, sample whichever thread has burned the most CPU by now: that
		// is the one holding the frame rate down.
		auto target = requested;
		if (target == 1) {
			// The thread with the most CPU is the window pump; the one to look at is whichever
			// has spent the most time inside guest command processing.
			target = Common::WaitTrace::HottestOsThread(Common::WaitTrace::Kind::GpuProcess);
			printf("sampler: command-processor thread %lu\n", target);
		} else if (target == 0) {
			const auto process = GetCurrentProcessId();
			const auto self    = GetCurrentThreadId();
			uint64_t   best    = 0;
			auto*      snap    = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
			if (snap != INVALID_HANDLE_VALUE) {
				THREADENTRY32 entry {};
				entry.dwSize = sizeof(entry);
				if (Thread32First(snap, &entry) != 0) {
					do {
						if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self) {
							continue;
						}
						auto* probe = OpenThread(THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
						if (probe == nullptr) {
							continue;
						}
						FILETIME creation {};
						FILETIME exited {};
						FILETIME kernel {};
						FILETIME user {};
						if (GetThreadTimes(probe, &creation, &exited, &kernel, &user) != 0) {
							const auto to_ticks = [](const FILETIME& t) {
								return (static_cast<uint64_t>(t.dwHighDateTime) << 32u) |
								       t.dwLowDateTime;
							};
							const auto total = to_ticks(kernel) + to_ticks(user);
							if (total > best) {
								best   = total;
								target = entry.th32ThreadID;
							}
						}
						CloseHandle(probe);
					} while (Thread32Next(snap, &entry) != 0);
				}
				CloseHandle(snap);
			}
			printf("sampler: auto-selected thread %lu (cpu %.1f ms)\n", target,
			       static_cast<double>(best) / 10000.0);
		}
		auto* handle =
		    OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION, FALSE,
		               target);
		if (handle == nullptr) {
			printf("sampler: cannot open thread %lu\n", target);
			return;
		}
		const auto base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
		constexpr uint64_t                     ModuleSpan = 0x2000000;
		std::unordered_map<uint64_t, uint32_t> histogram;
		std::unordered_map<uint64_t, uint32_t> callers;
		int                                    dumped = 0;
		uint64_t                               taken = 0;
		for (uint64_t i = 0; i < static_cast<uint64_t>(seconds) * 1000; i++) {
			Common::Thread::SleepMicro(1000);
			if (SuspendThread(handle) == static_cast<DWORD>(-1)) {
				break;
			}
			CONTEXT context {};
			context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
			if (GetThreadContext(handle, &context) != 0) {
				histogram[context.Rip]++;
				taken++;
				// A blocked thread's instruction pointer only names the syscall. Scanning the
				// stack for the nearest return address inside this module names the caller.
				if (context.Rip < base || context.Rip >= base + ModuleSpan) {
					// The first few samples blocked in the GPU driver's kernel interface get their
					// raw stack printed with module attribution, enough to read the chain by hand.
					HMODULE rip_module   = nullptr;
					char    rip_name[64] = "";
					if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
					                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
					                       reinterpret_cast<LPCSTR>(context.Rip), &rip_module) != 0) {
						char full[MAX_PATH] = "";
						GetModuleFileNameA(rip_module, full, MAX_PATH);
						const char* slash = std::strrchr(full, '\\');
						std::snprintf(rip_name, sizeof(rip_name), "%s", slash != nullptr ? slash + 1 : full);
					}
					if (dumped < 4 && std::strstr(rip_name, "win32u") != nullptr) {
						dumped++;
						printf("  STACK sample rip=0x%016" PRIx64 " rsp=0x%016" PRIx64 "\n",
						       context.Rip, context.Rsp);
						const auto* stack = reinterpret_cast<const uint64_t*>(context.Rsp);
						for (int slot = 0; slot < 96; slot++) {
							uint64_t value = 0;
							if (ReadProcessMemory(GetCurrentProcess(), stack + slot, &value,
							                      sizeof(value), nullptr) == 0) {
								break;
							}
							HMODULE module = nullptr;
							if (value < 0x10000 ||
							    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
							                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
							                       reinterpret_cast<LPCSTR>(value), &module) == 0 ||
							    module == nullptr) {
								continue;
							}
							char full[MAX_PATH] = "";
							GetModuleFileNameA(module, full, MAX_PATH);
							const char* slash = std::strrchr(full, '\\');
							printf("    [%2d] %-22s +0x%" PRIx64 "\n", slot,
							       slash != nullptr ? slash + 1 : full,
							       value - reinterpret_cast<uint64_t>(module));
						}
					}
					const auto* stack = reinterpret_cast<const uint64_t*>(context.Rsp);
					for (int slot = 0; slot < 256; slot++) {
						uint64_t value = 0;
						if (ReadProcessMemory(GetCurrentProcess(), stack + slot, &value,
						                      sizeof(value), nullptr) == 0) {
							break;
						}
						if (value >= base && value < base + ModuleSpan) {
							callers[value]++;
							break;
						}
					}
				}
			}
			ResumeThread(handle);
		}
		CloseHandle(handle);

		std::vector<std::pair<uint64_t, uint32_t>> rows(histogram.begin(), histogram.end());
		std::sort(rows.begin(), rows.end(),
		          [](const auto& a, const auto& b) { return a.second > b.second; });
		printf("SAMPLER thread=%lu samples=%" PRIu64 " module_base=0x%016" PRIx64 "\n", target,
		       taken, base);
		{
			std::vector<std::pair<uint64_t, uint32_t>> call_rows(callers.begin(), callers.end());
			std::sort(call_rows.begin(), call_rows.end(),
			          [](const auto& a, const auto& b) { return a.second > b.second; });
			printf("  -- nearest in-module caller while blocked --\n");
			for (size_t row = 0; row < call_rows.size() && row < 16; row++) {
				printf("  %5.1f%%  +0x%" PRIx64 "\n",
				       100.0 * call_rows[row].second / static_cast<double>(taken == 0 ? 1 : taken),
				       call_rows[row].first - base);
			}
			printf("  -- instruction pointer --\n");
		}
		for (size_t row = 0; row < rows.size() && row < 600; row++) {
			const auto rip = rows[row].first;
			char       module_name[MAX_PATH] = "?";
			uint64_t   module_base           = 0;
			HMODULE    module                = nullptr;
			if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
			                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			                       reinterpret_cast<LPCSTR>(rip), &module) != 0 &&
			    module != nullptr) {
				module_base = reinterpret_cast<uint64_t>(module);
				char full[MAX_PATH] = "";
				if (GetModuleFileNameA(module, full, MAX_PATH) != 0) {
					const char* slash = std::strrchr(full, '\\');
					std::snprintf(module_name, sizeof(module_name), "%s",
					              slash != nullptr ? slash + 1 : full);
				}
			}
			printf("  %5.1f%%  %-24s +0x%-10" PRIx64 " rip=0x%016" PRIx64 "\n",
			       100.0 * rows[row].second / static_cast<double>(taken == 0 ? 1 : taken),
			       module_name, module_base != 0 ? rip - module_base : 0, rip);
		}
		fflush(stdout);
	}).detach();
}

// KYTY_HW_WATCH=<hex address> arms an x86 data breakpoint on that 8-byte slot in every thread,
// so whichever instruction writes it is reported with its address, no matter which path it took.
static uint64_t g_hw_watch_address = 0;

static LONG CALLBACK HardwareWatchHandler(EXCEPTION_POINTERS* pointers) {
	if (pointers->ExceptionRecord->ExceptionCode != STATUS_SINGLE_STEP ||
	    (pointers->ContextRecord->Dr6 & 0xfull) == 0) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	auto* context = pointers->ContextRecord;
	// Only writes are armed: reading the slot here would retrigger a read breakpoint.
	const auto value = *reinterpret_cast<uint64_t*>(g_hw_watch_address);
	LOGF("\t HWWATCH: hit address=0x%016" PRIx64 " rip=0x%016" PRIx64 " value=0x%016" PRIx64
	     " os_thread=%lu\n",
	     g_hw_watch_address, context->Rip, value, GetCurrentThreadId());
	LOGF("\t HWWATCH:   rax=%016" PRIx64 " rbx=%016" PRIx64 " rcx=%016" PRIx64 " rdx=%016" PRIx64
	     "\n",
	     context->Rax, context->Rbx, context->Rcx, context->Rdx);
	LOGF("\t HWWATCH:   rsi=%016" PRIx64 " rdi=%016" PRIx64 " rbp=%016" PRIx64 " rsp=%016" PRIx64
	     "\n",
	     context->Rsi, context->Rdi, context->Rbp, context->Rsp);
	LOGF("\t HWWATCH:   r12=%016" PRIx64 " r13=%016" PRIx64 " r14=%016" PRIx64 " r15=%016" PRIx64
	     "\n",
	     context->R12, context->R13, context->R14, context->R15);
	const auto* code = reinterpret_cast<const uint8_t*>(context->Rip - 0x20);
	LOGF("\t HWWATCH:   code@rip-0x20:");
	for (uint32_t offset = 0; offset < 0x40; offset++) {
		LOGF("%s%02x", offset == 0x20 ? " |" : " ", code[offset]);
	}
	LOGF("\n");
	const auto* stack = reinterpret_cast<const uint64_t*>(context->Rsp);
	LOGF("\t HWWATCH:   stack:");
	for (uint32_t slot = 0; slot < 8; slot++) {
		LOGF(" %016" PRIx64, stack[slot]);
	}
	LOGF("\n");
	context->Dr6 = 0;
	return EXCEPTION_CONTINUE_EXECUTION;
}

static void StartHardwareWatch() {
	const char* text = std::getenv("KYTY_HW_WATCH");
	if (text == nullptr) {
		return;
	}
	g_hw_watch_address = std::strtoull(text, nullptr, 16);
	if (g_hw_watch_address == 0 || (g_hw_watch_address & 7u) != 0) {
		printf("KYTY_HW_WATCH needs an 8-byte aligned hex address\n");
		return;
	}
	AddVectoredExceptionHandler(1, HardwareWatchHandler);
	printf("Hardware write watch armed on 0x%016llx\n",
	       static_cast<unsigned long long>(g_hw_watch_address));

	std::thread([] {
		std::unordered_map<DWORD, bool> armed;
		const auto                      process = GetCurrentProcessId();
		const auto                      self    = GetCurrentThreadId();
		for (;;) {
			auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
			if (snapshot != INVALID_HANDLE_VALUE) {
				THREADENTRY32 entry {};
				entry.dwSize = sizeof(entry);
				if (Thread32First(snapshot, &entry) != 0) {
					do {
						if (entry.th32OwnerProcessID != process ||
						    entry.th32ThreadID == self || armed[entry.th32ThreadID]) {
							continue;
						}
						auto* handle = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT |
						                              THREAD_SUSPEND_RESUME,
						                          FALSE, entry.th32ThreadID);
						if (handle == nullptr) {
							continue;
						}
						if (SuspendThread(handle) != static_cast<DWORD>(-1)) {
							CONTEXT context {};
							context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
							if (GetThreadContext(handle, &context) != 0) {
								context.Dr0 = g_hw_watch_address;
								// L0 enabled, break on write, 8-byte length.
								context.Dr7 = (context.Dr7 & ~0xf0003ull) |
								              0xd0001ull;
								context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
								if (SetThreadContext(handle, &context) != 0) {
									armed[entry.th32ThreadID] = true;
								}
							}
							ResumeThread(handle);
						}
						CloseHandle(handle);
					} while (Thread32Next(snapshot, &entry) != 0);
				}
				CloseHandle(snapshot);
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
	}).detach();
}
#endif

static int Main(int argc, char* argv[]) {
	VirtualMemory::Init();
	InitializeThreads();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	StartSpinWatchdog();
	StartGpuSampler();
	StartHardwareWatch();
	StartSampler();
	StartHangWatch();
	StartThreadDump();
#endif

	RunOptions options;
	bool       show_help = false;

	if (argc < 2) {
		PrintUsage();
		return 0;
	}

	if (!ParseArgs(argc, argv, options, show_help)) {
		PrintUsage();
		return 1;
	}

	if (show_help) {
		PrintUsage();
		return 0;
	}

	Run(options);

	return 0;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

int wmain(int argc, wchar_t* argv[]) {
    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<size_t>(argc));

    for (int index = 0; index < argc; index++) {
        const std::wstring_view wide(argv[index]);
        const std::u16string utf16(wide.begin(), wide.end());

        utf8_args.push_back(Common::Utf16ToUtf8(utf16));
    }

    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_args.size());

    for (auto& argument: utf8_args) {
        utf8_argv.push_back(argument.data());
    }

    return Main(argc, utf8_argv.data());
}

#else

int main(int argc, char* argv[]) {
    return Main(argc, argv);
}

#endif
