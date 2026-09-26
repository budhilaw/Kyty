#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "kernel/memory.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <string>
#include <fmt/format.h>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, uint32_t*) { return false; };
	return runtime;
}

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Fetch: return "fetch";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

std::string Diagnostic(const ResourcePlan& program, uint32_t pc, const std::string& message) {
	return fmt::format("shader SRT: hash=0x{:016x} stage={} pc=0x{:08x} {}", program.shader_hash,
	                   StageName(program.stage), pc, message);
}

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto kind = values.memory_info[index].kind;
	return (op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
	       (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMin32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type)
	    : m_program(program), m_type(type) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type == RuntimeValueType::Integer &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type == RuntimeValueType::Integer && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};

class PlanBuilder {
public:
	explicit PlanBuilder(Program& program): m_program(program) {}

	void Run() {
		m_program.srt_reads.clear();
		m_program.dynamic_reads.clear();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
					const auto flags = inst.Flags<MemoryFlags>();
					if (flags.index < m_program.memory_info.size()) {
						const auto kind       = m_program.memory_info[flags.index].kind;
						const bool crosswired = (op == ValueOpcode::LoadAddressU32 &&
						                         kind == ResourceKind::ScalarBuffer) ||
						                        (op == ValueOpcode::ReadConstBuffer &&
						                         kind == ResourceKind::ScalarAddress);
						if (crosswired) {
							Fail(flags.pc,
							     fmt::format("{} has incompatible scalar memory metadata",
							                 ValueOpcodeName(op)));
						}
					}
				}
				if (IsDescriptorHandle(inst.GetOpcode())) {
					for (size_t index = 0; index < inst.NumArgs(); index++) {
						Collect(inst.Arg(index), 0);
					}
				}
			}
		}
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 && IsRawRead(m_program, inst) &&
				    inst.Arg(1).Resolve().IsImmediate() &&
				    ValidateRuntimeValue(m_program, Value(&inst))) {
					Collect(Value(&inst), inst.Flags<MemoryFlags>().pc);
				}
			}
		}
		PatchReads();
	}

private:
	struct Patch {
		Inst*    inst = nullptr;
		uint32_t slot = 0;
		bool     keep = false;
	};

	[[noreturn]] void Fail(uint32_t pc, const std::string& message) const {
		const auto diagnostic = Diagnostic(m_program, pc, message);
		EXIT("shader SRT planning failed: %s", diagnostic.c_str());
		std::abort();
	}

	void Collect(Value value, uint32_t use_pc) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return;
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			Fail(use_pc, "invalid typed planning value");
		}
		const auto cycle = std::ranges::find(m_visiting, inst);
		if (cycle != m_visiting.end()) {
			const auto contains_phi = std::any_of(cycle, m_visiting.end(), [](const Inst* value) {
				return value->GetOpcode() == ValueOpcode::Phi;
			});
			if (contains_phi) {
				return;
			}
			Fail(use_pc, fmt::format("cyclic typed planning value {} without a phi",
			                         ValueOpcodeName(inst->GetOpcode())));
		}
		if (std::ranges::find(m_visited, inst) != m_visited.end()) {
			return;
		}
		m_visiting.push_back(inst);
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			Collect(inst->Arg(index), use_pc);
		}
		m_visiting.pop_back();
		m_visited.push_back(inst);
		if (!IsRawRead(m_program, *inst)) {
			return;
		}
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) {
			if (std::ranges::find(m_program.dynamic_reads, value) ==
			    m_program.dynamic_reads.end()) {
				m_program.dynamic_reads.push_back(value);
			}
			return;
		}
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); slot++) {
			if (EquivalentValue(m_program, value, m_program.srt_reads[slot].value)) {
				m_patches.push_back({inst, slot, false});
				return;
			}
		}
		const auto slot = static_cast<uint32_t>(m_program.srt_reads.size());
		m_program.srt_reads.push_back({value, slot});
		m_patches.push_back({inst, slot, true});
	}

	void PatchReads() {
		for (const auto& patch: m_patches) {
			auto* block = patch.inst->Parent();
			auto& list  = block->Instructions();
			auto  where =
			    std::ranges::find_if(list, [&](const Inst& inst) { return &inst == patch.inst; });
			const auto resource =
			    Value(&*block->PrependNewInst(where, ValueOpcode::GetSrtResource));
			const auto flat = Value(&*block->PrependNewInst(where, ValueOpcode::ReadConst,
			                                                {resource, Value(patch.slot)}));
			const auto uses = patch.inst->Uses();
			for (const auto& use: uses) {
				use.user->SetArg(use.operand, flat);
			}
			for (auto& info: m_program.block_info) {
				if (info.condition.Resolve() == Value(patch.inst)) {
					info.condition = flat;
				}
				if (info.indirect_target.Resolve() == Value(patch.inst)) {
					info.indirect_target = flat;
				}
			}
			if (patch.keep) {
				const auto memory = patch.inst->Flags<MemoryFlags>().index;
				if (memory < m_program.memory_info.size()) {
					m_program.memory_info[memory].planning_only = true;
				}
				block->AppendNewInst(ValueOpcode::ReferenceU32, {Value(patch.inst)});
			}
		}
	}

	Program&           m_program;
	std::vector<Inst*> m_visiting;
	std::vector<Inst*> m_visited;
	std::vector<Patch> m_patches;
};

thread_local std::string t_srt_failure;

// Evaluated values of every live evaluator on this thread, stamped with the owning evaluator's
// epoch. Zeroing two plan-sized arrays per evaluator (several per draw) and a hash lookup per
// value were a large part of per-draw resource setup; a stale stamp now simply means "absent".
struct EvaluationSlots {
	std::vector<uint64_t> values;
	std::vector<uint32_t> stamps;
	uint32_t              epoch = 0;

	uint32_t NextEpoch() {
		if (++epoch == 0) {
			std::fill(stamps.begin(), stamps.end(), 0u);
			epoch = 1;
		}
		return epoch;
	}
	void Reserve(size_t count) {
		if (values.size() < count) {
			values.resize(count);
			stamps.resize(count, 0u);
		}
	}
};
thread_local EvaluationSlots t_evaluation_slots;

class Evaluator {
public:
	Evaluator(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, Evaluator* clean_evaluator = nullptr,
	          Value active_mask = {})
	    : m_program(program), m_runtime(runtime), m_clean_flat_slots(clean_flat_slots),
	      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
	      m_epoch(t_evaluation_slots.NextEpoch()) {}

	bool Evaluate(Value value, uint32_t& result) {
		uint64_t wide = 0;
		if (!EvaluateWide(value, wide)) {
			return false;
		}
		result = static_cast<uint32_t>(wide);
		return true;
	}

private:
	static float Float32(uint64_t bits) {
		return std::bit_cast<float>(static_cast<uint32_t>(bits));
	}

	bool EvaluateWide(Value value, uint64_t& result) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			switch (value.GetType()) {
				case Type::U1: result = value.U1(); return true;
				case Type::U8: result = value.U8(); return true;
				case Type::U16: result = value.U16(); return true;
				case Type::U32: result = value.U32(); return true;
				case Type::U64: result = value.U64(); return true;
				case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); return true;
				default: return false;
			}
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return false;
		}
		Prepare();
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
		    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
			return EvaluateWide(inst->Arg(1), result);
		}
		return EvaluateIndexed(inst, result);
	}

public:
	// A root compiled with the plan (an SRT read or a descriptor-source dword).
	bool EvaluateRoot(const CompiledEvalOperand& operand, uint32_t& result) {
		Prepare();
		uint64_t wide = 0;
		if (!Operand(operand, wide)) {
			return false;
		}
		result = static_cast<uint32_t>(wide);
		return true;
	}

	void Prepare() {
		if (!m_reserved) {
			// One flat slot per plan value instead of a hash node per evaluated value: this
			// evaluation runs for hundreds of draws per frame. Plan values carry their slot.
			if (m_program.evaluation_numbered != m_program.value_storage.size()) {
				uint32_t count = 0;
				for (const auto& stored: m_program.value_storage) {
					const auto index = stored.AssignedEvaluationIndex();
					if (index != UINT32_MAX) {
						count = std::max(count, index + 1u);
					}
				}
				for (const auto& stored: m_program.value_storage) {
					(void)stored.EvaluationIndex(count);
				}
				m_program.evaluation_numbered = m_program.value_storage.size();
				BuildCompiled(count);
			} else if (m_program.compiled_eval.empty() && !m_program.value_storage.empty()) {
				uint32_t count = 0;
				for (const auto& stored: m_program.value_storage) {
					const auto index = stored.AssignedEvaluationIndex();
					if (index != UINT32_MAX) {
						count = std::max(count, index + 1u);
					}
				}
				BuildCompiled(count);
			}
			t_evaluation_slots.Reserve(m_program.compiled_eval.size());
			m_reserved = true;
		}
	}

private:
	bool EvaluateIndexed(const Inst* inst, uint64_t& result) {
		const auto slot = inst->AssignedEvaluationIndex();
		if (slot != UINT32_MAX && slot < m_program.compiled_eval.size()) {
			return EvaluateSlot(slot, result);
		}
		auto& slots = t_evaluation_slots;
		if (slot != UINT32_MAX) {
			slots.Reserve(size_t {slot} + 1u);
			if (slots.stamps[slot] == m_epoch) {
				result = slots.values[slot];
				return true;
			}
		} else if (const auto found = std::ranges::find(m_cache, inst, &std::pair<const Inst*, uint64_t>::first);
		           found != m_cache.end()) {
			result = found->second;
			return true;
		}
		if (std::ranges::find(m_visiting, inst) != m_visiting.end()) {
			return false;
		}
		m_visiting.push_back(inst);
		uint64_t out = 0;
		const bool evaluated = EvaluateInst(*inst, out);
		m_visiting.pop_back();
		if (!evaluated) {
			if (t_srt_failure.empty()) {
				t_srt_failure = fmt::format("{} ({} args)", ValueOpcodeName(inst->GetOpcode()),
				                            inst->NumArgs());
			}
			return false;
		}
		if (slot != UINT32_MAX) {
			slots.values[slot] = out;
			slots.stamps[slot] = m_epoch;
		} else {
			m_cache.emplace_back(inst, out);
		}
		result = out;
		return true;
	}

	// --- Compiled path -------------------------------------------------------------------------
	// Arithmetic plan values are evaluated from CompiledEvalNode (operands resolved once per
	// shader); everything with special semantics goes through EvaluateInst. Both share the slot
	// cache, so the result is the interpreter's. Value::Resolve / Inst::Arg per operand and per draw
	// were the evaluator's largest self cost. KYTY_NO_COMPILED_SRT=1 disables the fast operations.

	static bool IsFastOp(ValueOpcode op) {
		switch (op) {
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::CompositeConstructU64:
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::IMul64:
			case ValueOpcode::UMin32:
			case ValueOpcode::ConvertF32U32:
			case ValueOpcode::ConvertU32F32:
			case ValueOpcode::FPMul32:
			case ValueOpcode::FPTrunc32:
			case ValueOpcode::FPIsNan32:
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::BitwiseNot32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftLeftLogical64:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightLogical64:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::ShiftRightArithmetic64:
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract:
			case ValueOpcode::BitFieldInsert:
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::SGreaterThanEqual32:
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr:
			case ValueOpcode::LogicalXor:
			case ValueOpcode::LogicalNot: return true;
			default: return false;
		}
	}

	CompiledEvalOperand CompileOperand(Value value, uint32_t count) const {
		CompiledEvalOperand operand;
		value = value.Resolve();
		if (value.IsImmediate()) {
			operand.kind = CompiledEvalOperand::Immediate;
			switch (value.GetType()) {
				case Type::U1: operand.imm = value.U1(); break;
				case Type::U8: operand.imm = value.U8(); break;
				case Type::U16: operand.imm = value.U16(); break;
				case Type::U32: operand.imm = value.U32(); break;
				case Type::U64: operand.imm = value.U64(); break;
				case Type::F32: operand.imm = std::bit_cast<uint32_t>(value.F32Value()); break;
				default: operand.kind = CompiledEvalOperand::Invalid; break;
			}
			return operand;
		}
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return operand; // Invalid
		}
		const auto source_index = source->AssignedEvaluationIndex();
		if (source_index != UINT32_MAX && source_index < count) {
			operand.kind = CompiledEvalOperand::Node;
			operand.node = source_index;
		} else {
			operand.kind  = CompiledEvalOperand::Generic;
			operand.value = value;
		}
		return operand;
	}

	void BuildCompiled(uint32_t count) {
		static const bool disabled = std::getenv("KYTY_NO_COMPILED_SRT") != nullptr;
		auto&             nodes    = m_program.compiled_eval;
		nodes.assign(count, CompiledEvalNode {});
		// Roots: the evaluation entry points of every draw.
		const auto root = [&](Value value) {
			if (disabled) {
				CompiledEvalOperand operand;
				operand.kind  = CompiledEvalOperand::Generic;
				operand.value = value;
				return operand;
			}
			return CompileOperand(value, count);
		};
		m_program.compiled_srt_roots.clear();
		for (const auto& read: m_program.srt_reads) {
			m_program.compiled_srt_roots.push_back(root(read.value));
		}
		m_program.compiled_source_roots.assign(m_program.descriptor_sources.size(), {});
		for (size_t s = 0; s < m_program.descriptor_sources.size(); s++) {
			const auto& source = m_program.descriptor_sources[s];
			for (uint32_t d = 0; d < source.dword_count && d < 8u; d++) {
				m_program.compiled_source_roots[s][d] = root(source.dwords[d]);
			}
		}
		for (const auto& stored: m_program.value_storage) {
			const auto index = stored.AssignedEvaluationIndex();
			if (index == UINT32_MAX || index >= count) {
				continue;
			}
			auto& node = nodes[index];
			node.inst  = &stored;
			node.op    = stored.GetOpcode();
			if (disabled) {
				continue;
			}
			const auto compile = [&](Value value) { return CompileOperand(value, count); };
			const auto args = stored.NumArgs();
			if (IsFastOp(node.op) && args <= node.operands.size()) {
				node.fast = true;
				node.args = static_cast<uint8_t>(args);
				for (size_t k = 0; k < args; k++) {
					node.operands[k] = compile(stored.Arg(k));
				}
				continue;
			}
			switch (node.op) {
				case ValueOpcode::GetUserData:
					node.special = CompiledEvalNode::UserData;
					node.aux     = RegIndex(stored.Arg(0).ScalarRegister());
					break;
				case ValueOpcode::GetShaderBase: node.special = CompiledEvalNode::ShaderBase; break;
				case ValueOpcode::ReadConst: {
					const auto slot = stored.NumArgs() >= 2 ? stored.Arg(1).Resolve() : Value {};
					if (!slot.IsEmpty() && slot.IsImmediate() && slot.GetType() == Type::U32 &&
					    slot.U32() < m_program.srt_reads.size()) {
						node.special     = CompiledEvalNode::ReadConst;
						node.aux         = slot.U32();
						node.operands[0] = compile(m_program.srt_reads[slot.U32()].value);
					}
					break;
				}
				case ValueOpcode::SelectU32:
				case ValueOpcode::SelectU1:
				case ValueOpcode::SelectF32:
					if (args == 3) {
						node.special = CompiledEvalNode::Select;
						for (size_t k = 0; k < 3; k++) {
							node.operands[k] = compile(stored.Arg(k));
						}
					}
					break;
				case ValueOpcode::LoadAddressU32:
				case ValueOpcode::ReadConstBuffer: {
					if (!IsRawRead(m_program, stored)) {
						break;
					}
					const auto flags = stored.Flags<MemoryFlags>();
					const auto* handle = stored.Arg(0).ResolveInstruction();
					if (flags.index >= m_program.memory_info.size() || handle == nullptr ||
					    handle->NumArgs() < 2 || stored.NumArgs() < 2) {
						break;
					}
					if (node.op == ValueOpcode::ReadConstBuffer && handle->NumArgs() != 4u) {
						break;
					}
					node.special = CompiledEvalNode::RawRead;
					node.aux     = flags.index;
					node.four    = handle->NumArgs() == 4u;
					for (size_t k = 0; k < std::min<size_t>(handle->NumArgs(), 4u); k++) {
						node.operands[k] = compile(handle->Arg(k));
					}
					node.operands[4] = compile(stored.Arg(1));
					break;
				}
				default: break;
			}
		}
	}

	bool EvaluateSpecial(const CompiledEvalNode& node, uint64_t& result) {
		switch (node.special) {
			case CompiledEvalNode::UserData: {
				const auto reg = node.aux;
				if (reg < m_program.user_data_base ||
				    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
					return false;
				}
				result = m_runtime.user_data[reg - m_program.user_data_base];
				return true;
			}
			case CompiledEvalNode::ShaderBase: result = m_runtime.shader_base; return true;
			case CompiledEvalNode::ReadConst:
				if (node.aux < m_clean_flat_slots.size() && m_clean_flat_slots[node.aux] != 0u &&
				    m_clean_evaluator != nullptr) {
					return m_clean_evaluator->Operand(node.operands[0], result);
				}
				return Operand(node.operands[0], result);
			case CompiledEvalNode::Select: {
				auto&    predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
				uint64_t a         = 0;
				if (!predicate.Operand(node.operands[0], a)) {
					return false;
				}
				return Operand(node.operands[a != 0u ? 1u : 2u], result);
			}
			case CompiledEvalNode::RawRead: {
				const auto& mem    = m_program.memory_info[node.aux];
				uint64_t    low    = 0;
				uint64_t    high   = 0;
				uint64_t    offset = 0;
				if (!Operand(node.operands[0], low) || !Operand(node.operands[1], high) ||
				    !Operand(node.operands[4], offset)) {
					return false;
				}
				const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
				const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
				uint64_t   address   = 0;
				if (node.op == ValueOpcode::ReadConstBuffer) {
					uint64_t records = 0;
					uint64_t word3   = 0;
					if (!node.four || !Operand(node.operands[2], records) ||
					    !Operand(node.operands[3], word3)) {
						return false;
					}
					if (immediate < 0) {
						return false;
					}
					const auto byte_offset =
					    static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
					const auto aligned = byte_offset & ~uint64_t {3};
					const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
					const auto size    = stride == 0u
					                         ? static_cast<uint64_t>(static_cast<uint32_t>(records))
					                         : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
					if (aligned > size || size - aligned < sizeof(uint32_t)) {
						return false;
					}
					address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
				} else {
					const auto relative = (immediate & ~int64_t {3}) +
					                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
					if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
						return false;
					}
				}
				uint32_t   word = 0;
				const bool read = m_runtime.read_memory != nullptr
				                      ? m_runtime.read_memory(m_runtime.userdata, address, &word)
				                      : LibKernel::Memory::TryReadBacking(address, &word, sizeof(word));
				if (!read) {
					if (node.op != ValueOpcode::LoadAddressU32) {
						if (t_srt_failure.empty()) {
							t_srt_failure = fmt::format("{} read at 0x{:x} failed",
							                            ValueOpcodeName(node.op), address);
						}
						return false;
					}
					word = 0;
				}
				result = word;
				return true;
			}
			default: return EvaluateInst(*node.inst, result);
		}
	}

	bool Operand(const CompiledEvalOperand& operand, uint64_t& result) {
		switch (operand.kind) {
			case CompiledEvalOperand::Immediate: result = operand.imm; return true;
			case CompiledEvalOperand::Node: {
				const auto& slots = t_evaluation_slots;
				if (slots.stamps[operand.node] == m_epoch) {
					result = slots.values[operand.node];
					return true;
				}
				const auto& node = m_program.compiled_eval[operand.node];
				if (!m_active_mask.IsEmpty() && IsRuntimeSelect(node.op)) {
					return EvaluateWide(Value(const_cast<Inst*>(node.inst)), result);
				}
				return EvaluateSlot(operand.node, result);
			}
			case CompiledEvalOperand::Generic: return EvaluateWide(operand.value, result);
			default: return false;
		}
	}

	bool EvaluateSlot(uint32_t slot, uint64_t& result) {
		auto& slots = t_evaluation_slots;
		if (slots.stamps[slot] == m_epoch) {
			result = slots.values[slot];
			return true;
		}
		const auto& node = m_program.compiled_eval[slot];
		const auto* inst = node.inst;
		if (std::ranges::find(m_visiting, inst) != m_visiting.end()) {
			return false;
		}
		m_visiting.push_back(inst);
		uint64_t   out       = 0;
		const bool evaluated = node.fast ? EvaluateFast(node, out) : EvaluateSpecial(node, out);
		m_visiting.pop_back();
		if (!evaluated) {
			if (t_srt_failure.empty()) {
				t_srt_failure = fmt::format("{} ({} args)", ValueOpcodeName(inst->GetOpcode()),
				                            inst->NumArgs());
			}
			return false;
		}
		// The slot arrays may have grown during the recursion.
		t_evaluation_slots.values[slot] = out;
		t_evaluation_slots.stamps[slot] = m_epoch;
		result                          = out;
		return true;
	}

	bool EvaluateFast(const CompiledEvalNode& node, uint64_t& result) {
		uint64_t   a = 0;
		uint64_t   b = 0;
		uint64_t   c = 0;
		uint64_t   d = 0;
		const auto unary   = [&]() { return node.args >= 1 && Operand(node.operands[0], a); };
		const auto binary  = [&]() { return node.args >= 2 && unary() && Operand(node.operands[1], b); };
		const auto ternary = [&]() { return node.args >= 3 && binary() && Operand(node.operands[2], c); };
		switch (node.op) {
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
				if (unary()) {
					result = a;
					return true;
				}
				return false;
			case ValueOpcode::CompositeConstructU64:
				if (!binary()) return false;
				result = static_cast<uint32_t>(a) | (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
				return true;
			case ValueOpcode::IAdd32: if (!binary()) return false; result = static_cast<uint32_t>(a + b); return true;
			case ValueOpcode::IAdd64: if (!binary()) return false; result = a + b; return true;
			case ValueOpcode::ISub32: if (!binary()) return false; result = static_cast<uint32_t>(a - b); return true;
			case ValueOpcode::ISub64: if (!binary()) return false; result = a - b; return true;
			case ValueOpcode::IMul32: if (!binary()) return false; result = static_cast<uint32_t>(a * b); return true;
			case ValueOpcode::IMul64: if (!binary()) return false; result = a * b; return true;
			case ValueOpcode::UMin32:
				if (!binary()) return false;
				result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
				return true;
			case ValueOpcode::ConvertF32U32:
				if (!unary()) return false;
				result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
				return true;
			case ValueOpcode::ConvertU32F32: {
				if (!unary()) return false;
				const auto value = Float32(a);
				if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > UINT32_MAX) {
					return false;
				}
				result = static_cast<uint32_t>(value);
				return true;
			}
			case ValueOpcode::FPMul32:
				if (!binary()) return false;
				result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
				return true;
			case ValueOpcode::FPTrunc32:
				if (!unary()) return false;
				result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
				return true;
			case ValueOpcode::FPIsNan32: if (!unary()) return false; result = std::isnan(Float32(a)); return true;
			case ValueOpcode::FPOrdLessThanEqual32:
				if (!binary()) return false;
				result = Float32(a) <= Float32(b);
				return true;
			case ValueOpcode::FPOrdGreaterThanEqual32:
				if (!binary()) return false;
				result = Float32(a) >= Float32(b);
				return true;
			case ValueOpcode::BitwiseAnd32: if (!binary()) return false; result = static_cast<uint32_t>(a & b); return true;
			case ValueOpcode::BitwiseAnd64: if (!binary()) return false; result = a & b; return true;
			case ValueOpcode::BitwiseOr32: if (!binary()) return false; result = static_cast<uint32_t>(a | b); return true;
			case ValueOpcode::BitwiseXor32: if (!binary()) return false; result = static_cast<uint32_t>(a ^ b); return true;
			case ValueOpcode::BitwiseNot32: if (!unary()) return false; result = ~static_cast<uint32_t>(a); return true;
			case ValueOpcode::ShiftLeftLogical32:
				if (!binary()) return false;
				result = static_cast<uint32_t>(a) << (b & 31u);
				return true;
			case ValueOpcode::ShiftLeftLogical64: if (!binary()) return false; result = a << (b & 63u); return true;
			case ValueOpcode::ShiftRightLogical32:
				if (!binary()) return false;
				result = static_cast<uint32_t>(a) >> (b & 31u);
				return true;
			case ValueOpcode::ShiftRightLogical64: if (!binary()) return false; result = a >> (b & 63u); return true;
			case ValueOpcode::ShiftRightArithmetic32:
				if (!binary()) return false;
				result = static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
				return true;
			case ValueOpcode::ShiftRightArithmetic64:
				if (!binary()) return false;
				result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
				return true;
			case ValueOpcode::BitFieldUExtract: {
				if (!ternary()) return false;
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) return false;
				const auto mask = width == 32u ? UINT32_MAX : width == 0u ? 0u : (uint32_t {1} << width) - 1u;
				result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
				return true;
			}
			case ValueOpcode::BitFieldSExtract: {
				if (!ternary()) return false;
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) return false;
				if (width == 0u) {
					result = 0;
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				result = bits;
				return true;
			}
			case ValueOpcode::BitFieldInsert: {
				if (!ternary() || node.args < 4 || !Operand(node.operands[3], d)) return false;
				const auto offset = static_cast<uint32_t>(c);
				const auto width  = static_cast<uint32_t>(d);
				if (offset > 32u || width > 32u - offset) return false;
				if (width == 0u) {
					result = static_cast<uint32_t>(a);
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
				result = (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
				return true;
			}
			case ValueOpcode::IEqual32:
				if (!binary()) return false;
				result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
				return true;
			case ValueOpcode::INotEqual32:
				if (!binary()) return false;
				result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
				return true;
			case ValueOpcode::ULessThan32:
				if (!binary()) return false;
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			case ValueOpcode::UGreaterThan32:
				if (!binary()) return false;
				result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
				return true;
			case ValueOpcode::SGreaterThanEqual32:
				if (!binary()) return false;
				result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
				         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
				return true;
			case ValueOpcode::LogicalAnd: if (!binary()) return false; result = (a != 0u) && (b != 0u); return true;
			case ValueOpcode::LogicalOr: if (!binary()) return false; result = (a != 0u) || (b != 0u); return true;
			case ValueOpcode::LogicalXor: if (!binary()) return false; result = (a != 0u) != (b != 0u); return true;
			case ValueOpcode::LogicalNot: if (!unary()) return false; result = a == 0u; return true;
			default: return false;
		}
	}

	bool Arg(const Inst& inst, size_t index, uint64_t& result) {
		return EvaluateWide(inst.Arg(index), result);
	}

	bool EvaluatePhi(const Inst& inst, uint64_t& result) {
		const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
		return !value.IsEmpty() && EvaluateWide(value, result);
	}

	bool EvaluateExtract(const Inst& inst, uint64_t& result) {
		const auto index = inst.Arg(1).Resolve();
		if (!index.IsImmediate() || index.GetType() != Type::U32) {
			return false;
		}
		const auto component = index.U32();
		if (component >= 2u) {
			return false;
		}
		if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
			uint64_t packed = 0;
			if (!Arg(inst, 0, packed)) {
				return false;
			}
			result = static_cast<uint32_t>(packed >> (component * 32u));
			return true;
		}
		const auto* source = inst.Arg(0).ResolveInstruction();
		if (source == nullptr) {
			return false;
		}
		if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
			return EvaluateWide(source->Arg(component), result);
		}
		if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
			uint64_t lhs = 0;
			uint64_t rhs = 0;
			if (!Arg(*source, 0, lhs) || !Arg(*source, 1, rhs)) {
				return false;
			}
			const auto sum =
			    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
			result =
			    component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
			return true;
		}
		return false;
	}

	bool EvaluateRawRead(const Inst& inst, uint64_t& result) {
		const auto flags = inst.Flags<MemoryFlags>();
		if (flags.index >= m_program.memory_info.size()) {
			return false;
		}
		const auto& mem    = m_program.memory_info[flags.index];
		const auto* handle = inst.Arg(0).ResolveInstruction();
		if (handle == nullptr) {
			return false;
		}
		uint64_t low    = 0;
		uint64_t high   = 0;
		uint64_t offset = 0;
		if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
			return false;
		}
		const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
		const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
		uint64_t   address   = 0;
		if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
			uint64_t records = 0;
			uint64_t word3   = 0;
			if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
				return false;
			}
			if (immediate < 0) {
				return false;
			}
			const auto byte_offset =
			    static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
			const auto aligned = byte_offset & ~uint64_t {3};
			const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
			const auto size = stride == 0u
			                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
			                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
			if (aligned > size || size - aligned < sizeof(uint32_t)) {
				return false;
			}
			address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
		} else {
			const auto relative = (immediate & ~int64_t {3}) +
			                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
			if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
				return false;
			}
		}
		uint32_t word = 0;
		const bool read = m_runtime.read_memory != nullptr
		                      ? m_runtime.read_memory(m_runtime.userdata, address, &word)
		                      : LibKernel::Memory::TryReadBacking(address, &word, sizeof(word));
		if (!read) {
			if (inst.GetOpcode() != ValueOpcode::LoadAddressU32) {
				if (t_srt_failure.empty()) {
					t_srt_failure = fmt::format("{} read at 0x{:x} failed",
					                            ValueOpcodeName(inst.GetOpcode()), address);
				}
				return false;
			}
			// A pointer into unmapped memory belongs to a stale table; it selects nothing.
			word = 0;
		}
		result = word;
		return true;
	}

	bool EvaluateInst(const Inst& inst, uint64_t& result) {
		uint64_t   a       = 0;
		uint64_t   b       = 0;
		uint64_t   c       = 0;
		const auto binary  = [&]() { return Arg(inst, 0, a) && Arg(inst, 1, b); };
		const auto ternary = [&]() {
			return Arg(inst, 0, a) && Arg(inst, 1, b) && Arg(inst, 2, c);
		};
		switch (inst.GetOpcode()) {
			case ValueOpcode::GetUserData: {
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base ||
				    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
					return false;
				}
				result = m_runtime.user_data[reg - m_program.user_data_base];
				return true;
			}
			case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
			case ValueOpcode::Phi: return EvaluatePhi(inst, result);
			case ValueOpcode::ReadFirstLane: {
				const auto clean_runtime = CleanRuntime(m_runtime);
				Evaluator  clean_active(m_program, clean_runtime, {}, nullptr, inst.Arg(1));
				Evaluator  active(m_program, m_runtime, m_clean_flat_slots, &clean_active,
				                  inst.Arg(1));
				return active.EvaluateWide(inst.Arg(0), result);
			}
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
			case ValueOpcode::CompositeConstructU64:
				if (!binary()) {
					return false;
				}
				result = static_cast<uint32_t>(a) |
				         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
				return true;
			case ValueOpcode::ReadConst: {
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
				    slot.U32() >= m_program.srt_reads.size()) {
					return false;
				}
				if (slot.U32() < m_clean_flat_slots.size() &&
				    m_clean_flat_slots[slot.U32()] != 0u && m_clean_evaluator != nullptr) {
					return m_clean_evaluator->EvaluateWide(m_program.srt_reads[slot.U32()].value,
					                                       result);
				}
				return EvaluateWide(m_program.srt_reads[slot.U32()].value, result);
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer:
				if (IsRawRead(m_program, inst)) {
					return EvaluateRawRead(inst, result);
				}
				break;
			case ValueOpcode::IAdd32:
				if (binary()) {
					result = static_cast<uint32_t>(a + b);
					return true;
				}
				return false;
			case ValueOpcode::IAdd64:
				if (binary()) {
					result = a + b;
					return true;
				}
				return false;
			case ValueOpcode::ISub32:
				if (binary()) {
					result = static_cast<uint32_t>(a - b);
					return true;
				}
				return false;
			case ValueOpcode::ISub64:
				if (binary()) {
					result = a - b;
					return true;
				}
				return false;
			case ValueOpcode::IMul32:
				if (binary()) {
					result = static_cast<uint32_t>(a * b);
					return true;
				}
				return false;
			case ValueOpcode::IMul64:
				if (binary()) {
					result = a * b;
					return true;
				}
				return false;
			case ValueOpcode::UMin32:
				if (binary()) {
					result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
					return true;
				}
				return false;
			case ValueOpcode::ConvertF32U32:
				if (Arg(inst, 0, a)) {
					result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
					return true;
				}
				return false;
			case ValueOpcode::ConvertU32F32:
				if (Arg(inst, 0, a)) {
					const auto value = Float32(a);
					if (!std::isfinite(value) || value < 0.0f ||
					    static_cast<double>(value) > UINT32_MAX) {
						return false;
					}
					result = static_cast<uint32_t>(value);
					return true;
				}
				return false;
			case ValueOpcode::FPMul32:
				if (binary()) {
					result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
					return true;
				}
				return false;
			case ValueOpcode::FPTrunc32:
				if (Arg(inst, 0, a)) {
					result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
					return true;
				}
				return false;
			case ValueOpcode::FPIsNan32:
				if (Arg(inst, 0, a)) {
					result = std::isnan(Float32(a));
					return true;
				}
				return false;
			case ValueOpcode::FPOrdLessThanEqual32:
				if (binary()) {
					result = Float32(a) <= Float32(b);
					return true;
				}
				return false;
			case ValueOpcode::FPOrdGreaterThanEqual32:
				if (binary()) {
					result = Float32(a) >= Float32(b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseAnd32:
				if (binary()) {
					result = static_cast<uint32_t>(a & b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseAnd64:
				if (binary()) {
					result = a & b;
					return true;
				}
				return false;
			case ValueOpcode::BitwiseOr32:
				if (binary()) {
					result = static_cast<uint32_t>(a | b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseXor32:
				if (binary()) {
					result = static_cast<uint32_t>(a ^ b);
					return true;
				}
				return false;
			case ValueOpcode::BitwiseNot32:
				if (Arg(inst, 0, a)) {
					result = ~static_cast<uint32_t>(a);
					return true;
				}
				return false;
			case ValueOpcode::ShiftLeftLogical32:
				if (binary()) {
					result = static_cast<uint32_t>(a) << (b & 31u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftLeftLogical64:
				if (binary()) {
					result = a << (b & 63u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightLogical32:
				if (binary()) {
					result = static_cast<uint32_t>(a) >> (b & 31u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightLogical64:
				if (binary()) {
					result = a >> (b & 63u);
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightArithmetic32:
				if (binary()) {
					result = static_cast<uint32_t>(
					    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
					return true;
				}
				return false;
			case ValueOpcode::ShiftRightArithmetic64:
				if (binary()) {
					result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
					return true;
				}
				return false;
			case ValueOpcode::BitFieldUExtract:
				if (ternary()) {
					const auto offset = static_cast<uint32_t>(b);
					const auto width  = static_cast<uint32_t>(c);
					if (offset > 32u || width > 32u - offset) {
						return false;
					}
					const auto mask = width == 32u  ? UINT32_MAX
					                  : width == 0u ? 0u
					                                : (uint32_t {1} << width) - 1u;
					result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
					return true;
				}
				return false;
			case ValueOpcode::BitFieldSExtract:
				if (ternary()) {
					const auto offset = static_cast<uint32_t>(b);
					const auto width  = static_cast<uint32_t>(c);
					if (offset > 32u || width > 32u - offset) {
						return false;
					}
					if (width == 0u) {
						result = 0;
						return true;
					}
					const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
					auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
					if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
						bits |= ~mask;
					}
					result = bits;
					return true;
				}
				return false;
			case ValueOpcode::BitFieldInsert: {
				uint64_t d = 0;
				if (!ternary() || !Arg(inst, 3, d)) {
					return false;
				}
				const auto offset = static_cast<uint32_t>(c);
				const auto width  = static_cast<uint32_t>(d);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = static_cast<uint32_t>(a);
					return true;
				}
				const auto mask =
				    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
				result = (static_cast<uint32_t>(a) & ~mask) |
				         ((static_cast<uint32_t>(b) << offset) & mask);
				return true;
			}
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32: {
				auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
				if (predicate.EvaluateWide(inst.Arg(0), a)) {
					return Arg(inst, a != 0u ? 1u : 2u, result);
				}
				return false;
			}
			case ValueOpcode::IEqual32:
				if (binary()) {
					result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::INotEqual32:
				if (binary()) {
					result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::ULessThan32:
				if (binary()) {
					result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::UGreaterThan32:
				if (binary()) {
					result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
					return true;
				}
				return false;
			case ValueOpcode::SGreaterThanEqual32:
				if (binary()) {
					result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
					         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
					return true;
				}
				return false;
			case ValueOpcode::LogicalAnd:
				if (binary()) {
					result = (a != 0u) && (b != 0u);
					return true;
				}
				return false;
			case ValueOpcode::LogicalOr:
				if (binary()) {
					result = (a != 0u) || (b != 0u);
					return true;
				}
				return false;
			case ValueOpcode::LogicalXor:
				if (binary()) {
					result = (a != 0u) != (b != 0u);
					return true;
				}
				return false;
			case ValueOpcode::LogicalNot:
				if (Arg(inst, 0, a)) {
					result = a == 0u;
					return true;
				}
				return false;
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU8:
			case ValueOpcode::UndefU16:
			case ValueOpcode::UndefU32:
			case ValueOpcode::UndefU64: return false;
			default: break;
		}
		return false;
	}

	const ResourcePlan&                       m_program;
	const SrtRuntime&                         m_runtime;
	std::span<const uint8_t>                  m_clean_flat_slots;
	Evaluator*                                m_clean_evaluator = nullptr;
	Value                                     m_active_mask;
	uint32_t                                  m_epoch = 0;
	// Values without an evaluation slot (rare). A vector: MSVC's unordered_map allocates when it is
	// constructed, and several evaluators are built per draw.
	std::vector<std::pair<const Inst*, uint64_t>> m_cache;
	std::vector<const Inst*>                  m_visiting;
	bool                                      m_reserved = false;
};

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

bool EvaluateRuntimeSourcesImpl(const ResourcePlan& program, std::span<const uint32_t> sources,
                                const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                                std::vector<uint32_t>& flat, bool evaluate_flat,
                                std::span<const uint8_t> clean_flat_slots,
                                std::vector<uint8_t>&    active_sources) {
	static const bool split_timing = std::getenv("KYTY_SAMPLE_GPU") != nullptr;
	const auto t_eval_start = split_timing ? std::chrono::steady_clock::now()
	                                       : std::chrono::steady_clock::time_point {};
	t_srt_failure.clear();
	if (!program.srt_plan_complete) {
		return false;
	}
	if (std::ranges::any_of(clean_flat_slots, [](uint8_t clean) { return clean != 0u; }) &&
	    runtime.read_specialization_memory == nullptr) {
		return false;
	}
	const auto           clean_runtime = CleanRuntime(runtime);
	Evaluator            clean_evaluator(program, clean_runtime);
	Evaluator            evaluator(program, runtime, clean_flat_slots, &clean_evaluator);
	std::vector<uint8_t> active;
	if (evaluate_flat) {
		active.assign(program.descriptor_sources.size(), 1u);
	}
	if (evaluate_flat && !program.control_flow.empty()) {
		for (const auto& block: program.control_flow) {
			for (const auto source: block.sources) {
				active.at(source) = 0u;
			}
		}
		// Reused per thread: two allocations per draw otherwise.
		thread_local std::vector<uint8_t>  visited;
		thread_local std::vector<uint32_t> pending;
		visited.assign(program.control_flow.size(), 0u);
		pending.assign(1, 0u);
		while (!pending.empty()) {
			const auto index = pending.back();
			pending.pop_back();
			if (visited.at(index)) {
				continue;
			}
			visited[index]    = 1u;
			const auto& block = program.control_flow[index];
			for (const auto source: block.sources) {
				active[source] = 1u;
			}
			uint32_t condition = 0;
			// A missing clean reader must never fall through to the evaluator's raw-memory path.
			if (!block.condition.IsEmpty() && runtime.read_specialization_memory != nullptr &&
			    clean_evaluator.Evaluate(block.condition, condition)) {
				pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
			} else {
				pending.insert(pending.end(), block.successors.begin(), block.successors.end());
			}
		}
	}
	std::vector<DescriptorValue> evaluated;
	evaluated.reserve(sources.size());
	for (const auto source_index: sources) {
		const auto* source = Source(program, source_index);
		if (source == nullptr) {
			return false;
		}
		DescriptorValue value;
		value.dword_count = source->dword_count;
		if (!evaluate_flat || active[source_index]) {
			evaluator.Prepare();
			const bool compiled = source_index < program.compiled_source_roots.size();
			for (uint32_t index = 0; index < source->dword_count; index++) {
				const bool ok = compiled && index < 8u
				                    ? evaluator.EvaluateRoot(program.compiled_source_roots[source_index][index],
				                                             value.dwords[index])
				                    : evaluator.Evaluate(source->dwords[index], value.dwords[index]);
				if (!ok) {
					return false;
				}
			}
		}
		evaluated.push_back(value);
	}
	// KYTY_SAMPLE_GPU=1: how evaluation time splits between descriptor sources and the flat SRT.
	static const bool split_stats = std::getenv("KYTY_SAMPLE_GPU") != nullptr;
	const auto        flat_start  = split_stats ? std::chrono::steady_clock::now()
	                                            : std::chrono::steady_clock::time_point {};
	std::vector<uint32_t> flattened;
	const auto record_split = [&]() {
		if (!split_stats) {
			return;
		}
		// Per thread (the GPU thread does nearly all of it): a lock here slowed the profiled runs.
		thread_local uint64_t calls = 0, sources_n = 0, flat_n = 0, sources_ns = 0, flat_ns = 0;
		thread_local auto     last  = std::chrono::steady_clock::now();
		const auto            now   = std::chrono::steady_clock::now();
		calls++;
		sources_n += sources.size();
		flat_n += evaluate_flat ? program.srt_reads.size() : 0u;
		sources_ns += static_cast<uint64_t>((flat_start - t_eval_start).count());
		flat_ns += static_cast<uint64_t>((now - flat_start).count());
		if (now - last > std::chrono::seconds(5)) {
			std::printf("SRTSPLIT calls=%llu sources=%llu flat_reads=%llu sources_ms=%llu flat_ms=%llu\n",
			            static_cast<unsigned long long>(calls), static_cast<unsigned long long>(sources_n),
			            static_cast<unsigned long long>(flat_n),
			            static_cast<unsigned long long>(sources_ns / 1000000u),
			            static_cast<unsigned long long>(flat_ns / 1000000u));
			calls = sources_n = flat_n = sources_ns = flat_ns = 0;
			last = now;
		}
	};
	if (evaluate_flat) {
		flattened.resize(program.srt_reads.size());
		evaluator.Prepare();
		clean_evaluator.Prepare();
		const bool compiled_roots = program.compiled_srt_roots.size() == program.srt_reads.size();
		for (size_t read_index = 0; read_index < program.srt_reads.size(); read_index++) {
			const auto& read = program.srt_reads[read_index];
			if (read.gpu) {
				// The shader loads this slot itself.
				continue;
			}
			const bool clean    = read.flat_offset < clean_flat_slots.size() &&
			                      clean_flat_slots[read.flat_offset] != 0u;
			auto&      selected = clean ? clean_evaluator : evaluator;
			if (read.flat_offset >= flattened.size() ||
			    !(compiled_roots ? selected.EvaluateRoot(program.compiled_srt_roots[read_index],
			                                             flattened[read.flat_offset])
			                     : selected.Evaluate(read.value, flattened[read.flat_offset]))) {
				return false;
			}
		}
	}
	record_split();
	results = std::move(evaluated);
	active_sources = std::move(active);
	if (evaluate_flat) {
		flat = std::move(flattened);
	}
	return true;
}

} // namespace

std::string_view SrtLastFailure() {
	return t_srt_failure;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

void BuildSrtPlan(Program& program) {
	if (program.resource_tracking_complete) {
		EXIT("shader SRT planning failed: cannot rebuild SRT after resource tracking");
	}
	program.srt_plan_complete = false;
	PlanBuilder(program).Run();
	program.srt_plan_complete = true;
}

bool EvaluateUniformValues(const ResourcePlan& program, std::span<const Value> values,
                            const SrtRuntime& runtime, std::span<uint32_t> results) {
	if (values.size() != results.size()) {
		return false;
	}
	const auto clean = CleanRuntime(runtime);
	Evaluator  evaluator(program, clean);
	for (size_t i = 0; i < values.size(); ++i) {
		if (!evaluator.Evaluate(values[i], results[i])) {
			return false;
		}
	}
	return true;
}

bool EvaluateDescriptorSource(const ResourcePlan& program, uint32_t source,
                              const SrtRuntime& runtime, DescriptorValue& result) {
	std::vector<DescriptorValue> results;
	if (!EvaluateDescriptorSources(program, std::span {&source, 1}, runtime, results)) {
		return false;
	}
	result = results.front();
	return true;
}

bool EvaluateDescriptorSources(const ResourcePlan& program, std::span<const uint32_t> sources,
                               const SrtRuntime& runtime, std::vector<DescriptorValue>& results) {
	std::vector<uint32_t> ignored;
	std::vector<uint8_t>  active;
	return EvaluateRuntimeSourcesImpl(program, sources, runtime, results, ignored, false, {},
	                                  active);
}

bool EvaluateRuntimeSources(const ResourcePlan& program, std::span<const uint32_t> sources,
                            const SrtRuntime& runtime, std::vector<DescriptorValue>& results,
                            std::vector<uint32_t>& flat, std::span<const uint8_t> clean_flat_slots,
                            std::vector<uint8_t>& active_sources) {
	return EvaluateRuntimeSourcesImpl(program, sources, runtime, results, flat, true,
	                                  clean_flat_slots, active_sources);
}

bool WalkSrt(const ResourcePlan& program, const SrtRuntime& runtime, std::vector<uint32_t>& flat) {
	std::vector<DescriptorValue> ignored;
	std::vector<uint8_t>         active;
	return EvaluateRuntimeSources(program, {}, runtime, ignored, flat, {}, active);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
