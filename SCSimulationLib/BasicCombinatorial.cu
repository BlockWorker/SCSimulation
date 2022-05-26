#include "cuda_base.cuh"

#include <memory.h>

#include "StochasticCircuit.cuh"
#include "BasicCombinatorial.cuh"

#undef COMP_IMPEXP_SPEC
#define COMP_IMPEXP_SPEC SCSIMAPI

namespace scsim {

	Inverter::Inverter(uint32_t input, uint32_t output, StochasticCircuitFactory* factory) : CombinatorialComponent(1, 1, 0, typehash(Inverter), sizeof(Inverter), alignof(Inverter), factory) {
		inputs_host[0] = input;
		outputs_host[0] = output;
		link_device_sim_function(Inverter);
	}

	def_device_statics(Inverter)

	void Inverter::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset = input_offsets_host[0];
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~circuit->net_values_host[in_offset + i];
		}
	}

	__device__ void Inverter::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (Inverter*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset = g->input_offsets_dev[0];
		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			g->net_values_dev[out_offset + i] = ~g->net_values_dev[in_offset + i];
		}
	}


	AndGate::AndGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory) : CombinatorialComponent(2, 1, 0, typehash(AndGate), sizeof(AndGate), alignof(AndGate), factory) {
		inputs_host[0] = input1;
		inputs_host[1] = input2;
		outputs_host[0] = output;
		link_device_sim_function(AndGate);
	}

	def_device_statics(AndGate)

	void AndGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = circuit->net_values_host[in_offset_1 + i] & circuit->net_values_host[in_offset_2 + i];
		}
	}

	__device__ void AndGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (AndGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			g->net_values_dev[out_offset + i] = g->net_values_dev[in_offset_1 + i] & g->net_values_dev[in_offset_2 + i];
		}
	}


	NandGate::NandGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory) : CombinatorialComponent(2, 1, 0, typehash(NandGate), sizeof(NandGate), alignof(NandGate), factory) {
		inputs_host[0] = input1;
		inputs_host[1] = input2;
		outputs_host[0] = output;
		link_device_sim_function(NandGate);
	}

	def_device_statics(NandGate)

	void NandGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~(circuit->net_values_host[in_offset_1 + i] & circuit->net_values_host[in_offset_2 + i]);
		}
	}

	__device__ void NandGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (NandGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			g->net_values_dev[out_offset + i] = ~(g->net_values_dev[in_offset_1 + i] & g->net_values_dev[in_offset_2 + i]);
		}
	}


	OrGate::OrGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory) : CombinatorialComponent(2, 1, 0, typehash(OrGate), sizeof(OrGate), alignof(OrGate), factory) {
		inputs_host[0] = input1;
		inputs_host[1] = input2;
		outputs_host[0] = output;
		link_device_sim_function(OrGate);
	}

	def_device_statics(OrGate)

	void OrGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = circuit->net_values_host[in_offset_1 + i] | circuit->net_values_host[in_offset_2 + i];
		}
	}

	__device__ void OrGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (OrGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			g->net_values_dev[out_offset + i] = g->net_values_dev[in_offset_1 + i] | g->net_values_dev[in_offset_2 + i];
		}
	}


	NorGate::NorGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory) : CombinatorialComponent(2, 1, 0, typehash(NorGate), sizeof(NorGate), alignof(NorGate), factory) {
		inputs_host[0] = input1;
		inputs_host[1] = input2;
		outputs_host[0] = output;
		link_device_sim_function(NorGate);
	}

	def_device_statics(NorGate)

	void NorGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~(circuit->net_values_host[in_offset_1 + i] | circuit->net_values_host[in_offset_2 + i]);
		}
	}

	__device__ void NorGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (NorGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			g->net_values_dev[out_offset + i] = ~(g->net_values_dev[in_offset_1 + i] | g->net_values_dev[in_offset_2 + i]);
		}
	}


	XorGate::XorGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory) : CombinatorialComponent(2, 1, 0, typehash(XorGate), sizeof(XorGate), alignof(XorGate), factory) {
		inputs_host[0] = input1;
		inputs_host[1] = input2;
		outputs_host[0] = output;
		link_device_sim_function(XorGate);
	}

	def_device_statics(XorGate)

	void XorGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = circuit->net_values_host[in_offset_1 + i] ^ circuit->net_values_host[in_offset_2 + i];
		}
	}

	__device__ void XorGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (XorGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			g->net_values_dev[out_offset + i] = g->net_values_dev[in_offset_1 + i] ^ g->net_values_dev[in_offset_2 + i];
		}
	}


	XnorGate::XnorGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory) : CombinatorialComponent(2, 1, 0, typehash(XnorGate), sizeof(XnorGate), alignof(XnorGate), factory) {
		inputs_host[0] = input1;
		inputs_host[1] = input2;
		outputs_host[0] = output;
		link_device_sim_function(XnorGate);
	}

	def_device_statics(XnorGate)

	void XnorGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~(circuit->net_values_host[in_offset_1 + i] ^ circuit->net_values_host[in_offset_2 + i]);
		}
	}

	__device__ void XnorGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (XnorGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			g->net_values_dev[out_offset + i] = ~(g->net_values_dev[in_offset_1 + i] ^ g->net_values_dev[in_offset_2 + i]);
		}
	}


	Multiplexer2::Multiplexer2(uint32_t input1, uint32_t input2, uint32_t select, uint32_t output, StochasticCircuitFactory* factory) : CombinatorialComponent(3, 1, 0, typehash(Multiplexer2), sizeof(Multiplexer2), alignof(Multiplexer2), factory) {
		inputs_host[0] = input1;
		inputs_host[1] = input2;
		inputs_host[2] = select;
		outputs_host[0] = output;
		link_device_sim_function(Multiplexer2);
	}

	def_device_statics(Multiplexer2)

	void Multiplexer2::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];
		auto sel_offset = input_offsets_host[2];
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			auto sel = circuit->net_values_host[sel_offset + i];
			circuit->net_values_host[out_offset + i] = (circuit->net_values_host[in_offset_1 + i] & ~sel) | (circuit->net_values_host[in_offset_2 + i] & sel);
		}
	}

	__device__ void Multiplexer2::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (Multiplexer2*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto sel_offset = g->input_offsets_dev[2];
		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			auto sel = g->net_values_dev[sel_offset + i];
			g->net_values_dev[out_offset + i] = (g->net_values_dev[in_offset_1 + i] & ~sel) | (g->net_values_dev[in_offset_2 + i] & sel);
		}
	}


	MultiplexerN::MultiplexerN(uint32_t _num_inputs, uint32_t* inputs, uint32_t* selects, uint32_t output, StochasticCircuitFactory* factory) : num_mux_inputs(_num_inputs), num_selects((uint32_t)ceil(log2((double)_num_inputs))),
		CombinatorialComponent(_num_inputs + (uint32_t)ceil(log2((double)_num_inputs)), 1, 0, typehash(MultiplexerN), sizeof(MultiplexerN), alignof(MultiplexerN), factory) {

		memcpy(this->inputs_host, inputs, _num_inputs * sizeof(uint32_t));
		memcpy((this->inputs_host + _num_inputs), selects, num_selects * sizeof(uint32_t));

		outputs_host[0] = output;

		link_device_sim_function(MultiplexerN);
	}

	MultiplexerN::MultiplexerN(uint32_t _num_inputs, uint32_t first_input, uint32_t first_select, uint32_t output, StochasticCircuitFactory* factory) : num_mux_inputs(_num_inputs), num_selects((uint32_t)ceil(log2((double)_num_inputs))),
		CombinatorialComponent(_num_inputs + (uint32_t)ceil(log2((double)_num_inputs)), 1, 0, typehash(MultiplexerN), sizeof(MultiplexerN), alignof(MultiplexerN), factory) {

		for (uint32_t i = 0; i < _num_inputs; i++) {
			this->inputs_host[i] = first_input + i;
		}

		for (uint32_t i = 0; i < num_selects; i++) {
			this->inputs_host[_num_inputs + i] = first_select + i;
		}

		outputs_host[0] = output;

		link_device_sim_function(MultiplexerN);
	}

	MultiplexerN::MultiplexerN(std::initializer_list<uint32_t> inputs, std::initializer_list<uint32_t> selects, uint32_t output, StochasticCircuitFactory* factory) : num_mux_inputs(inputs.size()), num_selects((uint32_t)ceil(log2((double)inputs.size()))),
		CombinatorialComponent(inputs.size() + (uint32_t)ceil(log2((double)inputs.size())), 1, 0, typehash(MultiplexerN), sizeof(MultiplexerN), alignof(MultiplexerN), factory) {

		if (selects.size() < num_selects) throw std::runtime_error("MultiplexerN: Not enough select nets given.");

		auto in = inputs.begin();
		for (uint32_t i = 0; i < inputs.size(); i++) {
			this->inputs_host[i] = *in++;
		}

		auto sel = selects.begin();
		for (uint32_t i = 0; i < num_selects; i++) {
			this->inputs_host[inputs.size() + i] = *sel++;
		}

		outputs_host[0] = output;

		link_device_sim_function(MultiplexerN);
	}

	def_device_statics(MultiplexerN)

	void MultiplexerN::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			uint32_t word = 0;
			for (uint32_t j = 0; j < num_mux_inputs; j++) {
				uint32_t input = circuit->net_values_host[input_offsets_host[j] + i];
				for (uint32_t k = 0; k < num_selects; k++) {
					auto sel = circuit->net_values_host[input_offsets_host[num_mux_inputs + k] + i];
					input &= ((j & (1 << k)) > 0) ? sel : ~sel; //only keep input bits where all select signals are correct to select the corresponding input
				}
				word |= input; //OR all filtered inputs together to produce final output
			}
			circuit->net_values_host[out_offset + i] = word;
		}
	}

	//device simulation equivalent to host simulation (but only one word per thread)
	__device__ void MultiplexerN::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (MultiplexerN*)comp;
		auto out_offset = g->output_offsets_dev[0];

		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			uint32_t word = 0;
			for (uint32_t j = 0; j < g->num_mux_inputs; j++) {
				uint32_t input = g->net_values_dev[g->input_offsets_dev[j] + i];
				for (uint32_t k = 0; k < g->num_selects; k++) {
					auto sel = g->net_values_dev[g->input_offsets_dev[g->num_mux_inputs + k] + i];
					input &= ((j & (1 << k)) > 0) ? sel : ~sel;
				}
				word |= input;
			}
			g->net_values_dev[out_offset + i] = word;
		}
	}


	Delay::Delay(uint32_t input, uint32_t output, StochasticCircuitFactory* factory) : CombinatorialComponent(1, 1, 1, typehash(Delay), sizeof(Delay), alignof(Delay), factory) {
		inputs_host[0] = input;
		outputs_host[0] = output;
		link_device_sim_function(Delay);
	}

	def_device_statics(Delay)

	void Delay::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset = input_offsets_host[0];
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		auto prev_word = current_progress_word == 0 ? 0 : circuit->net_values_host[in_offset + current_progress_word - 1];
		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			auto curr_word = circuit->net_values_host[in_offset + i];
			circuit->net_values_host[out_offset + i] = (curr_word >> 1) | (prev_word << 31);
			prev_word = curr_word;
		}
	}

	__device__ void Delay::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (Delay*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset = g->input_offsets_dev[0];
		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			auto curr_word = g->net_values_dev[in_offset + i];
			auto prev_word = i == 0 ? 0 : g->net_values_dev[in_offset + i - 1];
			g->net_values_dev[out_offset + i] = (curr_word >> 1) | (prev_word << 31);
		}
	}

}
