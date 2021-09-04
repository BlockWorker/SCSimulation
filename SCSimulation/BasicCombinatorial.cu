#include "cuda_base.cuh"
#include "circuit_component_defines.cuh"

#include <memory.h>
#include <typeinfo>

#include "BasicCombinatorial.cuh"
#include "StochasticCircuit.cuh"

namespace scsim {

	Inverter::Inverter(uint32_t input, uint32_t output) : CombinatorialComponent(1, 1, typehash(Inverter), sizeof(Inverter), alignof(Inverter)) {
		inputs[0] = input;
		outputs[0] = output;
	}

	link_device_sim_function(Inverter)

	void Inverter::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset = input_offsets_host[0];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~circuit->net_values_host[in_offset + i];
		}
	}

	__device__ void Inverter::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (Inverter*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset = g->input_offsets_dev[0];
		auto i = g->current_progress_word + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = ~g->net_values_dev[in_offset + i];
		}
	}


	AndGate::AndGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(AndGate), sizeof(AndGate), alignof(AndGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(AndGate)

	void AndGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = circuit->net_values_host[in_offset_1 + i] & circuit->net_values_host[in_offset_2 + i];
		}
	}

	__device__ void AndGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (AndGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = g->net_values_dev[in_offset_1 + i] & g->net_values_dev[in_offset_2 + i];
		}
	}


	NandGate::NandGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(NandGate), sizeof(NandGate), alignof(NandGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(NandGate)

	void NandGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~(circuit->net_values_host[in_offset_1 + i] & circuit->net_values_host[in_offset_2 + i]);
		}
	}

	__device__ void NandGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (NandGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = ~(g->net_values_dev[in_offset_1 + i] & g->net_values_dev[in_offset_2 + i]);
		}
	}


	OrGate::OrGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(OrGate), sizeof(OrGate), alignof(OrGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(OrGate)

	void OrGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = circuit->net_values_host[in_offset_1 + i] | circuit->net_values_host[in_offset_2 + i];
		}
	}

	__device__ void OrGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (OrGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = g->net_values_dev[in_offset_1 + i] | g->net_values_dev[in_offset_2 + i];
		}
	}


	NorGate::NorGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(NorGate), sizeof(NorGate), alignof(NorGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(NorGate)

	void NorGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~(circuit->net_values_host[in_offset_1 + i] | circuit->net_values_host[in_offset_2 + i]);
		}
	}

	__device__ void NorGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (NorGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = ~(g->net_values_dev[in_offset_1 + i] | g->net_values_dev[in_offset_2 + i]);
		}
	}


	XorGate::XorGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(XorGate), sizeof(XorGate), alignof(XorGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(XorGate)

	void XorGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = circuit->net_values_host[in_offset_1 + i] ^ circuit->net_values_host[in_offset_2 + i];
		}
	}

	__device__ void XorGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (XorGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = g->net_values_dev[in_offset_1 + i] ^ g->net_values_dev[in_offset_2 + i];
		}
	}


	XnorGate::XnorGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(XnorGate), sizeof(XnorGate), alignof(XnorGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(XnorGate)

	void XnorGate::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~(circuit->net_values_host[in_offset_1 + i] ^ circuit->net_values_host[in_offset_2 + i]);
		}
	}

	__device__ void XnorGate::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (XnorGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = ~(g->net_values_dev[in_offset_1 + i] ^ g->net_values_dev[in_offset_2 + i]);
		}
	}


	Multiplexer2::Multiplexer2(uint32_t input1, uint32_t input2, uint32_t select, uint32_t output) : CombinatorialComponent(3, 1, typehash(Multiplexer2), sizeof(Multiplexer2), alignof(Multiplexer2)) {
		inputs[0] = input1;
		inputs[1] = input2;
		inputs[2] = select;
		outputs[0] = output;
	}

	link_device_sim_function(Multiplexer2)

	void Multiplexer2::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];
		auto sel_offset = input_offsets_host[2];

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
		auto i = g->current_progress_word + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_step_progress_word) {
			auto sel = g->net_values_dev[sel_offset + i];
			g->net_values_dev[out_offset + i] = (g->net_values_dev[in_offset_1 + i] & ~sel) | (g->net_values_dev[in_offset_2 + i] & sel);
		}
	}


	MultiplexerN::MultiplexerN(uint32_t num_inputs, uint32_t* inputs, uint32_t* selects, uint32_t output) : num_mux_inputs(num_inputs), num_selects((uint32_t)ceil(log2((double)num_inputs))),
		CombinatorialComponent(num_inputs + num_selects, 1, typehash(MultiplexerN), sizeof(MultiplexerN), alignof(MultiplexerN)) {

		memcpy(this->inputs, inputs, num_inputs * sizeof(uint32_t));
		memcpy((this->inputs + num_inputs), selects, num_selects * sizeof(uint32_t));

		outputs[0] = output;
	}

	MultiplexerN::MultiplexerN(uint32_t num_inputs, uint32_t first_input, uint32_t first_select, uint32_t output) : num_mux_inputs(num_inputs), num_selects((uint32_t)ceil(log2((double)num_inputs))),
		CombinatorialComponent(num_inputs + num_selects, 1, typehash(MultiplexerN), sizeof(MultiplexerN), alignof(MultiplexerN)) {

		for (uint32_t i = 0; i < num_inputs; i++) {
			this->inputs[i] = first_input + i;
		}

		for (uint32_t i = 0; i < num_selects; i++) {
			this->inputs[num_inputs + i] = first_select + i;
		}

		outputs[0] = output;
	}

	MultiplexerN::MultiplexerN(std::initializer_list<uint32_t> inputs, std::initializer_list<uint32_t> selects, uint32_t output) : num_mux_inputs(inputs.size()), num_selects((uint32_t)ceil(log2((double)inputs.size()))),
		CombinatorialComponent(inputs.size() + (uint32_t)ceil(log2((double)inputs.size())), 1, typehash(MultiplexerN), sizeof(MultiplexerN), alignof(MultiplexerN)) {

		if (selects.size() < num_selects) throw;

		auto in = inputs.begin();
		for (uint32_t i = 0; i < inputs.size(); i++) {
			this->inputs[i] = *in++;
		}

		auto sel = selects.begin();
		for (uint32_t i = 0; i < num_selects; i++) {
			this->inputs[inputs.size() + i] = *sel++;
		}

		outputs[0] = output;
	}

	link_device_sim_function(MultiplexerN)

	void MultiplexerN::simulate_step_host() {
		auto out_offset = output_offsets_host[0];

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

		auto i = g->current_progress_word + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_step_progress_word) {
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


	Delay::Delay(uint32_t input, uint32_t output) : CombinatorialComponent(1, 1, typehash(Delay), sizeof(Delay), alignof(Delay)) {
		inputs[0] = input;
		outputs[0] = output;
	}

	link_device_sim_function(Delay)

	void Delay::calculate_simulation_progress() {
		current_progress = circuit->sim_length;
		for (uint32_t i = 0; i < num_outputs; i++) {
			auto out_progress = circuit->net_progress_host[outputs[i]];
			if (out_progress < current_progress) { //current progress equals the minimum progress of output nets
				current_progress = out_progress;
			}
		}

		next_step_progress = circuit->sim_length;
		for (uint32_t i = 0; i < num_inputs; i++) {
			auto in_progress = circuit->net_progress_host[inputs[i]];
			if (in_progress < next_step_progress) { //next step progress equals the minimum progress of input nets + 1 (delay can progress one step further)
				next_step_progress = in_progress + 1;
			}
		}

		if (next_step_progress < current_progress) {
			next_step_progress = current_progress;
		}

		current_progress_word = current_progress / 32;
		next_step_progress_word = (next_step_progress + 31) / 32;
	}

	void Delay::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset = input_offsets_host[0];

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
		auto i = g->current_progress_word + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_step_progress_word) {
			auto curr_word = g->net_values_dev[in_offset + i];
			auto prev_word = i == 0 ? 0 : g->net_values_dev[in_offset + i - 1];
			g->net_values_dev[out_offset + i] = (curr_word >> 1) | (prev_word << 31);
		}
	}

}
