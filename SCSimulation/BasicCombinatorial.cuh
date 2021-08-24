#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <memory.h>

#include "CombinatorialComponent.cuh"
#include "StochasticCircuit.cuh"

class Inverter : public CombinatorialComponent
{
public:
	Inverter(uint32_t input, uint32_t output) : CombinatorialComponent(1, 1, typehash(Inverter), sizeof(Inverter)) {
		inputs[0] = input;
		outputs[0] = output;
	}

	link_device_sim_function(Inverter)

	virtual void simulate_step_host() override {
		auto out_offset = output_offsets_host[0];
		auto in_offset = input_offsets_host[0];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~circuit->net_values_host[in_offset + i];
		}
	}

	static __device__ void _simulate_step_dev(CircuitComponent* comp) {
		auto g = (Inverter*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset = g->input_offsets_dev[0];
		auto i = g->current_progress_word + blockIdx.y * blockDim.x + threadIdx.x;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = ~g->net_values_dev[in_offset + i];
		}
	}
};

class AndGate : public CombinatorialComponent
{
public:
	AndGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(AndGate), sizeof(AndGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(AndGate)

	virtual void simulate_step_host() override {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = circuit->net_values_host[in_offset_1 + i] & circuit->net_values_host[in_offset_2 + i];
		}
	}

	static __device__ void _simulate_step_dev(CircuitComponent* comp) {
		auto g = (AndGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.x + threadIdx.x;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = g->net_values_dev[in_offset_1 + i] & g->net_values_dev[in_offset_2 + i];
		}
	}
	
};

class NandGate : public CombinatorialComponent
{
public:
	NandGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(NandGate), sizeof(NandGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(NandGate)

	virtual void simulate_step_host() override {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~(circuit->net_values_host[in_offset_1 + i] & circuit->net_values_host[in_offset_2 + i]);
		}
	}

	static __device__ void _simulate_step_dev(CircuitComponent* comp) {
		auto g = (NandGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.x + threadIdx.x;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = ~(g->net_values_dev[in_offset_1 + i] & g->net_values_dev[in_offset_2 + i]);
		}
	}
};

class OrGate : public CombinatorialComponent
{
public:
	OrGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(OrGate), sizeof(OrGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(OrGate)

	virtual void simulate_step_host() override {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = circuit->net_values_host[in_offset_1 + i] | circuit->net_values_host[in_offset_2 + i];
		}
	}

	static __device__ void _simulate_step_dev(CircuitComponent* comp) {
		auto g = (OrGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.x + threadIdx.x;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = g->net_values_dev[in_offset_1 + i] | g->net_values_dev[in_offset_2 + i];
		}
	}
};

class NorGate : public CombinatorialComponent
{
public:
	NorGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(NorGate), sizeof(NorGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(NorGate)

	virtual void simulate_step_host() override {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~(circuit->net_values_host[in_offset_1 + i] | circuit->net_values_host[in_offset_2 + i]);
		}
	}

	static __device__ void _simulate_step_dev(CircuitComponent* comp) {
		auto g = (NorGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.x + threadIdx.x;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = ~(g->net_values_dev[in_offset_1 + i] | g->net_values_dev[in_offset_2 + i]);
		}
	}
};

class XorGate : public CombinatorialComponent
{
public:
	XorGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(XorGate), sizeof(XorGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(XorGate)

	virtual void simulate_step_host() override {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = circuit->net_values_host[in_offset_1 + i] ^ circuit->net_values_host[in_offset_2 + i];
		}
	}

	static __device__ void _simulate_step_dev(CircuitComponent* comp) {
		auto g = (XorGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.x + threadIdx.x;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = g->net_values_dev[in_offset_1 + i] ^ g->net_values_dev[in_offset_2 + i];
		}
	}
};

class XnorGate : public CombinatorialComponent
{
public:
	XnorGate(uint32_t input1, uint32_t input2, uint32_t output) : CombinatorialComponent(2, 1, typehash(XnorGate), sizeof(XnorGate)) {
		inputs[0] = input1;
		inputs[1] = input2;
		outputs[0] = output;
	}

	link_device_sim_function(XnorGate)

	virtual void simulate_step_host() override {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			circuit->net_values_host[out_offset + i] = ~(circuit->net_values_host[in_offset_1 + i] ^ circuit->net_values_host[in_offset_2 + i]);
		}
	}

	static __device__ void _simulate_step_dev(CircuitComponent* comp) {
		auto g = (XnorGate*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto i = g->current_progress_word + blockIdx.y * blockDim.x + threadIdx.x;
		if (i < g->next_step_progress_word) {
			g->net_values_dev[out_offset + i] = ~(g->net_values_dev[in_offset_1 + i] ^ g->net_values_dev[in_offset_2 + i]);
		}
	}
};

class Multiplexer2 : public CombinatorialComponent
{
public:
	Multiplexer2(uint32_t input1, uint32_t input2, uint32_t select, uint32_t output) : CombinatorialComponent(3, 1, typehash(Multiplexer2), sizeof(Multiplexer2)) {
		inputs[0] = input1;
		inputs[1] = input2;
		inputs[2] = select;
		outputs[0] = output;
	}

	link_device_sim_function(Multiplexer2)

	virtual void simulate_step_host() override {
		auto out_offset = output_offsets_host[0];
		auto in_offset_1 = input_offsets_host[0];
		auto in_offset_2 = input_offsets_host[1];
		auto sel_offset = input_offsets_host[2];

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			auto sel = circuit->net_values_host[sel_offset + i];
			circuit->net_values_host[out_offset + i] = (circuit->net_values_host[in_offset_1 + i] & ~sel) | (circuit->net_values_host[in_offset_2 + i] & sel);
		}
	}

	static __device__ void _simulate_step_dev(CircuitComponent* comp) {
		auto g = (Multiplexer2*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset_1 = g->input_offsets_dev[0];
		auto in_offset_2 = g->input_offsets_dev[1];
		auto sel_offset = g->input_offsets_dev[2];
		auto i = g->current_progress_word + blockIdx.y * blockDim.x + threadIdx.x;
		if (i < g->next_step_progress_word) {
			auto sel = g->net_values_dev[sel_offset + i];
			g->net_values_dev[out_offset + i] = (g->net_values_dev[in_offset_1 + i] & ~sel) | (g->net_values_dev[in_offset_2 + i] & sel);
		}
	}
};

class MultiplexerN : public CombinatorialComponent
{
public:
	MultiplexerN(uint32_t num_inputs, uint32_t* inputs, uint32_t* selects, uint32_t output) : num_mux_inputs(num_inputs), num_selects((uint32_t)ceil(log2((double)num_inputs))),
		CombinatorialComponent(num_inputs + num_selects, 1, typehash(MultiplexerN), sizeof(MultiplexerN)) {

		memcpy(this->inputs, inputs, num_inputs * sizeof(uint32_t));
		memcpy((this->inputs + num_inputs), selects, num_selects * sizeof(uint32_t));

		outputs[0] = output;
	}

	MultiplexerN(std::initializer_list<uint32_t> inputs, std::initializer_list<uint32_t> selects, uint32_t output) : num_mux_inputs(inputs.size()), num_selects((uint32_t)ceil(log2((double)inputs.size()))),
		CombinatorialComponent(inputs.size() + (uint32_t)ceil(log2((double)inputs.size())), 1, typehash(MultiplexerN), sizeof(MultiplexerN)) {

		if (selects.size() < num_selects) throw;
		
		auto in = inputs.begin();
		for (auto i = 0; i < inputs.size(); i++) {
			this->inputs[i] = *in++;
		}

		auto sel = selects.begin();
		for (auto i = 0; i < num_selects; i++) {
			this->inputs[inputs.size() + i] = *sel++;
		}

		outputs[0] = output;
	}

	link_device_sim_function(MultiplexerN)

	virtual void simulate_step_host() override {
		auto out_offset = output_offsets_host[0];
		
		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			uint32_t word = 0;
			for (uint32_t j = 0; j < num_mux_inputs; j++) {
				uint32_t input = circuit->net_values_host[input_offsets_host[j] + i];
				for (uint32_t k = 0; k < num_selects; k++) {
					auto sel = circuit->net_values_host[input_offsets_host[num_mux_inputs + k] + i];
					input &= ((j & (1 << k)) > 0) ? sel : ~sel;
				}
				word |= input;
			}
			circuit->net_values_host[out_offset + i] = word;
		}
	}

	static __device__ void _simulate_step_dev(CircuitComponent* comp) {
		auto g = (MultiplexerN*)comp;
		auto out_offset = g->output_offsets_dev[0];

		auto i = g->current_progress_word + blockIdx.y * blockDim.x + threadIdx.x;
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

private:
	friend CircuitComponent;

	const uint32_t num_mux_inputs;
	const uint32_t num_selects;

};