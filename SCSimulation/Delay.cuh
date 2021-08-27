#pragma once

#include "cuda_base.cuh"
#include <stdint.h>
#include <memory.h>

#include "CombinatorialComponent.cuh"
#include "StochasticCircuit.cuh"

class Delay : public CombinatorialComponent
{
public:
	Delay(uint32_t input, uint32_t output) : CombinatorialComponent(1, 1, typehash(Delay), sizeof(Delay), alignof(Delay)) {
		inputs[0] = input;
		outputs[0] = output;
	}

	link_device_sim_function(Delay)

		virtual void simulate_step_host() override {
		auto out_offset = output_offsets_host[0];
		auto in_offset = input_offsets_host[0];

		auto prev_word = current_progress_word == 0 ? 0 : circuit->net_values_host[in_offset + current_progress_word - 1];
		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			auto curr_word = circuit->net_values_host[in_offset + i];
			circuit->net_values_host[out_offset + i] = (curr_word >> 1) | (prev_word << 31);
			prev_word = curr_word;
		}
	}

	static __device__ void _simulate_step_dev(CircuitComponent* comp) {
		auto g = (Delay*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset = g->input_offsets_dev[0];
		auto i = g->current_progress_word + blockIdx.y * blockDim.x + threadIdx.x;
		if (i < g->next_step_progress_word) {
			auto curr_word = g->net_values_dev[in_offset + i];
			auto prev_word = i == 0 ? 0 : g->net_values_dev[in_offset + i - 1];
			g->net_values_dev[out_offset + i] = (curr_word >> 1) | (prev_word << 31);
		}
	}
};