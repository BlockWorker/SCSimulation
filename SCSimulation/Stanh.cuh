﻿#pragma once

#include "cuda_base.cuh"
#include <stdint.h>
#include <memory.h>

#include "SequentialComponent.cuh"
#include "StochasticCircuit.cuh"

class Stanh : public SequentialComponent
{
public:
	Stanh(uint32_t input, uint32_t output, uint32_t k) : SequentialComponent(1, 1, 1, typehash(Stanh), sizeof(Stanh), alignof(Stanh)), k(k) {
		inputs[0] = input;
		outputs[0] = output;
	}

	link_device_sim_function(Stanh)

	virtual void reset_state() override {
		state_ptr()[0] = k / 2;
	}

	virtual void simulate_step_host() override {
		auto out_offset = output_offsets_host[0];
		auto in_offset = input_offsets_host[0];

		auto state = state_ptr();

		uint32_t out_word = circuit->net_values_host[out_offset + current_progress_word] >> (current_progress % 32);

		for (uint32_t i = current_progress_word; i < next_step_progress_word; i++) {
			auto in_word = circuit->net_values_host[in_offset + i];
			for (uint32_t j = 0; j < 32; j++) {
				auto bit = 32 * i + j;
				if (bit < current_progress) continue;
				if (bit >= next_step_progress) {
					out_word <<= (32 - j);
					break;
				}

				out_word <<= 1;
				putbit(out_word, state[0] >= k / 2);

				auto in_bit = takebit(in_word);
				in_word <<= 1;
				if (in_bit && state[0] < k - 1) state[0]++;
				else if (!in_bit && state[0] > 0) state[0]--;
			}
			circuit->net_values_host[out_offset + i] = out_word;
			out_word = 0;
		}
	}

	static __device__ void _simulate_step_dev(CircuitComponent* comp) {
		auto g = (Stanh*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset = g->input_offsets_dev[0];

		auto state = g->state_ptr();

		uint32_t out_word = g->net_values_dev[out_offset + g->current_progress_word] >> (g->current_progress % 32);

		for (uint32_t i = g->current_progress_word; i < g->next_step_progress_word; i++) {
			auto in_word = g->net_values_dev[in_offset + i];
			for (uint32_t j = 0; j < 32; j++) {
				auto bit = 32 * i + j;
				if (bit < g->current_progress) continue;
				if (bit >= g->next_step_progress) {
					out_word <<= (32 - j);
					break;
				}

				out_word <<= 1;
				putbit(out_word, state[0] >= g->k / 2);

				auto in_bit = takebit(in_word);
				in_word <<= 1;
				if (in_bit && state[0] < g->k - 1) state[0]++;
				else if (~in_bit && state[0] > 0) state[0]--;
			}
			g->net_values_dev[out_offset + i] = out_word;
			out_word = 0;
		}
	}

private:
	const uint32_t k;

};