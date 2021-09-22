#include "cuda_base.cuh"

#include <stdint.h>
#include <typeinfo>

#include "StochasticCircuit.cuh"
#include "StochasticCircuitFactory.cuh"
#include "Stanh.cuh"

#undef COMP_IMPEXP_SPEC
#define COMP_IMPEXP_SPEC SCSIMAPI

namespace scsim {

	Stanh::Stanh(uint32_t input, uint32_t output, uint32_t k, StochasticCircuitFactory* factory) : SequentialComponent(1, 1, typehash(Stanh), sizeof(Stanh), alignof(Stanh), factory), k(k) {
		inputs_host[0] = input;
		outputs_host[0] = output;
		link_device_sim_function(Stanh);
	}

	def_device_statics(Stanh)

	void Stanh::reset_state() {
		state[0] = k / 2;
	}

	void Stanh::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto in_offset = input_offsets_host[0];
		auto current_progress = current_sim_progress();
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress = next_sim_progress();
		auto next_step_progress_word = next_sim_progress_word();

		uint32_t out_word = circuit->net_values_host[out_offset + current_progress_word] >> (32 - (current_progress % 32)); //present output, shifted for seamless continuation

		for (uint32_t i = current_progress_word; i < next_step_progress_word; i++) {
			auto in_word = circuit->net_values_host[in_offset + i];
			for (uint32_t j = 0; j < 32; j++) {
				auto bit = 32 * i + j;
				if (bit < current_progress) { //ignore bits that have already been processed
					in_word <<= 1;
					continue;
				}
				if (bit >= next_step_progress) { //last word ends before being full: shift for correct bit position
					out_word <<= (32 - j);
					break;
				}

				out_word <<= 1;
				putbit(out_word, state[0] >= k / 2); //push zero if in lower half of states, one if in upper half of states

				auto in_bit = takebit(in_word);
				in_word <<= 1;
				if (in_bit && state[0] < k - 1) state[0]++; //increment state if one is received
				else if (!in_bit && state[0] > 0) state[0]--; //decrement state if zero is received
			}
			circuit->net_values_host[out_offset + i] = out_word;
			out_word = 0;
		}
	}

	//device simulation equivalent to host simulation
	__device__ void Stanh::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (Stanh*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto in_offset = g->input_offsets_dev[0];
		auto current_progress = g->current_sim_progress();
		auto current_progress_word = g->current_sim_progress_word();
		auto next_step_progress = g->next_sim_progress();
		auto next_step_progress_word = g->next_sim_progress_word();

		uint32_t out_word = g->net_values_dev[out_offset + current_progress_word] >> (32 - (current_progress % 32));

		for (uint32_t i = current_progress_word; i < next_step_progress_word; i++) {
			auto in_word = g->net_values_dev[in_offset + i];
			for (uint32_t j = 0; j < 32; j++) {
				auto bit = 32 * i + j;
				if (bit < current_progress) {
					in_word <<= 1;
					continue;
				}
				if (bit >= next_step_progress) {
					out_word <<= (32 - j);
					break;
				}

				out_word <<= 1;
				putbit(out_word, g->state[0] >= g->k / 2);

				auto in_bit = takebit(in_word);
				in_word <<= 1;
				if (in_bit && g->state[0] < g->k - 1) g->state[0]++;
				else if (~in_bit && g->state[0] > 0) g->state[0]--;
			}
			g->net_values_dev[out_offset + i] = out_word;
			out_word = 0;
		}
	}

}
