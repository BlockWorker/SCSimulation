#include "cuda_base.cuh"

#include <stdint.h>
#include <typeinfo>
#include <random>

#include "StochasticCircuit.cuh"
#include "StochasticCircuitFactory.cuh"
#include "LFSR.cuh"
#include "libpopcnt.h"

#undef COMP_IMPEXP_SPEC
#define COMP_IMPEXP_SPEC SCSIMAPI

namespace scsim {

	LFSR::LFSR(uint32_t output, StochasticCircuitFactory* factory) : SequentialComponent(0, 1, typehash(LFSR), sizeof(LFSR), alignof(LFSR), factory) {
		outputs_host[0] = output;
		link_device_sim_function(LFSR);
	}

	def_device_statics(LFSR)

	void LFSR::reset_state() {
		state[0] = std::random_device()() | 1u;
	}

	void LFSR::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto current_progress = current_sim_progress();
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress = next_sim_progress();
		auto next_step_progress_word = next_sim_progress_word();

		uint32_t out_word = circuit->net_values_host[out_offset + current_progress_word] >> (32 - (current_progress % 32)); //present output, shifted for seamless continuation

		for (uint32_t i = current_progress_word; i < next_step_progress_word; i++) {
			for (uint32_t j = 0; j < 32; j++) {
				auto bit = 32 * i + j;
				if (bit >= next_step_progress) { //last word ends before being full: shift for correct bit position
					out_word <<= (32 - j);
					break;
				}

				uint32_t tap_state = state[0] & 0x80200003u; //taps on bits 31, 21, 1, 0 for period 2^32-1
				auto feedback = popcnt(&tap_state, sizeof(uint32_t)) & 1u;

				auto outbit = takebit(state[0]); //remove bit, shift, add feedback
				state[0] <<= 1;
				putbit(state[0], feedback);

				out_word <<= 1;
				putbit(out_word, outbit); //push last removed bit
			}
			circuit->net_values_host[out_offset + i] = out_word;
			out_word = 0;
		}
	}

	//device simulation equivalent to host simulation
	__device__ void LFSR::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (LFSR*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto current_progress = g->current_sim_progress();
		auto current_progress_word = g->current_sim_progress_word();
		auto next_step_progress = g->next_sim_progress();
		auto next_step_progress_word = g->next_sim_progress_word();

		uint32_t out_word = g->net_values_dev[out_offset + current_progress_word] >> (32 - (current_progress % 32));

		for (uint32_t i = current_progress_word; i < next_step_progress_word; i++) {
			for (uint32_t j = 0; j < 32; j++) {
				auto bit = 32 * i + j;
				if (bit >= next_step_progress) {
					out_word <<= (32 - j);
					break;
				}

				auto feedback = __popc(g->state[0] & 0x80200003u) & 1u; //use popcount intrinsic for device

				auto outbit = takebit(g->state[0]);
				g->state[0] <<= 1;
				putbit(g->state[0], feedback);

				out_word <<= 1;
				putbit(out_word, outbit);
			}
			g->net_values_dev[out_offset + i] = out_word;
			out_word = 0;
		}
	}

}
