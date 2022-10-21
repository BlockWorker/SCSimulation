#include "cuda_base.cuh"

#include <stdint.h>
#include <typeinfo>

#include "StochasticCircuit.cuh"
#include "StochasticCircuitFactory.cuh"
#include "Btanh.cuh"

#undef COMP_IMPEXP_SPEC
#define COMP_IMPEXP_SPEC SCSIMAPI

namespace scsim {

	Btanh::Btanh(uint32_t n, uint32_t r, uint32_t* inputs, uint32_t output, StochasticCircuitFactory* factory) : n(n), input_width((uint32_t)floor(log2((double)n)) + 1), s_max(r - 1), s_half(r / 2),
		SequentialComponent((uint32_t)floor(log2((double)n)) + 1, 1, 0, typehash(Btanh), sizeof(Btanh), alignof(Btanh), factory) {

		if (r % 2 != 0) throw std::runtime_error("Btanh: r must me a multiple of 2.");
		
		memcpy(this->inputs_host, inputs, input_width * sizeof(uint32_t));
		outputs_host[0] = output;

		link_device_sim_function(Btanh);
	}

	Btanh::Btanh(uint32_t n, uint32_t r, uint32_t first_input, uint32_t output, StochasticCircuitFactory* factory) : n(n), input_width((uint32_t)floor(log2((double)n)) + 1), s_max(r - 1),
		s_half(r / 2), SequentialComponent((uint32_t)floor(log2((double)n)) + 1, 1, 0, typehash(Btanh), sizeof(Btanh), alignof(Btanh), factory) {

		if (r % 2 != 0) throw std::runtime_error("Btanh: r must me a multiple of 2.");

		for (uint32_t i = 0; i < input_width; i++) {
			this->inputs_host[i] = first_input + i;
		}
		outputs_host[0] = output;

		link_device_sim_function(Btanh);
	}

	Btanh::Btanh(uint32_t n, uint32_t r, std::initializer_list<uint32_t> inputs, uint32_t output, StochasticCircuitFactory* factory) : n(n), input_width((uint32_t)floor(log2((double)n)) + 1),
		s_max(r - 1), s_half(r / 2), SequentialComponent((uint32_t)floor(log2((double)n)) + 1, 1, 0, typehash(Btanh), sizeof(Btanh), alignof(Btanh), factory) {

		if (r % 2 != 0) throw std::runtime_error("Btanh: r must me a multiple of 2.");
		if (inputs.size() < input_width) throw std::runtime_error("Btanh: Not enough input nets given.");

		auto in = inputs.begin();
		for (uint32_t i = 0; i < input_width; i++) {
			this->inputs_host[i] = *in++;
		}
		outputs_host[0] = output;

		link_device_sim_function(Btanh);
	}

	def_device_statics(Btanh)

	void Btanh::reset_state() {
		state[0] = s_half;
		if (!circuit->host_only) cu(cudaMemcpy(((Btanh*)dev_ptr)->state, state, sizeof(uint32_t), cudaMemcpyHostToDevice));
	}

	void Btanh::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto current_progress = current_sim_progress();
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress = next_sim_progress();
		auto next_step_progress_word = next_sim_progress_word();

		uint32_t out_word = circuit->net_values_host[out_offset + current_progress_word] >> (32 - (current_progress % 32)); //present output, shifted for seamless continuation

		for (uint32_t i = current_progress_word; i < next_step_progress_word; i++) {
			uint32_t mask = 0x80000000u;
			for (uint32_t j = 0; j < 32; j++) {
				auto bit = 32 * i + j;
				if (bit < current_progress) { //ignore bits that have already been processed
					mask >>= 1;
					continue;
				}
				if (bit >= next_step_progress) { //last word ends before being full: shift for correct bit position
					out_word <<= (32 - j);
					break;
				}

				//add parallel counter value * 2 to state
				uint32_t factor = 2;
				for (uint32_t k = 0; k < input_width; k++) {
					if (circuit->net_values_host[input_offsets_host[k] + i] & mask > 0) state[0] += factor;
					factor <<= 1;
				}
				mask >>= 1;

				//subtract n from state (with saturation), corresponds to bipolar addition of input when combined with addition loop above
				if (state[0] <= n) state[0] = 0;
				else {
					state[0] -= n;
					if (state[0] > s_max) state[0] = s_max;
				}

				out_word <<= 1;
				putbit(out_word, state[0] >= s_half); //push zero if in lower half of states, one if in upper half of states
			}
			circuit->net_values_host[out_offset + i] = out_word;
			out_word = 0;
		}
	}

	//device simulation equivalent to host simulation
	__device__ void Btanh::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (Btanh*)comp;
		auto out_offset = g->output_offsets_dev[0];
		auto current_progress = g->current_sim_progress();
		auto current_progress_word = g->current_sim_progress_word();
		auto next_step_progress = g->next_sim_progress();
		auto next_step_progress_word = g->next_sim_progress_word();

		uint32_t out_word = g->net_values_dev[out_offset + current_progress_word] >> (32 - (current_progress % 32));

		for (uint32_t i = current_progress_word; i < next_step_progress_word; i++) {
			uint32_t mask = 0x80000000u;
			for (uint32_t j = 0; j < 32; j++) {
				auto bit = 32 * i + j;
				if (bit < current_progress) { //ignore bits that have already been processed
					mask >>= 1;
					continue;
				}
				if (bit >= next_step_progress) { //last word ends before being full: shift for correct bit position
					out_word <<= (32 - j);
					break;
				}

				//add parallel counter value * 2 to state
				uint32_t factor = 2;
				for (uint32_t k = 0; k < g->input_width; k++) {
					if (g->net_values_dev[g->input_offsets_dev[k] + i] & mask > 0) g->state[0] += factor;
					factor <<= 1;
				}
				mask >>= 1;

				//subtract n from state (with saturation), corresponds to bipolar addition of input when combined with addition loop above
				if (g->state[0] <= g->n) g->state[0] = 0;
				else {
					g->state[0] -= g->n;
					if (g->state[0] > g->s_max) g->state[0] = g->s_max;
				}

				out_word <<= 1;
				putbit(out_word, g->state[0] > g->s_half); //push zero if in lower half of states, one if in upper half of states
			}
			g->net_values_dev[out_offset + i] = out_word;
			out_word = 0;
		}
	}

	//see paper for calculation formulas
	uint32_t Btanh::calculate_r(uint32_t n, double s) {
		if (n == 0 || s <= 0) throw std::runtime_error("Btanh: Invalid inputs for r calculation");

		double q = 1.835 * pow(2. * n, -.5552);
		int64_t r_half = llround(((1. - s) * (n - 1.)) / (s * (1. - q))) + n;

		if (r_half < 1 || r_half > UINT32_MAX) throw std::runtime_error("Btanh: r calculation failed, no valid value found");

		return (uint32_t)(r_half * 2);
	}

}
