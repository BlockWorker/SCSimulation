#include "cuda_base.cuh"

#include <memory.h>

#include "libpopcnt.h"

#include "StochasticCircuit.cuh"
#include "MaxApprox.cuh"

#undef COMP_IMPEXP_SPEC
#define COMP_IMPEXP_SPEC SCSIMAPI

namespace scsim {

	MaxApprox::MaxApprox(uint32_t _num_inputs, uint32_t* inputs, uint32_t output, StochasticCircuitFactory* factory) : num_inputs(_num_inputs),
		CombinatorialComponent(_num_inputs, 1, 0, typehash(MaxApprox), sizeof(MaxApprox), alignof(MaxApprox), factory) {

		memcpy(this->inputs_host, inputs, _num_inputs * sizeof(uint32_t));
		outputs_host[0] = output;

		link_device_sim_function(MaxApprox);
	}

	MaxApprox::MaxApprox(uint32_t _num_inputs, uint32_t first_input, uint32_t output, StochasticCircuitFactory* factory) : num_inputs(_num_inputs),
		CombinatorialComponent(_num_inputs, 1, 0, typehash(MaxApprox), sizeof(MaxApprox), alignof(MaxApprox), factory) {

		for (uint32_t i = 0; i < _num_inputs; i++) {
			this->inputs_host[i] = first_input + i;
		}

		outputs_host[0] = output;

		link_device_sim_function(MaxApprox);
	}

	MaxApprox::MaxApprox(std::initializer_list<uint32_t> inputs, uint32_t output, StochasticCircuitFactory* factory) : num_inputs(inputs.size()),
		CombinatorialComponent(inputs.size(), 1, 0, typehash(MaxApprox), sizeof(MaxApprox), alignof(MaxApprox), factory) {

		auto in = inputs.begin();
		for (uint32_t i = 0; i < inputs.size(); i++) {
			this->inputs_host[i] = *in++;
		}

		outputs_host[0] = output;

		link_device_sim_function(MaxApprox);
	}

	def_device_statics(MaxApprox)

	void MaxApprox::simulate_step_host() {
		auto out_offset = output_offsets_host[0];
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			uint32_t max_ones = 0;
			uint32_t max_index = 0;

			for (auto j = 0; j < num_inputs; j++) { //iterate over inputs, find maximum
				uint32_t ones = popcnt64(circuit->net_values_host[input_offsets_host[j] + i]);
				if (ones > max_ones) {
					max_ones = ones;
					max_index = j;
				}
			}

			circuit->net_values_host[out_offset + i] = circuit->net_values_host[input_offsets_host[max_index] + i]; //output maximum word
		}
	}

	//device simulation equivalent to host simulation (but only one word per thread)
	__device__ void MaxApprox::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (MaxApprox*)comp;

		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			uint32_t max_ones = 0;
			uint32_t max_index = 0;

			for (auto j = 0; j < g->num_inputs; j++) { //iterate over inputs, find maximum
				uint32_t ones = __popc(g->net_values_dev[g->input_offsets_dev[j] + i]);
				if (ones > max_ones) {
					max_ones = ones;
					max_index = j;
				}
			}

			g->net_values_dev[g->output_offsets_dev[0] + i] = g->net_values_dev[g->input_offsets_dev[max_index] + i]; //output maximum word
		}
	}

}
