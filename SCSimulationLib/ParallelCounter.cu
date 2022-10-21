#include "cuda_base.cuh"

#include <memory.h>

#include "StochasticCircuit.cuh"
#include "ParallelCounter.cuh"

#undef COMP_IMPEXP_SPEC
#define COMP_IMPEXP_SPEC SCSIMAPI

namespace scsim {

	ParallelCounter::ParallelCounter(uint32_t _num_inputs, uint32_t* inputs, uint32_t* outputs, StochasticCircuitFactory* factory) : num_inputs(_num_inputs), num_outputs((uint32_t)floor(log2((double)_num_inputs)) + 1),
		CombinatorialComponent(_num_inputs, (uint32_t)floor(log2((double)_num_inputs)) + 1, 0, typehash(ParallelCounter), sizeof(ParallelCounter), alignof(ParallelCounter), factory) {

		memcpy(this->inputs_host, inputs, _num_inputs * sizeof(uint32_t));
		memcpy(this->outputs_host, outputs, num_outputs * sizeof(uint32_t));

		link_device_sim_function(ParallelCounter);
	}

	ParallelCounter::ParallelCounter(uint32_t _num_inputs, uint32_t first_input, uint32_t first_output, StochasticCircuitFactory* factory) : num_inputs(_num_inputs), num_outputs((uint32_t)floor(log2((double)_num_inputs)) + 1),
		CombinatorialComponent(_num_inputs, (uint32_t)floor(log2((double)_num_inputs)) + 1, 0, typehash(ParallelCounter), sizeof(ParallelCounter), alignof(ParallelCounter), factory) {

		for (uint32_t i = 0; i < _num_inputs; i++) {
			this->inputs_host[i] = first_input + i;
		}

		for (uint32_t i = 0; i < num_outputs; i++) {
			this->outputs_host[i] = first_output + i;
		}

		link_device_sim_function(ParallelCounter);
	}

	ParallelCounter::ParallelCounter(std::initializer_list<uint32_t> inputs, std::initializer_list<uint32_t> outputs, StochasticCircuitFactory* factory) : num_inputs(inputs.size()), num_outputs((uint32_t)floor(log2((double)inputs.size())) + 1),
		CombinatorialComponent(inputs.size(), (uint32_t)floor(log2((double)inputs.size())) + 1, 0, typehash(ParallelCounter), sizeof(ParallelCounter), alignof(ParallelCounter), factory) {

		if (outputs.size() < num_outputs) throw std::runtime_error("ParallelCounter: Not enough output nets given.");

		auto in = inputs.begin();
		for (uint32_t i = 0; i < inputs.size(); i++) {
			this->inputs_host[i] = *in++;
		}

		auto out = outputs.begin();
		for (uint32_t i = 0; i < num_outputs; i++) {
			this->outputs_host[i] = *out++;
		}

		link_device_sim_function(ParallelCounter);
	}

	def_device_statics(ParallelCounter)

	void ParallelCounter::simulate_step_host() {
		auto current_progress_word = current_sim_progress_word();
		auto next_step_progress_word = next_sim_progress_word();

		for (auto i = current_progress_word; i < next_step_progress_word; i++) {
			for (auto k = 0; k < num_outputs; k++) { //initialize all outputs to 0
				circuit->net_values_host[output_offsets_host[k] + i] = 0;
			}

			for (auto j = 0; j < num_inputs; j++) { //iterate over inputs
				auto carry = circuit->net_values_host[input_offsets_host[j] + i]; //carry from previous bit, starts out as input value
				for (auto k = 0; k < num_outputs; k++) { //ripple increment output bits one after another
					auto curr = circuit->net_values_host[output_offsets_host[k] + i]; //store current value of bit
					circuit->net_values_host[output_offsets_host[k] + i] ^= carry; //update bit with XOR of carry
					carry &= curr; //update carry for next bit
					if (carry == 0) break; //cancel loop early once no more carry
				}
			}
		}
	}

	//device simulation equivalent to host simulation (but only one word per thread)
	__device__ void ParallelCounter::_simulate_step_dev(CircuitComponent* comp) {
		auto g = (ParallelCounter*)comp;

		auto i = g->current_sim_progress_word() + blockIdx.y * blockDim.y + threadIdx.y;
		if (i < g->next_sim_progress_word()) {
			for (auto k = 0; k < g->num_outputs; k++) { //initialize all outputs to 0
				g->net_values_dev[g->output_offsets_dev[k] + i] = 0;
			}

			for (auto j = 0; j < g->num_inputs; j++) { //iterate over inputs
				auto carry = g->net_values_dev[g->input_offsets_dev[j] + i]; //carry from previous bit, starts out as input value
				for (auto k = 0; k < g->num_outputs; k++) { //ripple increment output bits one after another
					auto curr = g->net_values_dev[g->output_offsets_dev[k] + i]; //store current value of bit
					g->net_values_dev[g->output_offsets_dev[k] + i] ^= carry; //update bit with XOR of carry
					carry &= curr; //update carry for next bit
					if (carry == 0) break; //cancel loop early once no more carry
				}
			}
		}
	}

}
