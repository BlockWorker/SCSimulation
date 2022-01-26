#include "cuda_base.cuh"

#include "CircuitComponent.cuh"
#include "StochasticCircuit.cuh"
#include "StochasticCircuitFactory.cuh"

namespace scsim {

	template <class T1>
	__global__ void link_defprog_kern(void (**func_pointer)(scsim::CircuitComponent*)) {
		*func_pointer = &scsim::CircuitComponent::_calculate_simulation_progress_dev;
	}

	CircuitComponent::CircuitComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t type, size_t size, size_t align, StochasticCircuitFactory* factory) : component_type(type),
		mem_obj_size(size), mem_align(align), num_inputs(num_inputs), num_outputs(num_outputs) {
		circuit = nullptr;
		net_values_dev = nullptr;
		net_progress_dev = nullptr;
		sim_length = 0;
		dev_ptr = nullptr;
		simulate_step_dev_ptr = nullptr;
		current_progress_host_ptr = nullptr;
		next_step_progress_host_ptr = nullptr;
		current_progress_dev_ptr = nullptr;
		next_step_progress_dev_ptr = nullptr;
		io_array_offset = factory->component_io.size();
		factory->component_io.resize(io_array_offset + num_inputs + num_outputs); //make space in factory for IO
		inputs_host = factory->component_io.data() + io_array_offset; //link IO to factory-stored location for now
		outputs_host = inputs_host + num_inputs;
		input_offsets_host = nullptr;
		output_offsets_host = nullptr;
		inputs_dev = nullptr;
		outputs_dev = nullptr;
		input_offsets_dev = nullptr;
		output_offsets_dev = nullptr;

		if (!_dev_link_initialized && !factory->host_only) { //link default device-side progress calculation function
			_dev_link_initialized = true;
			void (*pointer)(CircuitComponent*);
			void (**dev_pointer)(CircuitComponent*);
			cu(cudaMalloc(&dev_pointer, sizeof(void (*)(CircuitComponent*))));
			link_defprog_kern<CircuitComponent><<<1, 1>>>(dev_pointer);
			cu_kernel_errcheck_nosync();
			cu(cudaMemcpy(&pointer, dev_pointer, sizeof(void (*)(CircuitComponent*)), cudaMemcpyDeviceToHost));
			_simprog_ptr = *pointer;
		}
		calc_progress_dev_ptr = _simprog_ptr;
	}

	CircuitComponent::~CircuitComponent() {

	}

	bool SCSIMAPI CircuitComponent::_dev_link_initialized = false;
	void SCSIMAPI (*CircuitComponent::_simprog_ptr)(CircuitComponent*) = nullptr;

	StochasticCircuit* CircuitComponent::get_circuit() const {
		return circuit;
	}

	void CircuitComponent::calculate_simulation_progress_host() {
		uint32_t current_progress = circuit->sim_length;
		for (uint32_t i = 0; i < num_outputs; i++) {
			auto out_progress = circuit->net_progress_host[outputs_host[i]];
			if (out_progress < current_progress) { //current progress equals the minimum progress of output nets
				current_progress = out_progress;
			}
		}

		uint32_t next_step_progress = circuit->sim_length;
		for (uint32_t i = 0; i < num_inputs; i++) {
			auto in_progress = circuit->net_progress_host[inputs_host[i]];
			if (in_progress < next_step_progress) { //next step progress equals the minimum progress of input nets
				next_step_progress = in_progress;
			}
		}

		if (next_step_progress < current_progress) {
			next_step_progress = current_progress;
		}

		*current_progress_host_ptr = current_progress;
		*next_step_progress_host_ptr = next_step_progress;
	}

	__device__ void CircuitComponent::calculate_simulation_progress_dev() {
		(*calc_progress_dev_ptr)(this);
	}

	__device__ void CircuitComponent::simulate_step_dev() {
		(*simulate_step_dev_ptr)(this);
	}

	void CircuitComponent::sim_step_finished_host() {
		for (size_t i = 0; i < num_outputs; i++) {
			circuit->net_progress_host[outputs_host[i]] = next_sim_progress();
		}
	}

	__device__ void CircuitComponent::sim_step_finished_dev() {
		for (size_t i = 0; i < num_outputs; i++) {
			net_progress_dev[outputs_dev[i]] = next_sim_progress();
		}
	}

	void CircuitComponent::calculate_io_offsets(size_t* dev_offset_scratchpad) {
		for (uint32_t i = 0; i < num_inputs; i++) {
			input_offsets_host[i] = (size_t)circuit->sim_length_words * inputs_host[i];
		}

		for (uint32_t i = 0; i < num_outputs; i++) {
			output_offsets_host[i] = (size_t)circuit->sim_length_words * outputs_host[i];
		}

		if (!circuit->host_only) {
			size_t* dev_in = dev_offset_scratchpad + io_array_offset;
			size_t* dev_out = dev_in + num_inputs;

			for (uint32_t i = 0; i < num_inputs; i++) {
				dev_in[i] = (size_t)circuit->net_values_dev_pitch / sizeof(uint32_t) * inputs_host[i];
			}

			for (uint32_t i = 0; i < num_outputs; i++) {
				dev_out[i] = (size_t)circuit->net_values_dev_pitch / sizeof(uint32_t) * outputs_host[i];
			}
		}
	}

	__device__ void CircuitComponent::_calculate_simulation_progress_dev(CircuitComponent* comp) {
		uint32_t current_progress = comp->sim_length;
		for (uint32_t i = 0; i < comp->num_outputs; i++) {
			auto out_progress = comp->net_progress_dev[comp->outputs_dev[i]];
			if (out_progress < current_progress) { //current progress equals the minimum progress of output nets
				current_progress = out_progress;
			}
		}

		uint32_t next_step_progress = comp->sim_length;
		for (uint32_t i = 0; i < comp->num_inputs; i++) {
			auto in_progress = comp->net_progress_dev[comp->inputs_dev[i]];
			if (in_progress < next_step_progress) { //next step progress equals the minimum progress of input nets
				next_step_progress = in_progress;
			}
		}

		if (next_step_progress < current_progress) {
			next_step_progress = current_progress;
		}

		*comp->current_progress_dev_ptr = current_progress;
		*comp->next_step_progress_dev_ptr = next_step_progress;
	}

	void CircuitComponent::init_with_circuit(StochasticCircuit* circuit, uint32_t* progress_host_ptr, uint32_t* progress_dev_ptr, size_t* dev_offset_scratchpad) {
		this->circuit = circuit;
		current_progress_host_ptr = progress_host_ptr;
		next_step_progress_host_ptr = progress_host_ptr + 1;
		inputs_host = circuit->component_io_host + io_array_offset; //re-link to circuit-stored inputs and outputs, as well as offsets
		outputs_host = inputs_host + num_inputs;
		input_offsets_host = circuit->component_io_offsets_host + io_array_offset;
		output_offsets_host = input_offsets_host + num_inputs;
		if (!circuit->host_only) {
			net_values_dev = circuit->net_values_dev;
			net_progress_dev = circuit->net_progress_dev;
			sim_length = circuit->sim_length;
			current_progress_dev_ptr = progress_dev_ptr;
			next_step_progress_dev_ptr = progress_dev_ptr + 1;
			inputs_dev = circuit->component_io_dev + io_array_offset;
			outputs_dev = inputs_dev + num_inputs;
			input_offsets_dev = circuit->component_io_offsets_dev + io_array_offset;
			output_offsets_dev = input_offsets_dev + num_inputs;
		}

		calculate_io_offsets(dev_offset_scratchpad);
	}

}
