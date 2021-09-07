﻿#include "cuda_base.cuh"

#include "CircuitComponent.cuh"
#include "StochasticCircuit.cuh"

namespace scsim {

	CircuitComponent::CircuitComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t type, size_t size, size_t align) : component_type(type), mem_obj_size(size), mem_align(align),
		num_inputs(num_inputs), num_outputs(num_outputs) {
		circuit = nullptr;
		net_values_dev = nullptr;
		net_progress_dev = nullptr;
		sim_length = 0;
		dev_ptr = nullptr;
		simulate_step_dev_ptr = nullptr;
		calc_progress_dev_ptr = nullptr;
		progress_host_ptr = nullptr;
		progress_dev_ptr = nullptr;
		inputs_host = (uint32_t*)malloc(num_inputs * sizeof(uint32_t));
		outputs_host = (uint32_t*)malloc(num_outputs * sizeof(uint32_t));
		input_offsets_host = (size_t*)malloc(num_inputs * sizeof(size_t));
		output_offsets_host = (size_t*)malloc(num_outputs * sizeof(size_t));
		inputs_dev = nullptr;
		outputs_dev = nullptr;
		input_offsets_dev = nullptr;
		output_offsets_dev = nullptr;
	}

	CircuitComponent::~CircuitComponent() {
		free(inputs_host);
		free(outputs_host);
		free(input_offsets_host);
		free(output_offsets_host);
		if (!circuit->host_only) {
			cudaFree(inputs_dev);
			cudaFree(outputs_dev);
			cudaFree(input_offsets_dev);
			cudaFree(output_offsets_dev);
		}
	}

	__host__ __device__ uint32_t CircuitComponent::current_sim_progress() const {
#ifdef __CUDA_ARCH__
		return *progress_dev_ptr;
#else
		return *progress_host_ptr;
#endif
	}

	__host__ __device__ uint32_t CircuitComponent::current_sim_progress_word() const {
#ifdef __CUDA_ARCH__
		return *progress_dev_ptr / 32;
#else
		return *progress_host_ptr / 32;
#endif
	}

	__host__ __device__ uint32_t CircuitComponent::next_sim_progress() const {
#ifdef __CUDA_ARCH__
		return *(progress_dev_ptr + 1);
#else
		return *(progress_host_ptr + 1);
#endif
	}

	__host__ __device__ uint32_t CircuitComponent::next_sim_progress_word() const {
#ifdef __CUDA_ARCH__
		return (*(progress_dev_ptr + 1) + 31) / 32;
#else
		return (*(progress_host_ptr + 1) + 31) / 32;
#endif
	}

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

		*progress_host_ptr = current_progress;
		*(progress_host_ptr + 1) = next_step_progress;
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

	void CircuitComponent::calculate_io_offsets() {
		size_t* dev_in = (size_t*)malloc(num_inputs * sizeof(size_t));
		size_t* dev_out = (size_t*)malloc(num_outputs * sizeof(size_t));
		if (dev_in == nullptr || dev_out == nullptr) return;

		for (uint32_t i = 0; i < num_inputs; i++) {
			input_offsets_host[i] = (size_t)circuit->sim_length_words * inputs_host[i];
			dev_in[i] = (size_t)circuit->net_values_dev_pitch / sizeof(uint32_t) * inputs_host[i];
		}

		for (uint32_t i = 0; i < num_outputs; i++) {
			output_offsets_host[i] = (size_t)circuit->sim_length_words * outputs_host[i];
			dev_out[i] = (size_t)circuit->net_values_dev_pitch / sizeof(uint32_t) * outputs_host[i];
		}

		if (!circuit->host_only) {
			cu(cudaMemcpy(input_offsets_dev, dev_in, num_inputs * sizeof(size_t), cudaMemcpyHostToDevice));
			cu(cudaMemcpy(output_offsets_dev, dev_out, num_outputs * sizeof(size_t), cudaMemcpyHostToDevice));
		}

		free(dev_in);
		free(dev_out);
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

		*comp->progress_dev_ptr = current_progress;
		*(comp->progress_dev_ptr + 1) = next_step_progress;
	}

	void CircuitComponent::init_with_circuit(StochasticCircuit* circuit, uint32_t* progress_host_ptr, uint32_t* progress_dev_ptr) {
		this->circuit = circuit;
		this->progress_host_ptr = progress_host_ptr;
		if (!circuit->host_only) {
			net_values_dev = circuit->net_values_dev;
			net_progress_dev = circuit->net_progress_dev;
			sim_length = circuit->sim_length;
			this->progress_dev_ptr = progress_dev_ptr;
			cu(cudaMalloc(&input_offsets_dev, num_inputs * sizeof(size_t)));
			cu(cudaMalloc(&output_offsets_dev, num_outputs * sizeof(size_t)));
			cu(cudaMalloc(&inputs_dev, num_inputs * sizeof(uint32_t)));
			cu(cudaMalloc(&outputs_dev, num_outputs * sizeof(uint32_t)));
			cu(cudaMemcpy(inputs_dev, inputs_host, num_inputs * sizeof(uint32_t), cudaMemcpyHostToDevice));
			cu(cudaMemcpy(outputs_dev, outputs_host, num_outputs * sizeof(uint32_t), cudaMemcpyHostToDevice));
		}

		calculate_io_offsets();
	}

	__global__ void link_default_simprog_kern(CircuitComponent* comp) {
		comp->calc_progress_dev_ptr = &comp->_calculate_simulation_progress_dev;
	}

	void CircuitComponent::link_dev_functions() {
		link_default_simprog_kern<<<1, 1>>>(dev_ptr);
	}

}
