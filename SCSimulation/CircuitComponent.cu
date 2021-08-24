
#include <stdint.h>

#include "CircuitComponent.cuh"
#include "StochasticCircuit.cuh"

CircuitComponent::CircuitComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t type, size_t size, size_t align) : component_type(type), mem_obj_size(size), mem_align(align),
	num_inputs(num_inputs), num_outputs(num_outputs) {
	circuit = nullptr;
	net_values_dev = nullptr;
	dev_ptr = nullptr;
	simulate_step_dev_ptr = nullptr;
	current_progress = 0;
	current_progress_word = 0;
	next_step_progress = 0;
	next_step_progress_word = 0;
	inputs = (uint32_t*)malloc(num_inputs * sizeof(uint32_t));
	outputs = (uint32_t*)malloc(num_outputs * sizeof(uint32_t));
	input_offsets_host = (size_t*)malloc(num_inputs * sizeof(size_t));
	output_offsets_host = (size_t*)malloc(num_outputs * sizeof(size_t));
	input_offsets_dev = nullptr;
	output_offsets_dev = nullptr;
}

CircuitComponent::~CircuitComponent() {
	free(inputs);
	free(outputs);
	free(input_offsets_host);
	free(output_offsets_host);
	if (!circuit->host_only) {
		cu(cudaFree(input_offsets_dev));
		cu(cudaFree(output_offsets_dev));
	}
}

uint32_t CircuitComponent::current_sim_progress() const {
	return current_progress;
}

uint32_t CircuitComponent::current_sim_progress_word() const {
	return current_progress_word;
}

uint32_t CircuitComponent::next_sim_progress() const {
	return next_step_progress;
}

uint32_t CircuitComponent::next_sim_progress_word() const {
	return next_step_progress_word;
}

StochasticCircuit* CircuitComponent::get_circuit() const {
	return circuit;
}

void CircuitComponent::calculate_simulation_progress() {
	current_progress = UINT32_MAX;
	for (uint32_t i = 0; i < num_outputs; i++) {
		auto out_progress = circuit->net_progress_host[outputs[i]];
		if (out_progress < current_progress) {
			current_progress = out_progress;
		}
	}

	next_step_progress = UINT32_MAX;
	for (uint32_t i = 0; i < num_inputs; i++) {
		auto in_progress = circuit->net_progress_host[inputs[i]];
		if (in_progress < next_step_progress) {
			next_step_progress = in_progress;
		}
	}

	if (next_step_progress < current_progress) {
		next_step_progress = current_progress;
	}

	current_progress_word = current_progress / 32;
	next_step_progress_word = (next_step_progress + 31) / 32;
}

__device__ void CircuitComponent::simulate_step_dev() {
	(*simulate_step_dev_ptr)(this);
}

void CircuitComponent::sim_step_finished() {
	for (size_t i = 0; i < num_outputs; i++) {
		circuit->net_progress_host[outputs[i]] = next_step_progress;
	}
}

void CircuitComponent::calculate_io_offsets() {
	size_t* dev_in = (size_t*)malloc(num_inputs * sizeof(size_t));
	size_t* dev_out = (size_t*)malloc(num_outputs * sizeof(size_t));
	if (dev_in == nullptr || dev_out == nullptr) return;

	for (uint32_t i = 0; i < num_inputs; i++) {
		input_offsets_host[i] = (size_t)circuit->sim_length_words * inputs[i];
		dev_in[i] = (size_t)circuit->net_values_dev_pitch / sizeof(uint32_t) * inputs[i];
	}

	for (uint32_t i = 0; i < num_outputs; i++) {
		output_offsets_host[i] = (size_t)circuit->sim_length_words * outputs[i];
		dev_out[i] = (size_t)circuit->net_values_dev_pitch / sizeof(uint32_t) * outputs[i];
	}

	if (!circuit->host_only) {
		cu(cudaMemcpy(input_offsets_dev, dev_in, num_inputs * sizeof(size_t), cudaMemcpyHostToDevice));
		cu(cudaMemcpy(output_offsets_dev, dev_out, num_outputs * sizeof(size_t), cudaMemcpyHostToDevice));
	}

	free(dev_in);
	free(dev_out);
}

void CircuitComponent::init_with_circuit(StochasticCircuit* circuit) {
	this->circuit = circuit;
	if (!circuit->host_only) {
		net_values_dev = circuit->net_values_dev;
		cu(cudaMalloc(&input_offsets_dev, num_inputs * sizeof(size_t)));
		cu(cudaMalloc(&output_offsets_dev, num_outputs * sizeof(size_t)));
	}

	calculate_io_offsets();
}
