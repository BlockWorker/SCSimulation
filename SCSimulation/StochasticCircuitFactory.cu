#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>

#include "StochasticCircuitFactory.cuh"
#include "StochasticCircuit.cuh"
#include "CircuitComponent.cuh"
#include "CombinatorialComponent.cuh"
#include "SequentialComponent.cuh"

StochasticCircuitFactory::StochasticCircuitFactory() {
	reset();
}

StochasticCircuitFactory::~StochasticCircuitFactory() {

}

void StochasticCircuitFactory::reset() {
	host_only = false;
	sim_length = 0;
	num_nets = 0;
	num_comb_comp = 0;
	num_seq_comp = 0;
	components.clear();
	driven_nets.clear();
}

StochasticCircuit* StochasticCircuitFactory::create_circuit() {
	StochasticCircuit* circuit = nullptr;

	size_t sim_length_words = (sim_length + 31) / 32;

	uint32_t* net_values_host = (uint32_t*)malloc(sim_length_words * num_nets * sizeof(uint32_t));
	uint32_t* net_progress_host = (uint32_t*)calloc(num_nets, sizeof(uint32_t));
	CircuitComponent** components_host = (CircuitComponent**)malloc(components.size() * sizeof(CircuitComponent*));
	CircuitComponent** components_dev = nullptr;

	if (net_values_host == nullptr || net_progress_host == nullptr || components_host == nullptr) throw;

	std::sort(components.begin(), components.end(), [](auto a, auto b) { return a->component_type < b->component_type; });
	memcpy(components_host, components.data(), components.size() * sizeof(CircuitComponent*));

	if (host_only) circuit = new StochasticCircuit(sim_length, num_nets, net_values_host, net_progress_host, num_comb_comp, num_seq_comp, components_host);
	else {
		uint32_t* net_values_dev, * net_progress_dev;
		size_t net_pitch_dev;

		cudaMallocPitch(&net_values_dev, &net_pitch_dev, sim_length_words * sizeof(uint32_t), num_nets);
		cudaMalloc(&net_progress_dev, num_nets * sizeof(uint32_t));
		cudaMalloc(&components_dev, components.size() * sizeof(CircuitComponent*));

		circuit = new StochasticCircuit(sim_length, num_nets, net_values_host, net_values_dev, net_pitch_dev, net_progress_host, net_progress_dev, num_comb_comp, num_seq_comp, components_host, components_dev);
	}

	for (uint32_t i = 0; i < components.size(); i++) {
		auto comp = components_host[i];

		comp->init_with_circuit(circuit);

		if (!host_only) cudaMemcpy((void*)(components_dev + i), &comp->dev_ptr, sizeof(CircuitComponent*), cudaMemcpyHostToDevice);
	}

	circuit->reset_circuit();
	if (!host_only) circuit->copy_data_to_device();

	return circuit;
}

void StochasticCircuitFactory::set_host_only(bool host_only) {
	this->host_only = host_only;
}

void StochasticCircuitFactory::set_sim_length(uint32_t sim_length) {
	this->sim_length = sim_length;
}

uint32_t StochasticCircuitFactory::add_net() {
	driven_nets.push_back(false);
	return num_nets++;
}

void StochasticCircuitFactory::add_component(CombinatorialComponent* component) {
	auto index = add_component_internal(component);
	num_comb_comp++;
}

void StochasticCircuitFactory::add_component(SequentialComponent* component) {
	auto index = add_component_internal(component);
	num_seq_comp++;
}

uint32_t StochasticCircuitFactory::add_component_internal(CircuitComponent* component) {
	for (size_t i = 0; i < component->num_outputs; i++) {
		auto net = component->outputs[i];
		if (net >= num_nets || driven_nets[net]) throw;
	}

	for (size_t i = 0; i < component->num_outputs; i++) {
		auto net = component->outputs[i];
		driven_nets[net] = true;
	}

	components.push_back(component);

	return components.size() - 1;
}
