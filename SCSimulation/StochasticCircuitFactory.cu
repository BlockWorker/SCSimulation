﻿#include "cuda_base.cuh"

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
	max_component_size = 0;
	max_component_align = 0;
}

StochasticCircuit* StochasticCircuitFactory::create_circuit() {
	if (sim_length == 0 || num_nets == 0 || components.size() == 0) throw;

	StochasticCircuit* circuit = nullptr;

	size_t sim_length_words = (sim_length + 31) / 32;

	uint32_t* net_values_host = (uint32_t*)malloc(sim_length_words * num_nets * sizeof(uint32_t));
	uint32_t* net_progress_host = (uint32_t*)calloc(num_nets, sizeof(uint32_t));
	CircuitComponent** components_host = (CircuitComponent**)malloc(components.size() * sizeof(CircuitComponent*));
	CircuitComponent** components_dev = nullptr;

	if (net_values_host == nullptr || net_progress_host == nullptr || components_host == nullptr) throw;

	size_t component_pitch;
	char* component_array_host, * component_array_dev;
	size_t component_array_dev_pitch;

	std::sort(components.begin(), components.end(), [](auto a, auto b) { return a->component_type < b->component_type; });
	memcpy(components_host, components.data(), components.size() * sizeof(CircuitComponent*));

	if (host_only) {
		circuit = new StochasticCircuit(sim_length, num_nets, net_values_host, net_progress_host, num_comb_comp, num_seq_comp, components_host);
		for (uint32_t i = 0; i < components.size(); i++) {
			components_host[i]->init_with_circuit(circuit);
		}
	} else {
		uint32_t* net_values_dev, * net_progress_dev;
		size_t net_pitch_dev;

		cu(cudaMallocPitch(&net_values_dev, &net_pitch_dev, sim_length_words * sizeof(uint32_t), num_nets));
		cu(cudaMalloc(&net_progress_dev, num_nets * sizeof(uint32_t)));
		cu(cudaMalloc(&components_dev, components.size() * sizeof(CircuitComponent*)));

		auto max_comp_size_misalignment = max_component_size % max_component_align;
		if (max_comp_size_misalignment == 0) component_pitch = max_component_size;
		else component_pitch = max_component_size - max_comp_size_misalignment + max_component_align;

		component_array_host = (char*)malloc(components.size() * component_pitch);
		if (component_array_host == nullptr) throw;

		cu(cudaMallocPitch(&component_array_dev, &component_array_dev_pitch, component_pitch, components.size()));

		circuit = new StochasticCircuit(sim_length, num_nets, net_values_host, net_values_dev, net_pitch_dev, net_progress_host, net_progress_dev, num_comb_comp, num_seq_comp,
			components_host, components_dev, component_array_host, component_pitch, component_array_dev, component_array_dev_pitch);

		for (uint32_t i = 0; i < components.size(); i++) {
			auto comp = components_host[i];

			components_host[i] = (CircuitComponent*)(component_array_host + i * component_pitch);

			comp->dev_ptr = (CircuitComponent*)(component_array_dev + i * component_array_dev_pitch);
			cu(cudaMemcpy((void*)(components_dev + i), &comp->dev_ptr, sizeof(CircuitComponent*), cudaMemcpyHostToDevice));

			comp->init_with_circuit(circuit);

			memcpy(components_host[i], comp, comp->mem_obj_size);
			operator delete(comp);
		}

		circuit->reset_circuit();
		circuit->copy_data_to_device();

		for (uint32_t i = 0; i < components.size(); i++) {
			components_host[i]->link_devstep();
		}

		circuit->copy_data_from_device();
	}

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

std::pair<uint32_t, uint32_t> StochasticCircuitFactory::add_nets(uint32_t count) {
	for (uint32_t i = 0; i < count; i++) driven_nets.push_back(false);
	auto first = num_nets;
	num_nets += count;
	return std::make_pair(first, num_nets - 1);
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

	if (component->mem_obj_size > max_component_size) max_component_size = component->mem_obj_size;
	if (component->mem_align > max_component_align) max_component_align = component->mem_align;

	components.push_back(component);

	return components.size() - 1;
}