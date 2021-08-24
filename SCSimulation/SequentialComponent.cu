#include "cuda_base.cuh"

#include <memory.h>

#include "SequentialComponent.cuh"
#include "StochasticCircuit.cuh"

SequentialComponent::SequentialComponent(uint32_t num_inputs, uint32_t num_outputs, size_t state_size, uint32_t type, size_t size, size_t align) : CircuitComponent(num_inputs, num_outputs, type | 0x80000000u, size, align),
	state_size(state_size), state_fixed(state_size <= SEQUENTIAL_FIXED_STATE_SIZE) {
	var_state_host = state_fixed ? nullptr : (uint32_t*)malloc(state_size * sizeof(uint32_t));
	var_state_dev = nullptr;
}

SequentialComponent::~SequentialComponent() {
	if (state_fixed) return;
	free(var_state_host);
	if (!circuit->host_only) cu(cudaFree(var_state_dev));
}

void SequentialComponent::reset_state() {
	memset(state_ptr(), 0, state_size * sizeof(uint32_t));
}

void SequentialComponent::copy_state_host_to_device() {
	if (!circuit->host_only && !state_fixed) cu(cudaMemcpy(var_state_dev, var_state_host, state_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

void SequentialComponent::copy_state_device_to_host() {
	if (!circuit->host_only && !state_fixed) cu(cudaMemcpy(var_state_host, var_state_dev, state_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void SequentialComponent::init_with_circuit(StochasticCircuit* circuit) {
	reset_state();

	if (!circuit->host_only && !state_fixed) cu(cudaMalloc(&var_state_dev, state_size * sizeof(uint32_t)));

	CircuitComponent::init_with_circuit(circuit);
}