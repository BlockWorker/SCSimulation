#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory.h>

#include "SequentialComponent.cuh"
#include "StochasticCircuit.cuh"

SequentialComponent::SequentialComponent(uint32_t num_inputs, uint32_t num_outputs, size_t state_size, uint32_t type, size_t size) : CircuitComponent(num_inputs, num_outputs, type | 0x80000000u, size), state_size(state_size) {
	state_host = (uint32_t*)malloc(state_size * sizeof(uint32_t));
	state_dev = nullptr;
}

SequentialComponent::~SequentialComponent() {
	free(state_host);
	if (!circuit->host_only) cudaFree(state_dev);
}

void SequentialComponent::reset_state() {
	memset(state_host, 0, state_size * sizeof(uint32_t));
}

void SequentialComponent::copy_state_host_to_device() {
	if (!circuit->host_only) cudaMemcpy(state_dev, state_host, state_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void SequentialComponent::copy_state_device_to_host() {
	if (!circuit->host_only) cudaMemcpy(state_host, state_dev, state_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

void SequentialComponent::init_with_circuit(StochasticCircuit* circuit) {
	reset_state();

	if (!circuit->host_only) cudaMalloc(&state_dev, state_size * sizeof(uint32_t));

	CircuitComponent::init_with_circuit(circuit);
}