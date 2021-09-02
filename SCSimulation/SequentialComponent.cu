﻿#include "cuda_base.cuh"

#include <memory.h>

#include "SequentialComponent.cuh"
#include "StochasticCircuit.cuh"

namespace scsim {

	//sequential component types have upper bit set -> sequential components always come last once sorted
	SequentialComponent::SequentialComponent(uint32_t num_inputs, uint32_t num_outputs, size_t state_size, uint32_t type, size_t size, size_t align) : 
		CircuitComponent(num_inputs, num_outputs, type | 0x80000000u, size, align),
		state_size(state_size), state_is_fixed(state_size <= SEQUENTIAL_FIXED_STATE_SIZE) {
		var_state_host = state_is_fixed ? nullptr : (uint32_t*)malloc(state_size * sizeof(uint32_t));
		var_state_dev = nullptr;
	}

	SequentialComponent::~SequentialComponent() {
		if (state_is_fixed) return;
		free(var_state_host);
		if (!circuit->host_only) cudaFree(var_state_dev);
	}

	void SequentialComponent::reset_state() {
		memset(state_ptr(), 0, state_size * sizeof(uint32_t));
	}

	void SequentialComponent::copy_state_host_to_device() {
		if (!circuit->host_only && !state_is_fixed) cu(cudaMemcpy(var_state_dev, var_state_host, state_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
	}

	void SequentialComponent::copy_state_device_to_host() {
		if (!circuit->host_only && !state_is_fixed) cu(cudaMemcpy(var_state_host, var_state_dev, state_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	}

	void SequentialComponent::init_with_circuit(StochasticCircuit* circuit) {
		reset_state();

		if (!circuit->host_only && !state_is_fixed) {
			cu(cudaMalloc(&var_state_dev, state_size * sizeof(uint32_t)));
			copy_state_host_to_device();
		}

		CircuitComponent::init_with_circuit(circuit);
	}

	__host__ __device__ uint32_t* SequentialComponent::state_ptr() const {
		if (state_is_fixed) return (uint32_t*)this->fixed_state;
#ifdef __CUDA_ARCH__
		else return var_state_dev;
#else
		else return var_state_host;
#endif
	}

}
