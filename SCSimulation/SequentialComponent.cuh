#pragma once

#include <stdint.h>
#include "CircuitComponent.cuh"
#include "dll.h"

#define SEQUENTIAL_FIXED_STATE_SIZE 1

namespace scsim {

	class SCSIMAPI SequentialComponent : public CircuitComponent
	{
	public:
		const size_t state_size;

		SequentialComponent(uint32_t num_inputs, uint32_t num_outputs, size_t state_size, uint32_t type, size_t size, size_t align);
		virtual ~SequentialComponent();

		virtual void reset_state() override;

		virtual void copy_state_host_to_device() override;
		virtual void copy_state_device_to_host() override;

	protected:
		uint32_t fixed_state[SEQUENTIAL_FIXED_STATE_SIZE] = { 0 };
		uint32_t* var_state_host;
		uint32_t* var_state_dev;
		const bool state_fixed;

		__host__ __device__ uint32_t* state_ptr() const;

		virtual void init_with_circuit(StochasticCircuit* circuit) override;

	};

}
