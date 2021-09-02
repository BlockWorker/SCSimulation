﻿#pragma once

#include <stdint.h>
#include "CircuitComponent.cuh"
#include "dll.h"

#define SEQUENTIAL_FIXED_STATE_SIZE 1

namespace scsim {

	class SCSIMAPI SequentialComponent : public CircuitComponent
	{
	public:
		const size_t state_size;

		/// <param name="state_size">Size of internal state in 32-bit words</param>
		/// <param name="type">Unique component type index/hash, use typehash(Type) macro in circuit_component_defines.h</param>
		/// <param name="size">Memory size of component, use sizeof(Type)</param>
		/// <param name="align">Memory alignment of component, use alignof(Type)</param>
		SequentialComponent(uint32_t num_inputs, uint32_t num_outputs, size_t state_size, uint32_t type, size_t size, size_t align);

		virtual ~SequentialComponent();

		virtual void reset_state() override;

		virtual void copy_state_host_to_device() override;
		virtual void copy_state_device_to_host() override;

	protected:
		/// <returns>Pointer to internal component state</returns>
		__host__ __device__ uint32_t* state_ptr() const;

		virtual void init_with_circuit(StochasticCircuit* circuit) override;

	private:
		uint32_t fixed_state[SEQUENTIAL_FIXED_STATE_SIZE] = { 0 };
		uint32_t* var_state_host;
		uint32_t* var_state_dev;
		const bool state_is_fixed;

	};

}
