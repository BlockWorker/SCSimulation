#pragma once

#include <stdint.h>
#include "CircuitComponent.cuh"
#include "dll.h"

namespace scsim {

	template<size_t state_size>
	class SCSIMAPI SequentialComponent : public CircuitComponent
	{
	public:
		/// <param name="state_size">Size of internal state in 32-bit words</param>
		/// <param name="type">Unique component type index/hash, use typehash(Type) macro in circuit_component_defines.h</param>
		/// <param name="size">Memory size of component, use sizeof(Type)</param>
		/// <param name="align">Memory alignment of component, use alignof(Type)</param>
		SequentialComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t type, size_t size, size_t align) :
			CircuitComponent(num_inputs, num_outputs, type | 0x80000000u, size, align) { //sequential component types have upper bit set -> sequential components always come last once sorted

		}

		virtual void reset_state() override {
			memset(state, 0, state_size * sizeof(uint32_t));
		}

	protected:
		uint32_t state[state_size] = { 0 };

		virtual void init_with_circuit(StochasticCircuit* circuit, uint32_t* progress_host_ptr, uint32_t* progress_dev_ptr, size_t* calc_scratchpad) override {
			reset_state();

			CircuitComponent::init_with_circuit(circuit, progress_host_ptr, progress_dev_ptr, calc_scratchpad);
		}

	};

}
