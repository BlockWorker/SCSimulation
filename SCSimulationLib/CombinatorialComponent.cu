#include "CombinatorialComponent.cuh"

namespace scsim {

	//combinatorial component types do not have upper bit set -> combinatorial components always come first once sorted
	CombinatorialComponent::CombinatorialComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t progress_offset, uint32_t type, size_t size, size_t align, StochasticCircuitFactory* factory) :
		CircuitComponent(num_inputs, num_outputs, progress_offset, type & 0x7fffffffu, size, align, factory) {

	}

}
