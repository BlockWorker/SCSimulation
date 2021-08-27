#include "CombinatorialComponent.cuh"

namespace scsim {

	CombinatorialComponent::CombinatorialComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t type, size_t size, size_t align) :
		CircuitComponent(num_inputs, num_outputs, type & 0x7fffffffu, size, align) {

	}

	CombinatorialComponent::~CombinatorialComponent() {

	}

}
