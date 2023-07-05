#pragma once

#include <stdint.h>
#include "SequentialComponent.cuh"
#include "circuit_component_defines.cuh"

namespace scsim {

	/// <summary>
	/// Linear feedback shift register
	/// </summary>
	class SCSIMAPI LFSR : public SequentialComponent<1>
	{
	public:
		LFSR(uint32_t output, StochasticCircuitFactory* factory);

		virtual void reset_state() override;

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics()

	};

}
