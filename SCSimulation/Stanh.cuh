#pragma once

#include <stdint.h>
#include "SequentialComponent.cuh"
#include "circuit_component_defines.cuh"

namespace scsim {

	//Stochastic tanh function
	class SCSIMAPI Stanh : public SequentialComponent<1>
	{
	public:
		/// <param name="k">Number of states: output ~= tanh((k/2) * input)</param>
		Stanh(uint32_t input, uint32_t output, uint32_t k, StochasticCircuitFactory* factory);

		virtual void reset_state() override;

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics(Stanh)

	private:
		const uint32_t k;

	};

}
