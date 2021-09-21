#pragma once

#include <stdint.h>
#include "SequentialComponent.cuh"
#include "dll.h"

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

	protected:
		virtual void link_dev_functions() override;

	private:
		const uint32_t k;

	};

}
