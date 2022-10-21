#pragma once

#include <stdint.h>
#include "SequentialComponent.cuh"
#include "circuit_component_defines.cuh"

namespace scsim {

	//Stochastic tanh function, following parallel counter (Kim et al., 2016)
	class SCSIMAPI Btanh : public SequentialComponent<1>
	{
	public:
		/// <param name="n">As in paper: number of parallel counter inputs</param>
		/// <param name="r">As in paper: number of states, must be multiple of 2</param>
		/// <param name="inputs">pointer to array of input net indices (from parallel counter), LSB first, must be correspondingly long (floor(log2(n)) + 1)</param>
		Btanh(uint32_t n, uint32_t r, uint32_t* inputs, uint32_t output, StochasticCircuitFactory* factory);

		/// <param name="n">As in paper: number of parallel counter inputs</param>
		/// <param name="r">As in paper: number of states, must be multiple of 2</param>
		/// <param name="inputs">first input net index (LSB from parallel counter), further outputs assigned consecutive indices, must be correspondingly available (floor(log2(n)) + 1)</param>
		Btanh(uint32_t n, uint32_t r, uint32_t first_input, uint32_t output, StochasticCircuitFactory* factory);

		/// <param name="n">As in paper: number of parallel counter inputs</param>
		/// <param name="r">As in paper: number of states, must be multiple of 2</param>
		/// <param name="inputs">list of input net indices (from parallel counter), LSB first, must be correspondingly long (floor(log2(n)) + 1)</param>
		Btanh(uint32_t n, uint32_t r, std::initializer_list<uint32_t> inputs, uint32_t output, StochasticCircuitFactory* factory);

		virtual void reset_state() override;

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		/// <param name="n">Number of parallel counter inputs</param>
		/// <param name="s">Desired value of s where Btanh(n, r, t) ~= tanh(t/s), as in paper</param>
		/// <returns>Optimal value of the parameter r for the given values</returns>
		static uint32_t calculate_r(uint32_t n, double s);

		decl_device_statics()

	private:
		const uint32_t n;
		const uint32_t input_width;
		const uint32_t s_max;
		const uint32_t s_half;

	};

}
