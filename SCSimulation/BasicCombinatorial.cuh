﻿#pragma once

#include <stdint.h>
#include <initializer_list>
#include "CombinatorialComponent.cuh"
#include "circuit_component_defines.cuh"

namespace scsim {

	class SCSIMAPI Inverter : public CombinatorialComponent
	{
	public:
		Inverter(uint32_t input, uint32_t output, StochasticCircuitFactory* factory);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics(Inverter)

	};

	class SCSIMAPI AndGate : public CombinatorialComponent
	{
	public:
		AndGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics(AndGate)

	};

	class SCSIMAPI NandGate : public CombinatorialComponent
	{
	public:
		NandGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics(NandGate)

	};

	class SCSIMAPI OrGate : public CombinatorialComponent
	{
	public:
		OrGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics(OrGate)

	};

	class SCSIMAPI NorGate : public CombinatorialComponent
	{
	public:
		NorGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics(NorGate)

	};

	class SCSIMAPI XorGate : public CombinatorialComponent
	{
	public:
		XorGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics(XorGate)

	};

	class SCSIMAPI XnorGate : public CombinatorialComponent
	{
	public:
		XnorGate(uint32_t input1, uint32_t input2, uint32_t output, StochasticCircuitFactory* factory);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics(XnorGate)

	};

	class SCSIMAPI Multiplexer2 : public CombinatorialComponent
	{
	public:
		Multiplexer2(uint32_t input1, uint32_t input2, uint32_t select, uint32_t output, StochasticCircuitFactory* factory);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics(Multiplexer2)

	};

	/// <summary>
	/// N-input multiplexer
	/// </summary>
	class SCSIMAPI MultiplexerN : public CombinatorialComponent
	{
	public:
		/// <param name="inputs">pointer to array of input net indices</param>
		/// <param name="selects">pointer to array of select net indices, must be sufficiently long for required number of selects (ceil(log2(num_inputs)))</param>
		MultiplexerN(uint32_t _num_inputs, uint32_t* inputs, uint32_t* selects, uint32_t output, StochasticCircuitFactory* factory);

		/// <param name="first_input">first input net index, further inputs assigned consecutive indices</param>
		/// <param name="first_select">first select net index, further selects  assigned consecutive indices, must have sufficient nets available (ceil(log2(num_inputs)))</param>
		MultiplexerN(uint32_t _num_inputs, uint32_t first_input, uint32_t first_select, uint32_t output, StochasticCircuitFactory* factory);

		/// <param name="inputs">list of input net indices</param>
		/// <param name="selects">list of select net indices, must be sufficiently long for required number of selects (ceil(log2(num_inputs)))</param>
		MultiplexerN(std::initializer_list<uint32_t> inputs, std::initializer_list<uint32_t> selects, uint32_t output, StochasticCircuitFactory* factory);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics(MultiplexerN)

	private:
		const uint32_t num_mux_inputs;
		const uint32_t num_selects;

	};

	/// <summary>
	/// Delay component (D-Flip-Flop), delays bit stream by one bit time
	/// </summary>
	class SCSIMAPI Delay : public CombinatorialComponent
	{
	public:
		Delay(uint32_t input, uint32_t output, StochasticCircuitFactory* factory);

		virtual void calculate_simulation_progress_host() override;

		virtual void simulate_step_host() override;

		static __device__ void _calculate_simulation_progress_dev(CircuitComponent* comp);

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

		decl_device_statics(Delay)

	};

}
