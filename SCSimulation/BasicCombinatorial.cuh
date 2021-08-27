#pragma once

#include <stdint.h>
#include <initializer_list>
#include "CombinatorialComponent.cuh"
#include "dll.h"

namespace scsim {

	class SCSIMAPI Inverter : public CombinatorialComponent
	{
	public:
		Inverter(uint32_t input, uint32_t output);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

	protected:
		virtual void link_devstep() override;

	};

	class SCSIMAPI AndGate : public CombinatorialComponent
	{
	public:
		AndGate(uint32_t input1, uint32_t input2, uint32_t output);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

	protected:
		virtual void link_devstep() override;

	};

	class SCSIMAPI NandGate : public CombinatorialComponent
	{
	public:
		NandGate(uint32_t input1, uint32_t input2, uint32_t output);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

	protected:
		virtual void link_devstep() override;

	};

	class SCSIMAPI OrGate : public CombinatorialComponent
	{
	public:
		OrGate(uint32_t input1, uint32_t input2, uint32_t output);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

	protected:
		virtual void link_devstep() override;

	};

	class SCSIMAPI NorGate : public CombinatorialComponent
	{
	public:
		NorGate(uint32_t input1, uint32_t input2, uint32_t output);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

	protected:
		virtual void link_devstep() override;

	};

	class SCSIMAPI XorGate : public CombinatorialComponent
	{
	public:
		XorGate(uint32_t input1, uint32_t input2, uint32_t output);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

	protected:
		virtual void link_devstep() override;

	};

	class SCSIMAPI XnorGate : public CombinatorialComponent
	{
	public:
		XnorGate(uint32_t input1, uint32_t input2, uint32_t output);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

	protected:
		virtual void link_devstep() override;

	};

	class SCSIMAPI Multiplexer2 : public CombinatorialComponent
	{
	public:
		Multiplexer2(uint32_t input1, uint32_t input2, uint32_t select, uint32_t output);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

	protected:
		virtual void link_devstep() override;

	};

	class SCSIMAPI MultiplexerN : public CombinatorialComponent
	{
	public:
		MultiplexerN(uint32_t num_inputs, uint32_t* inputs, uint32_t* selects, uint32_t output);
		MultiplexerN(uint32_t num_inputs, uint32_t first_input, uint32_t first_select, uint32_t output);
		MultiplexerN(std::initializer_list<uint32_t> inputs, std::initializer_list<uint32_t> selects, uint32_t output);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

	protected:
		virtual void link_devstep() override;

	private:
		const uint32_t num_mux_inputs;
		const uint32_t num_selects;

	};

	class SCSIMAPI Delay : public CombinatorialComponent
	{
	public:
		Delay(uint32_t input, uint32_t output);

		virtual void simulate_step_host() override;

		static __device__ void _simulate_step_dev(CircuitComponent* comp);

	protected:
		virtual void link_devstep() override;

	};

}
