#pragma once

#include <stdint.h>
#include "dll.h"

namespace scsim {

	class StochasticCircuit;
	class StochasticCircuitFactory;

	class SCSIMAPI CircuitComponent
	{
	public:
		const uint32_t component_type;
		const uint32_t num_inputs;
		const uint32_t num_outputs;
		uint32_t* inputs;
		uint32_t* outputs;

		void (*simulate_step_dev_ptr)(CircuitComponent*); //pointer to type-specific device simulation function

		/// <param name="type">Unique component type index/hash, use typehash(Type) macro in circuit_component_defines.h</param>
		/// <param name="size">Memory size of component, use sizeof(Type)</param>
		/// <param name="align">Memory alignment of component, use alignof(Type)</param>
		CircuitComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t type, size_t size, size_t align);

		virtual ~CircuitComponent();

		CircuitComponent(const CircuitComponent& other) = delete;
		CircuitComponent& operator=(const CircuitComponent& other) = delete;
		CircuitComponent(CircuitComponent&& other) = delete;
		CircuitComponent& operator=(CircuitComponent&& other) = delete;

		uint32_t current_sim_progress() const;
		uint32_t current_sim_progress_word() const;
		uint32_t next_sim_progress() const;
		uint32_t next_sim_progress_word() const;
		StochasticCircuit* get_circuit() const;

		virtual void reset_state() = 0;

		virtual void copy_state_host_to_device() = 0;
		virtual void copy_state_device_to_host() = 0;

		virtual void calculate_simulation_progress();
		virtual void simulate_step_host() = 0;

		__device__ void simulate_step_dev();

		void sim_step_finished();

	protected:
		friend StochasticCircuitFactory;

		StochasticCircuit* circuit;

		const size_t mem_obj_size;
		const size_t mem_align;
		CircuitComponent* dev_ptr;
		uint32_t* net_values_dev;

		uint32_t current_progress;
		uint32_t current_progress_word;
		uint32_t next_step_progress;
		uint32_t next_step_progress_word;

		size_t* input_offsets_host;
		size_t* output_offsets_host;
		size_t* input_offsets_dev;
		size_t* output_offsets_dev;

		void calculate_io_offsets();

		virtual void init_with_circuit(StochasticCircuit* circuit);

		virtual void link_devstep() = 0;

	};

}
