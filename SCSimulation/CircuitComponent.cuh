#pragma once

#include "cuda_base.cuh"
#include <stdint.h>
#include "dll.h"

/// <summary>
/// Use this macro to add components to a circuit factory.
/// </summary>
/// <param name="factory">Pointer to the factory to add the component to</param>
/// <param name="Type">Component type</param>
/// <param name="__VA_ARGS__">Component constructor arguments (excluding factory)</param>
#define factory_add_component(factory, Type, ...) factory->add_component(new Type(__VA_ARGS__, (StochasticCircuitFactory*)factory))

namespace scsim {

	class StochasticCircuit;
	class StochasticCircuitFactory;

	class SCSIMAPI CircuitComponent
	{
	public:
		const uint32_t component_type;
		const uint32_t num_inputs;
		const uint32_t num_outputs;
		uint32_t* inputs_host;
		uint32_t* outputs_host;

		void (*simulate_step_dev_ptr)(CircuitComponent*); //pointer to type-specific device simulation function
		void (*calc_progress_dev_ptr)(CircuitComponent*); //pointer to type-specific device progress calculation function

		/// <param name="type">Unique component type index/hash, use typehash(Type) macro in circuit_component_defines.h</param>
		/// <param name="size">Memory size of component, use sizeof(Type)</param>
		/// <param name="align">Memory alignment of component, use alignof(Type)</param>
		CircuitComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t type, size_t size, size_t align, StochasticCircuitFactory* factory);

		virtual ~CircuitComponent();

		CircuitComponent(const CircuitComponent& other) = delete;
		CircuitComponent& operator=(const CircuitComponent& other) = delete;
		CircuitComponent(CircuitComponent&& other) = delete;
		CircuitComponent& operator=(CircuitComponent&& other) = delete;

		__host__ __device__ uint32_t current_sim_progress() const {
#ifdef __CUDA_ARCH__
			return *progress_dev_ptr;
#else
			return *progress_host_ptr;
#endif
		}

		__host__ __device__ uint32_t current_sim_progress_word() const {
#ifdef __CUDA_ARCH__
			return *progress_dev_ptr / 32;
#else
			return *progress_host_ptr / 32;
#endif
		}

		__host__ __device__ uint32_t next_sim_progress() const {
#ifdef __CUDA_ARCH__
			return *(progress_dev_ptr + 1);
#else
			return *(progress_host_ptr + 1);
#endif
		}

		__host__ __device__ uint32_t next_sim_progress_word() const {
#ifdef __CUDA_ARCH__
			return (*(progress_dev_ptr + 1) + 31) / 32;
#else
			return (*(progress_host_ptr + 1) + 31) / 32;
#endif
		}

		StochasticCircuit* get_circuit() const;

		virtual void reset_state() = 0;

		virtual void calculate_simulation_progress_host();
		__device__ void calculate_simulation_progress_dev();

		virtual void simulate_step_host() = 0;
		__device__ void simulate_step_dev();

		void sim_step_finished_host();
		__device__ void sim_step_finished_dev();

		static __device__ void _calculate_simulation_progress_dev(CircuitComponent* comp);

	protected:
		friend StochasticCircuitFactory;

		StochasticCircuit* circuit;

		const size_t mem_obj_size;
		const size_t mem_align;
		CircuitComponent* dev_ptr;
		uint32_t* net_values_dev;
		uint32_t* net_progress_dev;
		uint32_t sim_length;

		uint32_t* progress_host_ptr;
		uint32_t* progress_dev_ptr;

		size_t* input_offsets_host;
		size_t* output_offsets_host;
		size_t* input_offsets_dev;
		size_t* output_offsets_dev;

		uint32_t* inputs_dev;
		uint32_t* outputs_dev;

		size_t io_array_offset;

		void calculate_io_offsets(size_t* dev_offset_scratchpad);

		virtual void init_with_circuit(StochasticCircuit* circuit, uint32_t* progress_host_ptr, uint32_t* progress_dev_ptr, size_t* dev_offset_scratchpad);

	private:
		static bool _dev_link_initialized;
		static void (*_simprog_ptr)(CircuitComponent*);

	};

}
