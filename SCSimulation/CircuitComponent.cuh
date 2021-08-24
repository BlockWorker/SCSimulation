#pragma once

#include "cuda_base.cuh"
#include <stdint.h>
#include <utility>
#include <typeinfo>

#define typehash(Type) ((uint32_t)typeid(Type).hash_code())
#define link_device_sim_function(Type) virtual void link_devstep() override { link_devstep_kern<<<1, 1>>>((Type*)dev_ptr); }

class StochasticCircuit;
class StochasticCircuitFactory;

template<class CompType>
__global__ void link_devstep_kern(CompType* comp) {
	comp->simulate_step_dev_ptr = &comp->_simulate_step_dev;
}

class CircuitComponent
{
public:
	const uint32_t component_type;
	const uint32_t num_inputs;
	const uint32_t num_outputs;
	uint32_t* inputs;
	uint32_t* outputs;

	void (*simulate_step_dev_ptr)(CircuitComponent*);

	CircuitComponent(uint32_t num_inputs, uint32_t num_outputs, uint32_t type, size_t size, size_t align);
	virtual ~CircuitComponent();

	uint32_t current_sim_progress() const;
	uint32_t current_sim_progress_word() const;
	uint32_t next_sim_progress() const;
	uint32_t next_sim_progress_word() const;
	StochasticCircuit* get_circuit() const;

	virtual void reset_state() = 0;

	virtual void copy_state_host_to_device() = 0;
	virtual void copy_state_device_to_host() = 0;

	void calculate_simulation_progress();
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