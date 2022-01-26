#pragma once

#include "cuda_base.cuh"
#include "StochasticCircuitFactory.cuh"
#include "StochasticCircuit.cuh"
#include "library_export.h"
#include <typeinfo>

#define COMP_IMPEXP_SPEC

//macro to generate unique type hash
#define typehash(Type) ((uint32_t)typeid(Type).hash_code())

//internal macro, do not use directly
#define _link_device_steponly_internal(StepType) do { \
		if (!_dev_link_initialized && !factory->host_only) { \
			_dev_link_initialized = true; \
			void (*pointer)(CircuitComponent*); \
			void (**dev_pointer)(CircuitComponent*); \
			cu(cudaMalloc(&dev_pointer, sizeof(void (*)(CircuitComponent*)))); \
			scsim::link_devfunc_kern_steponly<StepType, CircuitComponent><<<1, 1>>>(dev_pointer); \
			cu_kernel_errcheck_nosync(); \
			cu(cudaMemcpy(&pointer, dev_pointer, sizeof(void (*)(CircuitComponent*)), cudaMemcpyDeviceToHost)); \
			_devstep_ptr = *pointer; \
		} \
		simulate_step_dev_ptr = _devstep_ptr; \
	} while (false)

//internal macro, do not use directly
#define _link_device_both_internal(StepType, SimprogType) do { \
		if (!_dev_link_initialized && !factory->host_only) { \
			_dev_link_initialized = true; \
			void (*pointers[2])(CircuitComponent*); \
			void (**dev_pointers)(CircuitComponent*); \
			cu(cudaMalloc(&dev_pointers, 2 * sizeof(void (*)(CircuitComponent*)))); \
			scsim::link_devfunc_kern_both<StepType, SimprogType><<<1, 1>>>(dev_pointers); \
			cu_kernel_errcheck_nosync(); \
			cu(cudaMemcpy(pointers, dev_pointers, 2 * sizeof(void (*)(CircuitComponent*)), cudaMemcpyDeviceToHost)); \
			_devstep_ptr = pointers[0]; \
			_simprog_ptr = pointers[1]; \
		} \
		simulate_step_dev_ptr = _devstep_ptr; \
		calc_progress_dev_ptr = _simprog_ptr; \
	} while (false)

//Macro to link device simulation function, place at the end of each constructor of the component class
#define link_device_sim_function(CType) _link_device_steponly_internal(CType)

//Macro to link device simulation and progress calculation functions, place at the end of each constructor of the component class
#define link_device_sim_progress_functions(CType) _link_device_both_internal(CType, CType)

//Macro to declare static variables for device function linking, place at the end of the component class
#define decl_device_statics() \
	private: \
		static bool _dev_link_initialized; \
		static void (*_devstep_ptr)(CircuitComponent*); \
		static void (*_simprog_ptr)(CircuitComponent*);

//Macro to define static variables for device function linking, place in component class source file (not the header!)
#define def_device_statics(Type) \
	bool COMP_IMPEXP_SPEC Type::_dev_link_initialized = false; \
	void COMP_IMPEXP_SPEC (*Type::_devstep_ptr)(CircuitComponent*) = nullptr; \
	void COMP_IMPEXP_SPEC (*Type::_simprog_ptr)(CircuitComponent*) = nullptr;

//Macro to "take" bits from a word (queue-like behaviour), requires separate left-shift afterwards
#define takebit(x) (((x) & 0x80000000u) > 0)

//Macro to "put" bits into a word (queue-like behaviour), requires separate left-shift afterwards
#define putbit(x, v) ((x) |= (v) ? 1 : 0)

namespace scsim {

	class CircuitComponent;
	
	template<class StepType, class _AddType>
	//Device function linking kernel, do not use directly, use macros above instead
	__global__ void link_devfunc_kern_steponly(void (**func_pointer)(CircuitComponent*)) {
		*func_pointer = &StepType::_simulate_step_dev;
	}

	template<class StepType, class SimprogType>
	//Device function linking kernel, do not use directly, use macros above instead
	__global__ void link_devfunc_kern_both(void (**func_pointers)(CircuitComponent*)) {
		func_pointers[0] = &StepType::_simulate_step_dev;
		func_pointers[1] = &SimprogType::_calculate_simulation_progress_dev;
	}

}
