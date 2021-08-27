#pragma once

#include "cuda_base.cuh"

#define typehash(Type) ((uint32_t)typeid(Type).hash_code())
#define link_device_sim_function(Type) void Type::link_devstep() { scsim::link_devstep_kern<<<1, 1>>>((Type*)dev_ptr); }

#define takebit(x) (((x) & 0x80000000u) > 0)
#define putbit(x, v) ((x) |= (v) ? 1 : 0)

namespace scsim {

	template<class CompType>
	__global__ void link_devstep_kern(CompType* comp) {
		comp->simulate_step_dev_ptr = &comp->_simulate_step_dev;
	}

}
