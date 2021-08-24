#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*#include "CircuitComponent.cuh"
#include "ComponentTypes.h"
#include "BasicCombinatorial.cuh"
#include "Stanh.cuh"


__device__ void CircuitComponent::simulate_step_dev() {
	switch (component_type) {
		case COMPONENT_INV:
			((Inverter*)this)->_simulate_step_dev();
			break;
		case COMPONENT_AND:
			((AndGate*)this)->_simulate_step_dev();
			break;
		case COMPONENT_NAND:
			((NandGate*)this)->_simulate_step_dev();
			break;
		case COMPONENT_OR:
			((OrGate*)this)->_simulate_step_dev();
			break;
		case COMPONENT_NOR:
			((NorGate*)this)->_simulate_step_dev();
			break;
		case COMPONENT_XOR:
			((XorGate*)this)->_simulate_step_dev();
			break;
		case COMPONENT_XNOR:
			((XnorGate*)this)->_simulate_step_dev();
			break;
		case COMPONENT_MUX2:
			((Multiplexer2*)this)->_simulate_step_dev();
			break;
		case COMPONENT_MUXN:
			((MultiplexerN*)this)->_simulate_step_dev();
			break;
		case COMPONENT_STANH:
			((Stanh*)this)->_simulate_step_dev();
			break;
	}
}*/
