#include "cuda_base.cuh"

#include <iostream>
#include <chrono>

#include "StochasticNumber.cuh"
#include "StochasticCircuitFactory.cuh"
#include "StochasticCircuit.cuh"
#include "BasicCombinatorial.cuh"
#include "Stanh.cuh"

constexpr uint32_t SN_LENGTH = 1 << 16;
constexpr uint32_t MAX_PRINT = 1 << 6;

/*
__global__ void test(size_t* sizes) {
	sizes[0] = sizeof(AndGate);
	sizes[1] = offsetof(AndGate, component_type);
}//*/

int main_old() {

	cu(cudaSetDevice(0));

	/*
	size_t sizes_h[2];
	size_t* sizes_d;
	cudaMalloc(&sizes_d, 2 * sizeof(size_t));
	test<<<1, 1>>>(sizes_d);
	cudaMemcpy(sizes_h, sizes_d, 2 * sizeof(size_t), cudaMemcpyDeviceToHost);
	printf("Host: Size %d Offset %d\n", sizeof(AndGate), offsetof(AndGate, component_type));
	printf("Device: Size %d Offset %d\n", sizes_h[0], sizes_h[1]);//*/

	/*
	auto n1 = StochasticNumber::generate_unipolar(SN_LENGTH, 0.2);
	auto n2 = StochasticNumber::generate_unipolar(SN_LENGTH, 0.4);
	auto n3 = StochasticNumber::generate_unipolar(SN_LENGTH, 0.6);
	auto n4 = StochasticNumber::generate_unipolar(SN_LENGTH, 0.8);
	auto s1 = StochasticNumber::generate_unipolar(SN_LENGTH, 0.5);
	auto s2 = StochasticNumber::generate_unipolar(SN_LENGTH, 0.5);

	n1->print_unipolar(MAX_PRINT);
	n2->print_unipolar(MAX_PRINT);
	n3->print_unipolar(MAX_PRINT);
	n4->print_unipolar(MAX_PRINT);
	std::cout << std::endl;
	s1->print_unipolar(MAX_PRINT);
	s2->print_unipolar(MAX_PRINT);

	auto factory = new StochasticCircuitFactory();
	factory->set_host_only(false);
	factory->set_sim_length(SN_LENGTH);
	auto in1 = factory->add_net();
	auto in2 = factory->add_net();
	auto in3 = factory->add_net();
	auto in4 = factory->add_net();
	auto sel1 = factory->add_net();
	auto sel2 = factory->add_net();
	auto out = factory->add_net();
	factory->add_component(new MultiplexerN({ in1, in2, in3, in4 }, { sel1, sel2 }, out));
	//factory->add_component(new AndGate(in1, in4, out));

	auto circuit = factory->create_circuit();

	circuit->set_net_value(in1, n1);
	circuit->set_net_value(in2, n2);
	circuit->set_net_value(in3, n3);
	circuit->set_net_value(in4, n4);
	circuit->set_net_value(sel1, s1);
	circuit->set_net_value(sel2, s2);
	circuit->simulate_circuit();

	auto no = circuit->get_net_value(out);
	std::cout << std::endl;
	no->print_bipolar(MAX_PRINT);
	/*/
	auto factory = new StochasticCircuitFactory();
	factory->set_host_only(false);
	factory->set_sim_length(SN_LENGTH);

	constexpr uint32_t sample_count = 1000;

	uint32_t in[sample_count + 1], out[sample_count + 1];
	for (uint32_t i = 0; i <= sample_count; i++) {
		in[i] = factory->add_net();
		out[i] = factory->add_net();
		factory->add_component(new Stanh(in[i], out[i], 6));
	}

	auto beging = std::chrono::steady_clock::now();
	auto circuit = factory->create_circuit();
	auto endg = std::chrono::steady_clock::now();

	std::cout << "Circuit creation: " << std::chrono::duration_cast<std::chrono::milliseconds>(endg - beging).count() << " ms" << std::endl;

	auto numbers = (StochasticNumber**)malloc((sample_count + 1) * sizeof(StochasticNumber*));
	auto vals = (double*)malloc((sample_count + 1) * sizeof(double));
	for (uint32_t i = 0; i <= sample_count; i++) {
		vals[i] = (double)i / (double)sample_count;
	}

	StochasticNumber::generate_multiple_curand(numbers, SN_LENGTH, vals, sample_count + 1);

	for (uint32_t i = 0; i <= sample_count; i++) {
		circuit->set_net_value(in[i], numbers[i]);
	}

	auto beginh = std::chrono::steady_clock::now();
	circuit->simulate_circuit_host_only();
	auto endh = std::chrono::steady_clock::now();

	std::cout << "Host: " << std::chrono::duration_cast<std::chrono::milliseconds>(endh - beginh).count() << " ms" << std::endl;

	circuit->reset_circuit();

	for (uint32_t i = 0; i <= sample_count; i++) {
		circuit->set_net_value(in[i], numbers[i]);
	}

	auto begind = std::chrono::steady_clock::now();
	circuit->simulate_circuit();
	auto endd = std::chrono::steady_clock::now();

	std::cout << "Device: " << std::chrono::duration_cast<std::chrono::milliseconds>(endd - begind).count() << " ms" << std::endl;

	for (uint32_t i = 0; i <= sample_count; i++) {
		std::cout << circuit->get_net_value_bipolar(out[i]) << ",";
	}
	//*/

	return 0;
}