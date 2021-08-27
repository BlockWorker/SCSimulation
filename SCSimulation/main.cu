#include "cuda_base.cuh"

#include <iostream>
#include <fstream>
#include "StanhTestbench.cuh"
#include "SquarerTestbench.cuh"
#include "InverterTestbench.cuh"
#include "MuxNTestbench.cuh"
#include "Delay.cuh"

constexpr uint32_t SN_LENGTH = 1 << 11;
constexpr uint32_t MAX_PRINT = 1 << 7;

void run() {
	cu(cudaSetDevice(0));

	/*
	auto n1 = StochasticNumber::generate_unipolar(SN_LENGTH, 0.5);

	n1->print_unipolar(MAX_PRINT);

	auto factory = new StochasticCircuitFactory();
	factory->set_host_only(false);
	factory->set_sim_length(SN_LENGTH);
	auto in1 = factory->add_net();
	auto out = factory->add_net();
	factory->add_component(new Delay(in1, out));

	auto circuit = factory->create_circuit();

	circuit->set_net_value(in1, n1);
	circuit->simulate_circuit();

	auto no = circuit->get_net_value(out);
	no->print_unipolar(MAX_PRINT);
	/*/
	//auto bench = new MuxNTestbench(SN_LENGTH, 13);
	//auto bench = new InverterTestbench(SN_LENGTH, 14);
	auto bench = new SquarerTestbench(SN_LENGTH, 14);
	//auto bench = new StanhTestbench(SN_LENGTH, 6, 14);

	auto csv = bench->run();

	std::ofstream file;
	//file.open("muxn.csv");
	//file.open("inverter.csv");
	file.open("squarer.csv");
	//file.open("stanh.csv");
	file << csv;
	file.close();

	delete bench;
	//*/
}

int main() {
	try {
		run();
	} catch (std::exception e) {
		std::cerr << e.what();
	}

	return 0;
}