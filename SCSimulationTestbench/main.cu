#include "cuda_base.cuh"

#include <iostream>
#include <fstream>
#include "StanhTestbench.cuh"
#include "SquarerTestbench.cuh"
#include "InverterTestbench.cuh"
#include "MuxNTestbench.cuh"

constexpr uint32_t MIN_SN_LENGTH = 1 << 11;

void runBench(Testbench* bench, const char* filename) {
	auto csv = bench->run();

	std::ofstream file;
	file.open(filename);
	file << csv;
	file.close();

	delete bench;
}

void run() {
	cu(cudaSetDevice(0));

	runBench(new MuxNTestbench(MIN_SN_LENGTH, 13, 10), "results\\muxn.csv");
	runBench(new InverterTestbench(MIN_SN_LENGTH, 14, 10), "results\\inverter.csv");
	runBench(new SquarerTestbench(MIN_SN_LENGTH, 14, 10), "results\\squarer.csv");
	runBench(new StanhTestbench(MIN_SN_LENGTH, 6, 14, 10), "results\\stanh.csv");
}

int main() {
	try {
		run();
	} catch (std::exception e) {
		std::cerr << e.what();
	}

	return 0;
}