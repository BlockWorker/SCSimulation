#include "cuda_base.cuh"

#include <iostream>
#include <fstream>
#include "StanhTestbench.cuh"
#include "SquarerTestbench.cuh"
#include "InverterTestbench.cuh"
#include "MuxNTestbench.cuh"

constexpr uint32_t SN_LENGTH = 1 << 11;
constexpr uint32_t MAX_PRINT = 1 << 7;

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

	runBench(new MuxNTestbench(SN_LENGTH, 13), "results\\muxn.csv");
	runBench(new InverterTestbench(SN_LENGTH, 14), "results\\inverter.csv");
	runBench(new SquarerTestbench(SN_LENGTH, 14), "results\\squarer.csv");
	runBench(new StanhTestbench(SN_LENGTH, 6, 14), "results\\stanh.csv");
}

int main() {
	try {
		run();
	} catch (std::exception e) {
		std::cerr << e.what();
	}

	return 0;
}