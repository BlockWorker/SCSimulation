#include "cuda_base.cuh"

#include <iostream>
#include <fstream>
#include <filesystem>
#include "StanhTestbench.cuh"
#include "SquarerTestbench.cuh"
#include "InverterTestbench.cuh"
#include "ChainedInverterTestbench.cuh"
#include "MuxNTestbench.cuh"
#include "MLPLayerCountTestbench.cuh"
#include "MLPLayerSizeTestbench.cuh"

constexpr uint32_t SIM_RUNS = 5;

constexpr uint32_t MIN_SN_LENGTH_MLP = 1 << 8;
constexpr uint32_t MIN_SN_LENGTH = 1 << 11;

void runBench(Testbench* bench, const char* dirname, const char* filename) {
	auto csv = bench->run();

	char path[128];
	snprintf(path, 128, "%s\\%s", dirname, filename);

	std::ofstream file;
	file.open(path);
	file << csv;
	file.close();

	delete bench;
}

void run() {
	cu(cudaSetDevice(0));
	char dir[10];

	for (uint32_t i = 0; i < SIM_RUNS; i++) {
		std::cout << "******* BEGIN SIMULATION RUN " << i << " *******" << std::endl;

		snprintf(dir, 10, "results%d", i);

		std::filesystem::create_directory(dir);

		//*
		sim_bitwise = true;
		std::cout << "**** Running bitwise inverter testbench ****" << std::endl;
		runBench(new InverterTestbench(MIN_SN_LENGTH, 14, 10), dir, "inverter_bitwise.csv");
		std::cout << "**** Running bitwise squarer testbench ****" << std::endl;
		runBench(new SquarerTestbench(MIN_SN_LENGTH, 13, 10), dir, "squarer_bitwise.csv");
		//*/
		//*
		sim_bitwise = false;
		std::cout << "**** Running MuxN testbench ****" << std::endl;
		runBench(new MuxNTestbench(MIN_SN_LENGTH, 13, 10), dir, "muxn.csv");
		std::cout << "**** Running inverter testbench ****" << std::endl;
		runBench(new InverterTestbench(MIN_SN_LENGTH, 14, 10), dir, "inverter.csv");
		std::cout << "**** Running squarer testbench ****" << std::endl;
		runBench(new SquarerTestbench(MIN_SN_LENGTH, 13, 10), dir, "squarer.csv");
		std::cout << "**** Running Stanh testbench ****" << std::endl;
		runBench(new StanhTestbench(MIN_SN_LENGTH, 6, 14, 10), dir, "stanh.csv");
		std::cout << "**** Running MLP layer-count testbench ****" << std::endl;
		runBench(new MLPLayerCountTestbench(MIN_SN_LENGTH_MLP, 128, 8, 8), dir, "mlp_lc.csv");
		std::cout << "**** Running MLP layer-size testbench ****" << std::endl;
		runBench(new MLPLayerSizeTestbench(MIN_SN_LENGTH_MLP, 4, 5, 10), dir, "mlp_ls.csv");
		//*/
		//*
		std::cout << "**** Running chained inverter testbench ****" << std::endl;
		runBench(new ChainedInverterTestbench(MIN_SN_LENGTH, 13, 10), dir, "chained_inverter.csv");
		//*/
	}
}

int main() {
	try {
		run();
	} catch (std::exception e) {
		std::cerr << e.what();
	}

	return 0;
}