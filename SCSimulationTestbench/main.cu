﻿#include "cuda_base.cuh"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>

#include "StanhTestbench.cuh"
#include "SquarerTestbench.cuh"
#include "InverterTestbench.cuh"
#include "ChainedInverterTestbench.cuh"
#include "MuxNTestbench.cuh"
#include "MLPLayerCountTestbench.cuh"
#include "MLPLayerSizeTestbench.cuh"
#include "CycleTestbench.cuh"

//#include "Graph.h"

constexpr uint32_t SIM_RUNS = 5;

constexpr uint32_t MIN_SN_LENGTH_MLP = 1 << 8;
constexpr uint32_t MIN_SN_LENGTH = 1 << 11;

constexpr uint64_t RESERVED_MEM = 1 << 29;
constexpr uint64_t MEM_PER_COMP = 512;

uint32_t max_setups(uint64_t max_mem, uint32_t comp_per_unit, uint32_t nets_per_unit, uint32_t min_bits) {
	uint32_t min_sim_len_words = (min_bits + 31) / 32;
	if (min_sim_len_words <= 8) min_sim_len_words = 8;
	else if (min_sim_len_words <= 16) min_sim_len_words = 16;
	else min_sim_len_words = ((min_sim_len_words + 31) / 32) * 32;
	uint64_t unit_bytes = MEM_PER_COMP * comp_per_unit + sizeof(uint32_t) * (min_sim_len_words + 1) * nets_per_unit;
	uint64_t max_units = max_mem / unit_bytes;
	return (uint32_t)floor(log2(max_units)) - 3;
}

std::pair<uint32_t, uint32_t> max_setups_nn(uint64_t max_mem, uint32_t layercount, uint32_t min_bits) {
	uint32_t min_sim_len_words = (min_bits + 31) / 32;
	if (min_sim_len_words <= 8) min_sim_len_words = 8;
	else if (min_sim_len_words <= 16) min_sim_len_words = 16;
	else min_sim_len_words = ((min_sim_len_words + 31) / 32) * 32;
	uint32_t max_setups = 0;
	uint64_t comp_bytes = 0;
	uint64_t net_bytes = 0;
	while (max_setups < 10) {
		max_setups++;
		uint32_t setup_layersize = 16 * (1 << (max_setups - 1));
		uint32_t setup_components = layercount * setup_layersize * (setup_layersize + 2);
		uint32_t setup_nets = setup_layersize + layercount * ((uint32_t)ceil(log2(setup_layersize)) + setup_layersize * (2 * setup_layersize + 2));
		uint64_t setup_bytes = MEM_PER_COMP * setup_components + sizeof(uint32_t) * (min_sim_len_words + 1) * setup_nets;
		if (setup_bytes > max_mem) {
			max_setups--;
			break;
		}
		comp_bytes = MEM_PER_COMP * setup_components;
		net_bytes = sizeof(uint32_t) * min_sim_len_words * setup_nets;
	}

	uint64_t max_lastsetup_simlen_factor = (max_mem - comp_bytes) / net_bytes;
	uint32_t max_lastsetup_iters = (uint32_t)floor(log2(max_lastsetup_simlen_factor)) + 1;
	uint32_t max_iters = max_lastsetup_iters + max_setups - 1;

	return std::make_pair(max_setups, max_iters);
}

std::pair<uint32_t, uint32_t> max_layers_nn(uint64_t max_mem, uint32_t layersize, uint32_t min_bits) {
	uint32_t min_sim_len_words = (min_bits + 31) / 32;
	if (min_sim_len_words <= 8) min_sim_len_words = 8;
	else if (min_sim_len_words <= 16) min_sim_len_words = 16;
	else min_sim_len_words = ((min_sim_len_words + 31) / 32) * 32;
	uint32_t layer_components = layersize * (layersize + 2);
	uint32_t layer_nets = layersize * (2 * layersize + 2) + (uint32_t)ceil(log2(layersize));
	uint32_t max_layers = 0;
	uint64_t comp_bytes = 0;
	uint64_t net_bytes = 0;
	while (max_layers <= 32) {
		max_layers++;
		uint32_t setup_components = max_layers * layer_components;
		uint32_t setup_nets = layersize + max_layers * layer_nets;
		uint64_t setup_bytes = MEM_PER_COMP * setup_components + sizeof(uint32_t) * (min_sim_len_words + 1) * setup_nets;
		if (setup_bytes > max_mem) {
			max_layers--;
			break;
		}
		comp_bytes = MEM_PER_COMP * setup_components;
		net_bytes = sizeof(uint32_t) * min_sim_len_words * setup_nets;
	}

	uint64_t max_simlen_factor = (max_mem - comp_bytes) / net_bytes;
	uint32_t max_iters = (uint32_t)floor(log2(max_simlen_factor)) + 1;

	return std::make_pair(max_layers, max_iters);
}

void runBench(Testbench* bench, const char* dirname, const char* filename) {
	auto csv = bench->run();

	char path[128];
	snprintf(path, 128, "%s%c%s", dirname, std::filesystem::path::preferred_separator, filename);

	std::ofstream file;
	file.open(path);
	file << csv;
	file.close();

	delete bench;
}

void run(uint64_t max_mem) {
	char dir[10];

	uint32_t max_setups_1c2n = __min(max_setups(max_mem, 1, 2, MIN_SN_LENGTH) - 1, 18);
	uint32_t max_setups_2c2n = __min(max_setups(max_mem, 2, 2, MIN_SN_LENGTH) - 1, 17);
	uint32_t max_setups_1c9n = __min(max_setups(max_mem, 1, 9, MIN_SN_LENGTH) - 1, 17);
	auto nn_size_maximums = max_setups_nn(max_mem, 4, MIN_SN_LENGTH_MLP);
	auto nn_count_maximums = max_layers_nn(max_mem, 128, MIN_SN_LENGTH_MLP);
	uint32_t nn_size_max_setups = __min(nn_size_maximums.first - 1, 7);
	uint32_t nn_count_max_setups = __min(nn_count_maximums.first / 2, 16);
	uint32_t nn_count_max_iters = __min(nn_count_maximums.second + 1, 10);

	for (uint32_t i = 0; i < SIM_RUNS; i++) {
		std::cout << "******* BEGIN SIMULATION RUN " << i << " *******" << std::endl;

		snprintf(dir, 10, "results%d", i);

		std::filesystem::create_directory(dir);

		/*
		sim_bitwise = true;
		std::cout << "**** Running bitwise inverter testbench ****" << std::endl;
		runBench(new InverterTestbench(MIN_SN_LENGTH, max_setups_1c2n, 10), dir, "inverter_bitwise.csv");
		std::cout << "**** Running bitwise squarer testbench ****" << std::endl;
		runBench(new SquarerTestbench(MIN_SN_LENGTH, max_setups_2c2n, 10), dir, "squarer_bitwise.csv");
		//*/
		//*
		sim_bitwise = false;
		std::cout << "**** Running MuxN testbench ****" << std::endl;
		runBench(new MuxNTestbench(MIN_SN_LENGTH, max_setups_1c9n, 10), dir, "muxn.csv");
		std::cout << "**** Running inverter testbench ****" << std::endl;
		runBench(new InverterTestbench(MIN_SN_LENGTH, max_setups_1c2n, 10), dir, "inverter.csv");
		std::cout << "**** Running squarer testbench ****" << std::endl;
		runBench(new SquarerTestbench(MIN_SN_LENGTH, max_setups_2c2n, 10), dir, "squarer.csv");
		std::cout << "**** Running Stanh testbench ****" << std::endl;
		runBench(new StanhTestbench(MIN_SN_LENGTH, 6, max_setups_1c2n, 10), dir, "stanh.csv");
		std::cout << "**** Running MLP layer-count testbench ****" << std::endl;
		runBench(new MLPLayerCountTestbench(MIN_SN_LENGTH_MLP, 128, nn_count_max_setups, nn_count_max_iters), dir, "mlp_lc.csv");
		std::cout << "**** Running MLP layer-size testbench ****" << std::endl;
		runBench(new MLPLayerSizeTestbench(MIN_SN_LENGTH_MLP, 4, nn_size_max_setups, 10), dir, "mlp_ls.csv");
		//*/
		//*
		std::cout << "**** Running chained inverter testbench ****" << std::endl;
		runBench(new ChainedInverterTestbench(MIN_SN_LENGTH, 13, 10), dir, "chained_inverter.csv");
		//*/
	}
	/*
	std::cout << "**** Running cycle testbench ****" << std::endl;
	runBench(new CycleTestbench(MIN_SN_LENGTH, max_setups_2c2n - 1, 7), "results0", "cycle.csv");
	//*/
	std::cout << std::endl << "******** SIMULATION SUCCESSFULLY FINISHED ********" << std::endl;
}

int main() {
	//*
	try {
		cu(cudaSetDevice(0));
		cudaDeviceProp prop;
		cu(cudaGetDeviceProperties(&prop, 0));
		uint64_t mem_bytes = prop.totalGlobalMem;
		uint64_t sim_max_mem = mem_bytes - RESERVED_MEM;
		std::cout << "Detected device with " << (mem_bytes / (double)(1 << 30)) << " GB of memory" << std::endl;

		std::this_thread::sleep_for(std::chrono::seconds(5));

		run(sim_max_mem);
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
	}
	/*/
	scsim::Graph g(7);
	g.add_edge(4, 1);
	g.add_edge(4, 3);
	g.add_edge(1, 2);
	g.add_edge(3, 2);
	g.add_edge(2, 0);
	g.add_edge(6, 0);
	g.add_edge(5, 3);
	g.add_edge(5, 2);
	g.add_edge(1, 3);

	std::vector<uint32_t> topsort1;
	auto res1 = g.topological_sort(topsort1);

	std::vector<int64_t> sssp_distances;
	std::vector<int64_t> sslp_distances;
	g.topological_sssp(topsort1, -1, sssp_distances);
	g.topological_sslp(topsort1, -1, sslp_distances);

	std::vector<int64_t> sdsp_distances;
	std::vector<int64_t> sdlp_distances;
	g.topological_sdsp(topsort1, -1, sdsp_distances);
	g.topological_sdlp(topsort1, -1, sdlp_distances);

	std::cout << "First: " << res1 << std::endl;
	std::cout << "Order: ";
	for each (auto v in topsort1) std::cout << v << ", ";
	std::cout << std::endl << std::endl;

	std::cout << "SSSP distances: ";
	for each (auto d in sssp_distances) std::cout << d << ", ";
	std::cout << std::endl << std::endl;

	std::cout << "SSLP distances: ";
	for each (auto d in sslp_distances) std::cout << d << ", ";
	std::cout << std::endl << std::endl;

	std::cout << "SDSP distances: ";
	for each (auto d in sdsp_distances) std::cout << d << ", ";
	std::cout << std::endl << std::endl;

	std::cout << "SDLP distances: ";
	for each (auto d in sdlp_distances) std::cout << d << ", ";
	std::cout << std::endl << std::endl;
	//*/

	std::cout << "Press return to exit...";
	std::cin.get();

	return 0;
}
