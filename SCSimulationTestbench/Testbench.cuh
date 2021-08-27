#pragma once

#include <sstream>
#include <chrono>
#include <algorithm>
#include "StochasticCircuitFactory.cuh"
#include "StochasticCircuit.cuh"

using namespace scsim;

constexpr char CSV_SEPARATOR = ';';
constexpr char CSV_DECIMAL_SEP = ',';

class Testbench
{
public:
	const uint32_t num_setups;
	const uint32_t num_iterations;

	Testbench(uint32_t num_setups, uint32_t num_iterations) : num_setups(num_setups), num_iterations(num_iterations), factory(new StochasticCircuitFactory()) {
		circuit = nullptr;
	}

	virtual ~Testbench() {
		delete circuit;
		delete factory;
	}

	std::string run() {
		std::stringstream ss;

		std::cout << "---- Running testbench ----" << std::endl;

		ss << "Setup index" << CSV_SEPARATOR << "Build";
		for (uint32_t i = 0; i < num_iterations; i++) {
			ss << CSV_SEPARATOR << i;
		}
		ss << "\n";

		for (uint32_t setup = 0; setup < num_setups; setup++) {
			delete circuit;

			std::cout << "Building setup " << setup << std::endl;

			factory->reset();
			factory->set_host_only(false);
			build_circuit(setup);

			auto build_start = std::chrono::steady_clock::now();
			circuit = factory->create_circuit();
			auto build_end = std::chrono::steady_clock::now();
			
			ss << setup << CSV_SEPARATOR << std::chrono::duration_cast<std::chrono::microseconds>(build_end - build_start).count() / 1000.0;

			for (uint32_t iteration = 0; iteration < num_iterations; iteration++) {
				std::cout << "Running iteration " << iteration << std::endl;

				circuit->reset_circuit();
				bool host_sim = config_circuit(iteration);

				auto iter_start = std::chrono::steady_clock::now();
				if (host_sim) {
					circuit->simulate_circuit_host_only();
				} else {
					circuit->simulate_circuit();
					cu(cudaDeviceSynchronize());
				}
				auto iter_end = std::chrono::steady_clock::now();

				ss << CSV_SEPARATOR << std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count() / 1000.0;

				post_iter(setup, iteration, ss);
			}

			ss << "\n";
		}

		std::cout << "---- Testbench finished ----" << std::endl;

		auto str = ss.str();
		std::replace(str.begin(), str.end(), '.', CSV_DECIMAL_SEP);

		return str;
	}

protected:
	StochasticCircuit* circuit;
	StochasticCircuitFactory* const factory;

	virtual void build_circuit(uint32_t setup) = 0;

	virtual bool config_circuit(uint32_t iteration) = 0;

	virtual void post_iter(uint32_t setup, uint32_t iteration, std::stringstream& ss) { }

};
