#pragma once

#include <sstream>
#include <chrono>
#include <algorithm>
#include "StochasticCircuitFactory.cuh"
#include "StochasticCircuit.cuh"

using namespace scsim;

constexpr char CSV_SEPARATOR = ';'; //separator for CSV values
constexpr char CSV_DECIMAL_SEP = ','; //CSV decimal point/separator

/// <summary>
/// Simulates component setups in different conditions and outputs results in CSV format
/// </summary>
class Testbench
{
public:
	const uint32_t num_setups;
	const uint32_t num_iterations;

	/// <param name="num_setups">Number of different circuit setups to test</param>
	/// <param name="num_iterations">Number of iterations (with potentially different parameters) to run per setup</param>
	Testbench(uint32_t num_setups, uint32_t num_iterations) : num_setups(num_setups), num_iterations(num_iterations), factory(new StochasticCircuitFactory()) {
		circuit = nullptr;
	}

	virtual ~Testbench() {
		delete circuit;
		delete factory;
	}

	Testbench(const Testbench& other) = delete;
	Testbench& operator=(const Testbench& other) = delete;
	Testbench(Testbench&& other) = delete;
	Testbench& operator=(Testbench&& other) = delete;

	std::string run() {
		std::stringstream ss;

		std::cout << "---- Running testbench ----" << std::endl;

		ss << "Setup index" << CSV_SEPARATOR << "Build";
		for (uint32_t i = 0; i < num_iterations; i++) { //write iteration indices to title row
			ss << CSV_SEPARATOR << i;
		}
		write_additional_column_titles(ss); //write additional column titles if necessary
		ss << "\n";

		for (uint32_t setup = 0; setup < num_setups; setup++) { //iterate over setups
			delete circuit;

			std::cout << "Building setup " << setup << std::endl;

			factory->reset();
			factory->set_host_only(false);
			uint32_t desired_iters = build_circuit(setup); //build circuit setup
			uint32_t setup_iters = __min(desired_iters, num_iterations);

			//create circuit and measure time taken
			auto build_start = std::chrono::steady_clock::now();
			circuit = factory->create_circuit();
			auto build_end = std::chrono::steady_clock::now();
			
			ss << setup << CSV_SEPARATOR << std::chrono::duration_cast<std::chrono::microseconds>(build_end - build_start).count() / 1000.0;

			for (uint32_t iteration = 0; iteration < setup_iters; iteration++) { //iterate over iterations
				std::cout << "Running iteration " << iteration << std::endl;

				circuit->reset_circuit();
				bool host_sim = config_circuit(setup, iteration); //configure circuit for current iteration

				//simulate circuit and measure time taken
				auto iter_start = std::chrono::steady_clock::now();
				if (host_sim) {
					circuit->simulate_circuit_host_only();
				} else {
					circuit->simulate_circuit();
					cu(cudaDeviceSynchronize());
				}
				auto iter_end = std::chrono::steady_clock::now();

				ss << CSV_SEPARATOR << std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count() / 1000.0;

				post_iter(setup, iteration); //run post-iteration tasks if required
			}

			for (uint32_t i = setup_iters; i < num_iterations; i++) ss << CSV_SEPARATOR; //pad columns for missing iterations

			post_setup(setup, ss); //run post-setup tasks if required (may write additional column values)

			ss << "\n";
		}

		std::cout << "---- Testbench finished ----" << std::endl;

		auto str = ss.str();
		std::replace(str.begin(), str.end(), '.', CSV_DECIMAL_SEP); //correct decimal point/separator

		return str;
	}

protected:
	StochasticCircuit* circuit;
	StochasticCircuitFactory* const factory;

	/// <summary>
	/// Write additional CSV column titles if required, each preceded by CSV_SEPARATOR
	/// </summary>
	virtual void write_additional_column_titles(std::stringstream& ss) { }

	/// <summary>
	/// Build circuit for given setup index, use "factory" variable
	/// </summary>
	/// <returns>How many iterations this setup should run</returns>
	virtual uint32_t build_circuit(uint32_t setup) = 0;

	/// <summary>
	/// Configure circuit for given iteration index, use "circuit" variable
	/// </summary>
	/// <returns>Whether this iteration should be simulated on the host instead of the device</returns>
	virtual bool config_circuit(uint32_t setup, uint32_t iteration) = 0;

	/// <summary>
	/// Run post-iteration tasks/calculations if required
	/// </summary>
	virtual void post_iter(uint32_t setup, uint32_t iteration) { }

	/// <summary>
	/// Run post-setup tasks/calculations if required, may append additional column values, each preceded by CSV_SEPARATOR
	/// </summary>
	virtual void post_setup(uint32_t setup, std::stringstream& ss) { }

};
