#pragma once

#include <sstream>
#include <chrono>
#include <algorithm>
#include <iostream>
#include "StochasticCircuitFactory.cuh"
#include "StochasticCircuit.cuh"
#include "AsapScheduler.cuh"

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
	Testbench(uint32_t num_setups, uint32_t num_iterations) : num_setups(num_setups), num_iterations(num_iterations), factory(StochasticCircuitFactory(false)) {
		circuit = nullptr;
	}

	virtual ~Testbench() {
		delete circuit;
	}

	Testbench(const Testbench& other) = delete;
	Testbench& operator=(const Testbench& other) = delete;
	Testbench(Testbench&& other) = delete;
	Testbench& operator=(Testbench&& other) = delete;

	std::string run() {
		std::stringstream ss;

		std::cout << "---- Running testbench ----" << std::endl;

		ss << "Setup index" << CSV_SEPARATOR << "Scheduler" << CSV_SEPARATOR << "Build time" << CSV_SEPARATOR << "Schedule time";
		for (uint32_t i = 0; i < num_iterations; i++) { //write iteration indices to title row
			ss << CSV_SEPARATOR << i << "H" << CSV_SEPARATOR << i << "DT" << CSV_SEPARATOR << i << "DC";
		}
		write_additional_column_titles(ss); //write additional column titles if necessary
		ss << "\n";

		try {
			for (uint32_t setup = 0; setup < num_setups; setup++) { //iterate over setups
				delete circuit;
				circuit = nullptr;

				std::cout << "Building setup " << setup << std::endl;

				factory.reset();

				//create circuit and measure time taken
				auto build_start = std::chrono::steady_clock::now();

				uint32_t desired_iters = build_circuit(setup); //build circuit setup
				circuit = factory.create_circuit();

				auto build_end = std::chrono::steady_clock::now();

				uint32_t setup_iters = __min(desired_iters, num_iterations);

				for (uint32_t scheduler = 0; scheduler < 2; scheduler++) {
					int64_t scheduling_duration;
					std::chrono::steady_clock::time_point schedule_start, schedule_end;
					char* scheduler_name = "None";

					switch (scheduler) {
						case 0:
							scheduling_duration = 0;
							break;
						case 1:
							schedule_start = std::chrono::steady_clock::now();
							circuit->set_scheduler(new AsapScheduler());
							schedule_end = std::chrono::steady_clock::now();
							scheduling_duration = std::chrono::duration_cast<std::chrono::microseconds>(schedule_end - schedule_start).count();
							break;
					}

					if (circuit->get_scheduler() != nullptr) scheduler_name = (char*)circuit->get_scheduler()->name;

					std::cout << "Running setup " << setup << " with scheduler " << scheduler_name << std::endl;

					ss << setup << CSV_SEPARATOR << scheduler_name << CSV_SEPARATOR << std::chrono::duration_cast<std::chrono::microseconds>(build_end - build_start).count() / 1000.0 << CSV_SEPARATOR << scheduling_duration / 1000.0;

					for (uint32_t iteration = 0; iteration < setup_iters; iteration++) { //go through iterations
						std::cout << "Running iteration " << iteration << " host" << std::endl;

						circuit->reset_circuit();
						config_circuit(setup, iteration, false); //configure circuit for host iteration

						//simulate circuit and measure time taken
						auto h_iter_start = std::chrono::steady_clock::now();
						circuit->simulate_circuit_host_only();
						auto h_iter_end = std::chrono::steady_clock::now();

						ss << CSV_SEPARATOR << std::chrono::duration_cast<std::chrono::microseconds>(h_iter_end - h_iter_start).count() / 1000.0;

						std::cout << "Running iteration " << iteration << " device" << std::endl;

						circuit->reset_circuit();
						uint32_t iter_length = config_circuit(setup, iteration, true); //configure circuit for host iteration

						//simulate circuit and measure time taken
						auto d_iter_start = std::chrono::steady_clock::now();
						circuit->copy_data_to_device(iter_length);
						auto d_comp_start = std::chrono::steady_clock::now();
						circuit->simulate_circuit_dev_nocopy();
						cudaDeviceSynchronize();
						auto d_comp_end = std::chrono::steady_clock::now();
						circuit->copy_data_from_device(iter_length);
						auto d_iter_end = std::chrono::steady_clock::now();

						ss << CSV_SEPARATOR << std::chrono::duration_cast<std::chrono::microseconds>(d_iter_end - d_iter_start).count() / 1000.0;
						ss << CSV_SEPARATOR << std::chrono::duration_cast<std::chrono::microseconds>(d_comp_end - d_comp_start).count() / 1000.0;

						post_iter(setup, iteration); //run post-iteration tasks if required
					}

					for (uint32_t i = setup_iters; i < num_iterations; i++) ss << CSV_SEPARATOR << CSV_SEPARATOR << CSV_SEPARATOR; //pad columns for missing iterations

					post_setup(setup, ss); //run post-setup tasks if required (may write additional column values)

					ss << "\n";
				}
			}
		} catch (CudaError& error) {
			if (error.error != cudaErrorMemoryAllocation) throw error;
			std::cout << "Out of memory, stopping testbench" << std::endl;
			ss << "\n";
		}

		std::cout << "---- Testbench finished ----" << std::endl;

		auto str = ss.str();
		std::replace(str.begin(), str.end(), '.', CSV_DECIMAL_SEP); //correct decimal point/separator

		return str;
	}

protected:
	StochasticCircuit* circuit;
	StochasticCircuitFactory factory;

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
	/// <returns>How many bits per net are relevant (at most)</returns>
	virtual uint32_t config_circuit(uint32_t setup, uint32_t iteration, bool device) = 0;

	/// <summary>
	/// Run post-iteration tasks/calculations if required
	/// </summary>
	virtual void post_iter(uint32_t setup, uint32_t iteration) { }

	/// <summary>
	/// Run post-setup tasks/calculations if required, may append additional column values, each preceded by CSV_SEPARATOR
	/// </summary>
	virtual void post_setup(uint32_t setup, std::stringstream& ss) { }

};
