﻿#pragma once

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
/// Base class for different test benches. Simulates component setups in different conditions and outputs results in CSV format
/// </summary>
class Testbench
{
public:
	const uint32_t num_setups;
	const uint32_t num_iterations;
	const bool devside_config;

	/// <param name="num_setups">Number of different circuit setups to test</param>
	/// <param name="num_iterations">Number of iterations (with potentially different parameters) to run per setup</param>
	/// <param name="num_inputs">Number of different simulation inputs per iteration</param>
	/// <param name="devside_config">Whether simulation inputs for device are generated on device side directly (meaning no more copying is required)</param>
	Testbench(uint32_t num_setups, uint32_t num_iterations, bool devside_config = false) : num_setups(num_setups), num_iterations(num_iterations),
		devside_config(devside_config), factory(StochasticCircuitFactory(false)) {
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

		ss << "Setup index" << CSV_SEPARATOR << "Scheduler" << CSV_SEPARATOR << "Build time" << CSV_SEPARATOR << "Sched time";
		for (uint32_t i = 0; i < num_iterations; i++) { //write iteration indices to title row
			ss << CSV_SEPARATOR << i << "HostCfg" << CSV_SEPARATOR << i << "HostRun" << CSV_SEPARATOR << i << "DevTotal" << CSV_SEPARATOR << i << "DevCfg" << CSV_SEPARATOR << i << "DevRun";
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

#ifdef ALLOCSIZETEST
				continue;
#endif

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
							auto sched = new AsapScheduler();
							std::cout << "Scheduling setup " << setup << " with scheduler " << sched->name << std::endl;
							schedule_start = std::chrono::steady_clock::now();
							circuit->set_scheduler(sched);
							schedule_end = std::chrono::steady_clock::now();
							scheduling_duration = std::chrono::duration_cast<std::chrono::microseconds>(schedule_end - schedule_start).count();
							break;
					}

					if (circuit->get_scheduler() != nullptr) scheduler_name = (char*)circuit->get_scheduler()->name;

					std::cout << "Running setup " << setup << " with scheduler " << scheduler_name << std::endl;

					ss << setup << CSV_SEPARATOR << scheduler_name << CSV_SEPARATOR << std::chrono::duration_cast<std::chrono::microseconds>(build_end - build_start).count() / 1000.0 << CSV_SEPARATOR << scheduling_duration / 1000.0;

					for (uint32_t iteration = 0; iteration < setup_iters; iteration++) { //go through iterations
						double host_total_cfg_ms = 0;
						double host_total_ms = 0;

						uint32_t num_inputs_host = get_iter_inputs(setup, scheduler, iteration, false);

						for (uint32_t input = 0; input < num_inputs_host; input++) {
							std::cout << "Running iteration " << iteration << " input " << input << " host" << std::endl;

							auto h_cfg_start = std::chrono::steady_clock::now();
							circuit->reset_circuit();
							config_circuit(setup, scheduler, iteration, input, false); //configure circuit for host iteration
							auto h_cfg_end = std::chrono::steady_clock::now();

							host_total_cfg_ms += std::chrono::duration_cast<std::chrono::microseconds>(h_cfg_end - h_cfg_start).count() / 1000.0;

							//simulate circuit and measure time taken
							auto h_iter_start = std::chrono::steady_clock::now();
							circuit->simulate_circuit_host_only();
							auto h_iter_end = std::chrono::steady_clock::now();

							host_total_ms += std::chrono::duration_cast<std::chrono::microseconds>(h_iter_end - h_iter_start).count() / 1000.0;

							post_input(setup, scheduler, iteration, input, false);
						}

						ss << CSV_SEPARATOR << (host_total_cfg_ms / num_inputs_host);
						ss << CSV_SEPARATOR << (host_total_ms / num_inputs_host);

						double dev_total_cfg_ms = 0;
						double dev_total_comp_ms = 0;

						uint32_t iter_length = get_iter_length(setup, scheduler, iteration);
						uint32_t num_inputs_dev = get_iter_inputs(setup, scheduler, iteration, true);

						auto d_iter_start = std::chrono::steady_clock::now();
						if (devside_config) circuit->copy_data_to_device(iter_length);
						for (uint32_t input = 0; input < num_inputs_dev; input++) {
							std::cout << "Running iteration " << iteration << " input " << input << " device" << std::endl;

							auto d_cfg_start = std::chrono::steady_clock::now();
							circuit->reset_circuit();
							config_circuit(setup, scheduler, iteration, input, true); //configure circuit for device iteration
							auto d_cfg_end = std::chrono::steady_clock::now();

							dev_total_cfg_ms += std::chrono::duration_cast<std::chrono::microseconds>(d_cfg_end - d_cfg_start).count() / 1000.0;

							//simulate circuit and measure time taken
							if (!devside_config) circuit->copy_data_to_device(iter_length);
							auto d_comp_start = std::chrono::steady_clock::now();
							circuit->simulate_circuit_dev_nocopy();
							cudaDeviceSynchronize();
							auto d_comp_end = std::chrono::steady_clock::now();
							if (!devside_config) circuit->copy_data_from_device(iter_length);
							
							dev_total_comp_ms += std::chrono::duration_cast<std::chrono::microseconds>(d_comp_end - d_comp_start).count() / 1000.0;

							post_input(setup, scheduler, iteration, input, true);
						}
						if (devside_config) circuit->copy_data_from_device(iter_length);
						auto d_iter_end = std::chrono::steady_clock::now();

						ss << CSV_SEPARATOR << std::chrono::duration_cast<std::chrono::microseconds>(d_iter_end - d_iter_start).count() / 1000.0;
						ss << CSV_SEPARATOR << (dev_total_cfg_ms / num_inputs_dev);
						ss << CSV_SEPARATOR << (dev_total_comp_ms / num_inputs_dev);

						post_iter(setup, scheduler, iteration); //run post-iteration tasks if required
					}

					//pad columns for missing iterations
					for (uint32_t i = setup_iters; i < num_iterations; i++) ss << CSV_SEPARATOR << CSV_SEPARATOR << CSV_SEPARATOR << CSV_SEPARATOR << CSV_SEPARATOR;

					post_setup(setup, scheduler, ss); //run post-setup tasks if required (may write additional column values)

					ss << "\n";
				}
			}
			delete circuit;
			circuit = nullptr;
		} catch (CudaError& error) {
			delete circuit;
			circuit = nullptr;
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
	virtual void config_circuit(uint32_t setup, uint32_t scheduler, uint32_t iteration, uint32_t input, bool device) = 0;

	/// <returns>How many bits per net are relevant (at most)</returns>
	virtual uint32_t get_iter_length(uint32_t setup, uint32_t scheduler, uint32_t iteration) = 0;

	/// <returns>How many inputs should be used for the given iteration</returns>
	virtual uint32_t get_iter_inputs(uint32_t setup, uint32_t scheduler, uint32_t iteration, bool device) {
		return 1;
	}

	/// <summary>
	/// Run post-input tasks/calculations if required
	/// </summary>
	virtual void post_input(uint32_t setup, uint32_t scheduler, uint32_t iteration, uint32_t input, bool device) { }

	/// <summary>
	/// Run post-iteration tasks/calculations if required
	/// </summary>
	virtual void post_iter(uint32_t setup, uint32_t scheduler, uint32_t iteration) { }

	/// <summary>
	/// Run post-setup tasks/calculations if required, may append additional column values, each preceded by CSV_SEPARATOR
	/// </summary>
	virtual void post_setup(uint32_t setup, uint32_t scheduler, std::stringstream& ss) { }

};
