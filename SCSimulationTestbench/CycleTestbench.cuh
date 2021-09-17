﻿#pragma once

#include "Testbench.cuh"
#include "StochasticNumber.cuh"
#include "BitwiseAbleComponents.cuh"

class CycleTestbench : public Testbench
{
public:
	/// <param name="setups">Number of setups, setup i contains 2^(i+4) inverters</param>
	CycleTestbench(uint32_t min_sim_length, uint32_t setups, uint32_t max_iter_runs) : Testbench(setups, max_iter_runs), min_sim_length(min_sim_length), max_iter_runs(max_iter_runs) {
		numbers = nullptr;
		vals = nullptr;
		count = 0;
		first_in = 0;
		first_int = 0;
		first_out = 0;
		curr_max_sim_length = min_sim_length;
	}

	virtual ~CycleTestbench() {
		if (numbers != nullptr) for (uint32_t i = 0; i <= count; i++) delete numbers[i];
		free(numbers);
		free(vals);
	}

protected:
	const uint32_t min_sim_length;
	const uint32_t max_iter_runs;

	uint32_t curr_max_sim_length;

	uint32_t first_in;
	uint32_t first_int;
	uint32_t first_out;
	uint32_t count;

	StochasticNumber** numbers;
	double* vals;

	virtual uint32_t build_circuit(uint32_t setup) override {
		auto num_runs = __min(num_setups - setup, max_iter_runs);

		curr_max_sim_length = min_sim_length;
		count = 16;
		for (uint32_t i = 0; i < setup; i++) {
			count *= 2;
		}
		for (uint32_t i = 1; i < num_runs; i++) {
			curr_max_sim_length *= 2;
		}

		factory->set_sim_length(curr_max_sim_length);

		first_in = factory->add_nets(count + 1).first;
		first_int = factory->add_nets(count + 1).first;
		first_out = factory->add_nets(count + 1).first;

		for (uint32_t i = 0; i <= count; i++) {
			factory->add_component(new XorGate(first_in + i, first_int + i, first_out + i));
			factory->add_component(new Delay(first_out + i, first_int + i));
		}

		return num_runs;
	}

	virtual uint32_t config_circuit(uint32_t setup, uint32_t iteration, bool device) override {
		uint32_t iter_sim_length = min_sim_length;
		for (uint32_t i = 0; i < iteration; i++) {
			iter_sim_length *= 2;
		}

		if (!device) {
			numbers = (StochasticNumber**)calloc(count + 1, sizeof(StochasticNumber*));
			vals = (double*)malloc((count + 1) * sizeof(double));
			for (uint32_t i = 0; i <= count; i++) {
				vals[i] = (double)i / (double)count;
			}

			StochasticNumber::generate_multiple_curand(numbers, iter_sim_length, vals, count + 1);
		}

		for (uint32_t i = 0; i <= count; i++) {
			circuit->set_net_value(first_in + i, numbers[i]);
		}

		if (device) {
			for (uint32_t i = 0; i <= count; i++) delete numbers[i];
			free(numbers);
			free(vals);
			numbers = nullptr;
			vals = nullptr;
		}

		return iter_sim_length;
	}

	virtual void write_additional_column_titles(std::stringstream& ss) override {
		ss << CSV_SEPARATOR << min_sim_length;
	}

};