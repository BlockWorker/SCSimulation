﻿#pragma once

#include "Testbench.cuh"
#include "StochasticNumber.cuh"
#include "ParallelCounter.cuh"
#include "Btanh.cuh"

class PCBtanhTestbench : public Testbench
{
public:
	/// <param name="pc_inputs">Number of parallel cocunter inputs before each Btanh component</param>
	/// <param name="setups">Number of setups, setup i contains 2^(i+4) PC-Btanh circuits</param>
	PCBtanhTestbench(uint32_t min_sim_length, uint32_t pc_inputs, uint32_t setups, uint32_t max_iter_runs) : Testbench(setups, max_iter_runs), min_sim_length(min_sim_length), pc_inputs(pc_inputs),
		pc_intermediates((uint32_t)floor(log2((double)pc_inputs)) + 1), max_iter_runs(max_iter_runs), btanh_r(Btanh::calculate_r(pc_inputs, 1.0)) {
		numbers = nullptr;
		vals = nullptr;
		count = 0;
		first_in = 0;
		first_intermediate = 0;
		first_out = 0;
		curr_max_sim_length = min_sim_length;
	}

	virtual ~PCBtanhTestbench() {
		if (numbers != nullptr) for (uint32_t i = 0; i <= count * pc_intermediates; i++) delete numbers[i];
		free(numbers);
		free(vals);
	}

protected:
	const uint32_t min_sim_length;
	const uint32_t max_iter_runs;
	const uint32_t pc_inputs;
	const uint32_t pc_intermediates;
	const uint32_t btanh_r;

	uint32_t curr_max_sim_length;

	uint32_t first_in;
	uint32_t first_intermediate;
	uint32_t first_out;
	uint32_t count;

	StochasticNumber** numbers;
	double* vals;

	virtual uint32_t build_circuit(uint32_t setup) override {
		auto num_runs = __min(num_setups - setup, max_iter_runs);
		if (setup >= 2 && num_runs >= 2 && num_runs < max_iter_runs) num_runs = 2; //reduce simulation runs - only max length for small setups, then restrict to 2 runs each

		curr_max_sim_length = min_sim_length << (num_runs - 1);
		count = 16 << setup;

		factory.set_sim_length(curr_max_sim_length);

		first_in = factory.add_nets(pc_inputs * (count + 1)).first;
		first_intermediate = factory.add_nets(pc_intermediates * (count + 1)).first;
		first_out = factory.add_nets(count + 1).first;

		for (uint32_t i = 0; i <= count; i++) {
			factory_add_component(factory, ParallelCounter, pc_inputs, first_in + pc_inputs * i, first_intermediate + pc_intermediates * i);
			factory_add_component(factory, Btanh, pc_inputs, btanh_r, first_intermediate + pc_intermediates * i, first_out + i);
		}

		return num_runs;
	}

	virtual void config_circuit(uint32_t setup, uint32_t scheduler, uint32_t iteration, uint32_t input, bool device) override {
		uint32_t iter_sim_length = min_sim_length << iteration;

		if (!device) {
			numbers = (StochasticNumber**)calloc(pc_inputs * (count + 1), sizeof(StochasticNumber*));
			vals = (double*)malloc(pc_inputs * (count + 1) * sizeof(double));
			for (uint32_t i = 0; i <= count; i++) { //create inputs - desired tanh input is (i / count), distribute over PC inputs by way of terminated geometric series
				double total = 2.0 * (double)i / (double)count - 1.0;
				double factor = 1.0;
				for (uint32_t j = 0; j < pc_inputs - 1; j++) {
					factor *= .5;
					vals[pc_inputs * i + j] = (factor * total + 1.0) / 2.0;
				}
				vals[pc_inputs * (i + 1) - 1] = (factor * total + 1.0) / 2.0; //repeat last term of geometric series, making sum equal to 1 (times desired input)
			}

			StochasticNumber::generate_multiple_curand(numbers, iter_sim_length, vals, pc_inputs * (count + 1));
		}

		for (uint32_t i = 0; i < pc_inputs * (count + 1); i++) {
			circuit->set_net_value(first_in + i, *numbers[i]);
		}

		if (device) {
			for (uint32_t i = 0; i < pc_inputs * (count + 1); i++) delete numbers[i];
			free(numbers);
			free(vals);
			numbers = nullptr;
			vals = nullptr;
		}
	}

	virtual uint32_t get_iter_length(uint32_t setup, uint32_t scheduler, uint32_t iteration) override {
		return min_sim_length << iteration;
	}

	virtual void write_additional_column_titles(std::stringstream& ss) override {
		ss << CSV_SEPARATOR << "RMSE(sn)" << CSV_SEPARATOR << "RMSE(circuit)" << CSV_SEPARATOR << "RMSE(total)" << CSV_SEPARATOR << min_sim_length;
	}

	//calculate and log RMSE of generated SN vs. expected value, RMSE of the circuit calculation itself, and total RMSE
	virtual void post_setup(uint32_t setup, uint32_t scheduler, std::stringstream& ss) override {
		double errsum_in = 0;
		double errsum_out = 0;
		double errsum_total = 0;
		for (uint32_t i = 0; i <= count; i++) {
			auto correct_in = 2.0 * (double)i / (double)count - 1.0;
			auto actual_in = 0.0;
			for (uint32_t j = 0; j < pc_inputs; j++) actual_in += circuit->get_net_value_bipolar(first_in + i * pc_inputs + j);
			auto in_err = actual_in - correct_in;
			errsum_in += in_err * in_err;

			auto int_correct_out = tanh(actual_in);
			auto actual_out = circuit->get_net_value_bipolar(first_out + i);
			auto out_err = actual_out - int_correct_out;
			errsum_out += out_err * out_err;

			auto total_correct_out = tanh(correct_in);
			auto total_err = actual_out - total_correct_out;
			errsum_total += total_err * total_err;
		}
		auto rmse_in = sqrt(errsum_in / (double)(count + 1));
		auto rmse_out = sqrt(errsum_out / (double)(count + 1));
		auto rmse_total = sqrt(errsum_total / (double)(count + 1));
		ss << CSV_SEPARATOR << rmse_in << CSV_SEPARATOR << rmse_out << CSV_SEPARATOR << rmse_total;
	}

};
