#pragma once

#include "Testbench.cuh"
#include "StochasticNumber.cuh"
#include "Stanh.cuh"

class StanhTestbench : public Testbench
{
public:
	StanhTestbench(uint32_t sim_length, uint32_t states, uint32_t setups) : Testbench(setups, 2), sim_length(sim_length), states(states) {
		numbers = nullptr;
		vals = nullptr;
	}

	virtual ~StanhTestbench() {
		if (numbers != nullptr) for (uint32_t i = 0; i <= count; i++) delete numbers[i];
		free(numbers);
		free(vals);
	}

protected:
	const uint32_t sim_length;
	const uint32_t states;

	uint32_t first_in;
	uint32_t first_out;
	uint32_t count;

	StochasticNumber** numbers;
	double* vals;

	virtual void build_circuit(uint32_t setup) override {
		factory->set_sim_length(sim_length);

		count = 16;
		for (uint32_t i = 0; i < setup; i++) count *= 2;

		first_in = factory->add_nets(count + 1).first;
		first_out = factory->add_nets(count + 1).first;

		for (uint32_t i = 0; i <= count; i++) {
			factory->add_component(new Stanh(first_in + i, first_out + i, states));
		}
	}

	virtual bool config_circuit(uint32_t iteration) override {
		if (iteration == 0) {
			numbers = (StochasticNumber**)malloc((count + 1) * sizeof(StochasticNumber*));
			vals = (double*)malloc((count + 1) * sizeof(double));
			for (uint32_t i = 0; i <= count; i++) {
				vals[i] = (double)i / (double)count;
			}

			StochasticNumber::generate_multiple_curand(numbers, sim_length, vals, count + 1);
		}

		for (uint32_t i = 0; i <= count; i++) {
			circuit->set_net_value(first_in + i, numbers[i]);
		}

		if (iteration == 1) {
			for (uint32_t i = 0; i <= count; i++) delete numbers[i];
			free(numbers);
			free(vals);
			numbers = nullptr;
			vals = nullptr;
		}

		return iteration == 0;
	}

	virtual void post_iter(uint32_t setup, uint32_t iteration, std::stringstream& ss) override {
		if (iteration == 1) {
			double errsum_in = 0;
			double errsum_out = 0;
			double errsum_total = 0;
			for (uint32_t i = 0; i <= count; i++) {
				auto correct_in = 2.0 * (double)i / (double)count - 1.0;
				auto actual_in = circuit->get_net_value_bipolar(first_in + i);
				auto in_err = actual_in - correct_in;
				errsum_in += in_err * in_err;

				auto int_correct_out = tanh((states / 2.0) * actual_in);
				auto actual_out = circuit->get_net_value_bipolar(first_out + i);
				auto out_err = actual_out - int_correct_out;
				errsum_out += out_err * out_err;

				auto total_correct_out = tanh((states / 2.0) * correct_in);
				auto total_err = actual_out - total_correct_out;
				errsum_total += total_err * total_err;
			}
			auto rmse_in = sqrt(errsum_in / (double)(count + 1));
			auto rmse_out = sqrt(errsum_out / (double)(count + 1));
			auto rmse_total = sqrt(errsum_total / (double)(count + 1));
			ss << CSV_SEPARATOR << rmse_in << CSV_SEPARATOR << rmse_out << CSV_SEPARATOR << rmse_total;
		}
	}

};
