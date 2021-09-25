#pragma once

#include "Testbench.cuh"
#include "StochasticNumber.cuh"
#include "BasicCombinatorial.cuh"

class ChainedInverterTestbench : public Testbench
{
public:
	/// <param name="setups">Number of setups, setup i contains 2^(i+4) inverters</param>
	ChainedInverterTestbench(uint32_t min_sim_length, uint32_t setups, uint32_t max_iter_runs) : Testbench(2 * setups, max_iter_runs), min_sim_length(min_sim_length), max_iter_runs(max_iter_runs), setups(setups) {
		count = 0;
		in = 0;
		out = 0;
		curr_max_sim_length = min_sim_length;
	}

protected:
	const uint32_t min_sim_length;
	const uint32_t max_iter_runs;
	const uint32_t setups;

	uint32_t curr_max_sim_length;

	uint32_t in;
	uint32_t out;
	uint32_t count;

	StochasticNumber* number;

	virtual uint32_t build_circuit(uint32_t setup) override {
		auto setup_id = setup / 2;
		auto setup_kind = setup % 2;
		auto num_runs = __min(setups - setup_id, max_iter_runs);

		curr_max_sim_length = min_sim_length;
		count = 16;
		for (uint32_t i = 0; i < setup_id; i++) {
			count *= 2;
		}
		for (uint32_t i = 1; i < num_runs; i++) {
			curr_max_sim_length *= 2;
		}

		factory.set_sim_length(curr_max_sim_length);

		auto netids = factory.add_nets(count + 1);
		in = netids.first;
		out = netids.second;

		if (setup_kind == 0) {
			for (uint32_t i = 0; i < count; i++) {
				factory_add_component(factory, Inverter, in + i, in + i + 1);
			}
		} else {
			for (uint32_t i = 0; i < count; i++) {
				factory_add_component(factory, Inverter, out - i - 1, out - i);
			}
		}

		return num_runs;
	}

	virtual uint32_t config_circuit(uint32_t setup, uint32_t iteration, bool device) override {
		uint32_t iter_sim_length = min_sim_length;
		for (uint32_t i = 0; i < iteration; i++) {
			iter_sim_length *= 2;
		}

		if (!device) number = StochasticNumber::generate_unipolar(iter_sim_length, 0.25);

		circuit->set_net_value(in, number);

		if (device) delete number;

		return iter_sim_length;
	}

	virtual void write_additional_column_titles(std::stringstream& ss) override {
		ss << CSV_SEPARATOR << "Order" << CSV_SEPARATOR << min_sim_length;
	}

	virtual void post_setup(uint32_t setup, std::stringstream& ss) override {
		ss << CSV_SEPARATOR << (setup % 2 == 0) ? "Forward" : "Backward";
	}

};
