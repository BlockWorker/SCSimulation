#include "HostAsapScheduler.h"
#include "StochasticCircuit.cuh"
#include "CircuitComponent.cuh"
#include "Graph.h"
#include <stdexcept>

namespace scsim {

	HostAsapScheduler::HostAsapScheduler() : Scheduler("HostAsapScheduler") {
		circuit = nullptr;
		graph = nullptr;
	}

	HostAsapScheduler::~HostAsapScheduler() {
		delete graph;
	}

	void HostAsapScheduler::compile(StochasticCircuit* circuit) {
		this->circuit = circuit;

		//first step: get component dependencies on nets
		std::vector<std::unordered_set<uint32_t>> net_children(circuit->num_nets); //for each net: set of components dependent on that net's value

		for (uint32_t i = 0; i < circuit->num_components; i++) {
			auto comp = circuit->components_host[i]; //get component i
			for (uint32_t j = 0; j < comp->num_inputs; j++) { //for each input net: add component i as a child of that input net
				net_children[comp->inputs_host[j]].insert(i);
			}
		}

		//second step: create component dependency graph based on net dependencies
		graph = new Graph(circuit->num_components);

		for (uint32_t i = 0; i < circuit->num_components; i++) {
			auto comp = circuit->components_host[i]; //get component i
			for (uint32_t j = 0; j < comp->num_outputs; j++) { //for each output net: add children of output net as children of component i
				auto& output_children = net_children[comp->outputs_host[j]];
				for (uint32_t child : output_children) {
					graph->add_edge(i, child);
				}
			}
		}

		//third step: calculate ASAP scheduling of dependency graph (by means of longest-path calculation)
		std::vector<uint32_t> topological_order; //storage for the graph's topological order
		std::vector<int64_t> schedule_times; //storage for schedule times (longest distances from "root")

		bool acyclic = graph->topological_sort(topological_order); //topologically sort graph in preparation of longest-path calculation
		if (!acyclic) {
			delete graph;
			graph = nullptr;
			throw std::runtime_error("compile: HostAsapScheduler only supports acyclic circuits.");
		}

		graph->topological_sslp(topological_order, -1, schedule_times); //calculate longest path distances from "root"

		//fourth step: distribute components into buckets based on schedule time
		schedule_buckets.clear();

		for (uint32_t i = 0; i < circuit->num_components; i++) {
			auto time = schedule_times[i]; //get component i's schedule time

			if (time < 0 || time > UINT32_MAX) continue; //skip components with invalid schedule times (though they should not exist anyway)

			if (schedule_buckets.size() <= time) schedule_buckets.resize(time + 1); //add more buckets depending on required times

			schedule_buckets[time].push_back(i); //add index to corresponding bucket
		}
	}

	bool HostAsapScheduler::is_compiled() const {
		return graph != nullptr;
	}

	bool HostAsapScheduler::execute(bool host) {
		if (!host || !is_compiled()) return false; //execution only possible on the host, and if compiled

		//iterate through buckets and simulate components within those buckets
		for (auto& bucket : schedule_buckets) {
			for (auto index : bucket) {
				auto comp = circuit->components_host[index];
				comp->calculate_simulation_progress_host();
				comp->simulate_step_host();
				comp->sim_step_finished_host();
			}
		}

		return true;
	}

}