#include "Schedule.h"

#include <stdexcept>

namespace scsim {

	Schedule::Schedule(const Graph& graph) : graph(graph) {
		update_graph();
	}

	bool Schedule::is_graph_valid() const {
		return !topological_order.empty();
	}

	bool Schedule::update_graph() {
		schedule_times.clear();
		return graph.topological_sort(topological_order);
	}

	void Schedule::calculate_times_asap() {
		if (!is_graph_valid()) throw std::runtime_error("calculate_times_asap: Graph is not valid for scheduling");

		calculate_asap(schedule_times);
	}

	void Schedule::calculate_times_alap() {
		if (!is_graph_valid()) throw std::runtime_error("calculate_times_alap: Graph is not valid for scheduling");

		calculate_alap(schedule_times);
	}

	bool Schedule::schedule_times_calculated() const {
		return !schedule_times.empty();
	}

	const std::vector<uint32_t>& Schedule::get_schedule_times() const {
		return schedule_times;
	}

	void Schedule::calculate_asap(std::vector<uint32_t>& result_times) const {
		result_times.clear();
		
		std::vector<int64_t> sslp_distances;
		graph.topological_sslp(topological_order, -1, sslp_distances);

		result_times.reserve(graph.get_num_vertices());
		for each (auto dist in sslp_distances) {
			if (dist < 0 || dist > UINT32_MAX) result_times.push_back(0);
			else result_times.push_back((uint32_t)dist);
		}
	}

	void Schedule::calculate_alap(std::vector<uint32_t>& result_times) const {
		result_times.clear();

		std::vector<int64_t> sdlp_distances;
		graph.topological_sdlp(topological_order, -1, sdlp_distances);

		uint32_t max = 0;
		for each (auto dist in sdlp_distances) {
			if (dist < 0 || dist > UINT32_MAX) continue;
			auto dist32 = (uint32_t)dist;
			if (max < dist32) max = dist32;
		}

		result_times.reserve(graph.get_num_vertices());
		for each (auto dist in sdlp_distances) {
			if (dist < 0 || dist > UINT32_MAX) result_times.push_back(max);
			else result_times.push_back(max - (uint32_t)dist);
		}
	}

}