#pragma once

#include "Graph.h"

namespace scsim {

	class Schedule
	{
	public:
		const Graph& const graph;

		Schedule(const Graph& graph);

		/// <returns>Whether the underlying graph is valid for scheduling (i.e. acyclic)</returns>
		bool is_graph_valid() const;

		/// <summary>
		/// Updates the underlying graph. Invalidates previously calculated schedule times.
		/// </summary>
		/// <returns>Whether the updated underlying graph is valid</returns>
		bool update_graph();

		/// <summary>
		/// Calculates the schedule times using ASAP scheduling. Requires underlying graph to be valid.
		/// </summary>
		void calculate_times_asap();

		/// <summary>
		/// Calculates the schedule times using ALAP scheduling. Requires underlying graph to be valid.
		/// </summary>
		void calculate_times_alap();

		bool schedule_times_calculated() const;
		const std::vector<uint32_t>& get_schedule_times() const;

	protected:
		std::vector<uint32_t> topological_order;
		std::vector<uint32_t> schedule_times;

		void calculate_asap(std::vector<uint32_t>& result_times) const;
		void calculate_alap(std::vector<uint32_t>& result_times) const;

	};

}
