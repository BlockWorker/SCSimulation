#pragma once
#include "Scheduler.h"

#include <vector>

namespace scsim {

	class Graph;
	class AsapScheduler;

	/// <summary>
	/// ASAP (as-soon-as-possible) simulation scheduler for host-only simulations
	/// </summary>
	class SCSIMAPI HostAsapScheduler : public Scheduler
	{
	public:
		HostAsapScheduler();
		virtual ~HostAsapScheduler();

		virtual void compile(StochasticCircuit* circuit);
		virtual bool is_compiled() const;

		virtual bool execute(bool host);

	private:
		friend AsapScheduler;

		StochasticCircuit* circuit;
		Graph* graph;
		std::vector<std::vector<uint32_t>> schedule_buckets;

	};

}


