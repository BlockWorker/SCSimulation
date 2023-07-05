#pragma once
#include "cuda_base.cuh"
#include "Scheduler.h"

namespace scsim {

	class HostAsapScheduler;

	/// <summary>
	/// ASAP (as-soon-as-possible) simulation scheduler, supporting host-only and device-assisted simulations
	/// </summary>
	class SCSIMAPI AsapScheduler : public Scheduler
	{
	public:
		AsapScheduler();
		virtual ~AsapScheduler();

		virtual void compile(StochasticCircuit* circuit);
		virtual bool is_compiled() const;

		virtual bool execute(bool host);

	private:
		StochasticCircuit* circuit;
		HostAsapScheduler* host_scheduler;
		cudaGraph_t cuda_graph;
		cudaGraphExec_t cuda_graph_exec;
		uint32_t* sim_comb_dev;
		uint32_t* comb_type_counts_dev;
		uint32_t* comb_type_offsets_dev;
		uint32_t* sim_seq_dev;
		uint32_t* seq_type_counts_dev;
		uint32_t* seq_type_offsets_dev;

	};

}