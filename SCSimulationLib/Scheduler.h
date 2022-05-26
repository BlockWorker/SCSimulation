#pragma once
#include "library_export.h"
#include <stdint.h>

namespace scsim {

	class CircuitComponent;
	class StochasticCircuit;

	class SCSIMAPI Scheduler
	{
	public:
		const char* const name;

		Scheduler(const char* name) : name(name) {}
		virtual ~Scheduler() {}

		Scheduler(const Scheduler& other) = delete;
		Scheduler& operator=(const Scheduler& other) = delete;
		Scheduler(Scheduler&& other) = delete;
		Scheduler& operator=(Scheduler&& other) = delete;

		/// <summary>
		/// Compile the simulation schedule for the given circuit structure.
		/// </summary>
		/// <param name="circuit">Circuit object to schedule the simulation for.</param>
		virtual void compile(StochasticCircuit* circuit) = 0;
		virtual bool is_compiled() const = 0;
		
		/// <summary>
		/// Execute the previously compiled schedule.
		/// </summary>
		/// <param name="host">Whether the simulation should run on the host (as opposed to the device).</param>
		/// <returns>Whether the scheduler execution was successful.</returns>
		virtual bool execute(bool host) = 0;

	};

}
