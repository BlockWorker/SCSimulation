#pragma once

#ifdef SCSIMULATION_EXPORT
#define SCSIMAPI __declspec(dllexport)
#else
#define SCSIMAPI __declspec(dllimport)
#endif
