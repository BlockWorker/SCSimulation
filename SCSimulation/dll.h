#pragma once

//macros for DLL exporting/importing
#ifdef SCSIMULATION_EXPORT
#define SCSIMAPI __declspec(dllexport)
#else
#define SCSIMAPI __declspec(dllimport)
#endif
