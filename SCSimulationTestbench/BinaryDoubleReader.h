#pragma once

#include <stdint.h>
#include <sstream>
#include <vector>

/// <summary>
/// Reads a binary file with double-type entries and provides its data to the application.
/// </summary>
class BinaryDoubleReader
{
public:
	const std::string filename;

	/// <summary>
	/// Read and parse the given CSV file into memory, for easy access from this object.
	/// </summary>
	/// <param name="filename">Path to the CSV file</param>
	/// <param name="max_values">Maximum number of double values to read, 0 means no limit</param>
	BinaryDoubleReader(std::string filename, size_t max_values = 0);
	~BinaryDoubleReader();

	BinaryDoubleReader(const BinaryDoubleReader& other) = delete;
	BinaryDoubleReader& operator=(const BinaryDoubleReader& other) = delete;
	BinaryDoubleReader(BinaryDoubleReader&& other) = delete;
	BinaryDoubleReader& operator=(BinaryDoubleReader&& other) = delete;

	size_t get_value_count() const;
	const double* get_data() const;
	double get_value(size_t index) const;

private:
	size_t value_count;
	double* data;

};

