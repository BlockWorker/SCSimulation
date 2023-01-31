#pragma once

#include <sstream>
#include <vector>

/// <summary>
/// Reads a CSV-formatted file with only numerical (double type) entries and provides its data to the application.
/// </summary>
class NumericCSVReader
{
public:
	const std::string filename;

	/// <summary>
	/// Read and parse the given CSV file into memory, for easy access from this object.
	/// </summary>
	/// <param name="filename">Path to the CSV file</param>
	/// <param name="max_rows">Maximum number of rows/lines to read, 0 means no limit</param>
	/// <param name="csv_sep">Separating character between columns/values in the CSV file, may not be '.' (dot)</param>
	/// <param name="csv_decimal">Character used as a decicmal point in the CSV file</param>
	NumericCSVReader(std::string filename, size_t max_rows = 0, char csv_sep = ',', char csv_decimal = '.');
	~NumericCSVReader();

	NumericCSVReader(const NumericCSVReader& other) = delete;
	NumericCSVReader& operator=(const NumericCSVReader& other) = delete;
	NumericCSVReader(NumericCSVReader&& other) = delete;
	NumericCSVReader& operator=(NumericCSVReader&& other) = delete;

	size_t get_row_count() const;
	size_t get_row_length(size_t row) const;
	const size_t* get_row_lengths() const;
	const double* const* get_data() const;
	const double* get_row(size_t row) const;
	double get_value(size_t row, size_t column) const;

private:
	size_t row_count;
	size_t* row_lengths;
	double** data;

};

