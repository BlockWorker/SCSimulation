#include "BinaryDoubleReader.h"

#include <fstream>
#include <sys/stat.h>

using namespace std;

BinaryDoubleReader::BinaryDoubleReader(std::string filename, size_t max_values) : filename(filename) {
	struct stat filestats;
	if (stat(filename.c_str(), &filestats) != 0) throw runtime_error("BinaryDoubleReader: Unable to get file information");

	value_count = filestats.st_size / sizeof(double); //number of double values in the file

	if (value_count == 0) { //no values: no data to allocate
		data = nullptr;
		return;
	}

	if (max_values > 0 && value_count > max_values) value_count = max_values; //limit to maximum number of values
	data = new double[value_count]; //allocate data array

	ifstream file_stream(filename, ios::in | ios::binary); //open binary file
	if (!file_stream.read((char*)data, value_count * sizeof(double))) { //read file into array if possible
		file_stream.close();
		throw runtime_error("BinaryDoubleReader: Unable to read file");
	}
	file_stream.close();
}

BinaryDoubleReader::~BinaryDoubleReader() {
	delete[] data;
}

size_t BinaryDoubleReader::get_value_count() const {
	return value_count;
}

const double* BinaryDoubleReader::get_data() const {
	return data;
}

double BinaryDoubleReader::get_value(size_t index) const {
	if (index >= value_count) throw runtime_error("get_value: Invalid index");
	return data[index];
}
