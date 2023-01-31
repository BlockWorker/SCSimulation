#include "NumericCSVReader.h"

#include <fstream>
#include <algorithm>

using namespace std;


static vector<string> split(const string& s, char delim) {
    vector<string> result;
    stringstream ss(s);
    string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}


NumericCSVReader::NumericCSVReader(string filename, size_t max_rows, char csv_sep, char csv_decimal) : filename(filename) {
    if (csv_sep == '.') throw runtime_error("NumericCSVReader: CSV separator '.' (dot) is not supported.");

    size_t num_rows = 0;
    vector<size_t> row_lengths_v;
    vector<double*> row_v;

    ifstream file_stream(filename);

    string line;
    while (getline(file_stream, line, '\n') && (max_rows == 0 || num_rows < max_rows)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); //remove potentially left over CR
        if (line.empty() || line.front() == '#') continue; //skip empty lines or lines that start with a # (comments)

        num_rows++;

        replace(line.begin(), line.end(), csv_decimal, '.'); //replace decimal with dot

        auto parts = split(line, csv_sep); //split line into values
        auto num_parts = parts.size();
        row_lengths_v.push_back(num_parts);
        auto values = new double[num_parts];

        #pragma omp parallel for
        for (int i = 0; i < num_parts; i++) {
            values[i] = atof(parts[i].c_str());
        }

        row_v.push_back(values);
    }

    file_stream.close();

    row_count = num_rows;

    if (num_rows == 0) { //no rows -> no data
        data = nullptr;
        row_lengths = nullptr;
    } else { //otherwise, copy vectors to final arrays
        data = new double*[num_rows];
        row_lengths = new size_t[num_rows];

        memcpy(data, row_v.data(), row_v.size() * sizeof(double*));
        memcpy(row_lengths, row_lengths_v.data(), row_lengths_v.size() * sizeof(size_t));
    }
}

NumericCSVReader::~NumericCSVReader() {
    delete[] row_lengths;
    for (size_t i = 0; i < row_count; i++) delete[] data[i];
    delete[] data;
}

size_t NumericCSVReader::get_row_count() const {
    return row_count;
}

size_t NumericCSVReader::get_row_length(size_t row) const {
    if (row >= row_count) throw runtime_error("get_row_length: Invalid row index.");
    return row_lengths[row];
}

const size_t* NumericCSVReader::get_row_lengths() const {
    return row_lengths;
}

const double* const* NumericCSVReader::get_data() const {
    return data;
}

const double* NumericCSVReader::get_row(size_t row) const {
    if (row >= row_count) throw runtime_error("get_row: Invalid row index.");
    return data[row];
}

double NumericCSVReader::get_value(size_t row, size_t column) const {
    if (row >= row_count) throw runtime_error("get_value: Invalid row index.");
    if (column >= row_lengths[row]) throw runtime_error("get_value: Invalid column index.");
    return data[row][column];
}
