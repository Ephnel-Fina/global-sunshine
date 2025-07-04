#pragma once

#include <string>
#include <memory>

using namespace std;

class Raster
{
public:
    Raster() = default;
    Raster(const string &file_path);
    Raster(size_t rows, size_t cols);

    void copyGeoTransformFrom(const Raster &raster);

    size_t size() const;
    void printInfo();
    void save(const string &file_path);

    string file_path;
    shared_ptr<float> data;
    size_t rows = 0;
    size_t cols = 0;
    string projection;
    double geo_transform[6];
    float no_data_value = 65535;

    float max_value = 0;
    float min_value = 0;
};

struct IndexRange
{
    size_t row_from;
    size_t row_to;
    size_t col_from;
    size_t col_to;
};