#include <iostream>
#include <cmath>

#include "cuda_core.cuh"
#include "sunshine_hours.h"
#include "config.h"

using namespace std;

SunshineHours::SunshineHours(const Raster &dem, const float padding_degree, const int day_of_year, const int time_step)
{
    this->dem = dem;
    this->padding_degree = padding_degree;
    this->day_of_year = day_of_year;
    this->time_step = time_step;

    index_range = calculateIndexRange();

    result = Raster(index_range.row_to - index_range.row_from, index_range.col_to - index_range.col_from);
    result.projection = dem.projection;
    
    // set the geo transform of the result raster
    result.geo_transform[0] = dem.geo_transform[0] + index_range.col_from * dem.geo_transform[1];  // longtitude of top left corner
    result.geo_transform[1] = dem.geo_transform[1];  // pixel width
    result.geo_transform[2] = dem.geo_transform[2];  // rotation
    result.geo_transform[3] = dem.geo_transform[3] + index_range.row_from * dem.geo_transform[5];  // latitude of top left corner
    result.geo_transform[4] = dem.geo_transform[4];  // rotation
    result.geo_transform[5] = dem.geo_transform[5];  // pixel height
    
    result.no_data_value = NAN;
}

IndexRange SunshineHours::calculateIndexRange()
{
    IndexRange range;
    
    // calculate the boundary of the original DEM
    double dem_min_x = dem.geo_transform[0];  // longitude of top left corner
    double dem_max_x = dem.geo_transform[0] + dem.cols * dem.geo_transform[1];  // longitude of bottom right corner
    double dem_max_y = dem.geo_transform[3];  // latitude of top left corner
    double dem_min_y = dem.geo_transform[3] + dem.rows * dem.geo_transform[5];  // latitude of bottom right corner
    
    // calculate the boundary of the result raster (shrink by padding_degree)
    double result_min_x = dem_min_x + padding_degree;
    double result_max_x = dem_max_x - padding_degree;
    double result_max_y = dem_max_y - padding_degree;
    double result_min_y = dem_min_y + padding_degree;
    
    // calculate the index range
    range.col_from = static_cast<int>((result_min_x - dem_min_x) / dem.geo_transform[1]-1);
    range.col_to = static_cast<int>((result_max_x - dem_min_x) / dem.geo_transform[1]+1);
    range.row_from = static_cast<int>((result_max_y - dem_max_y) / dem.geo_transform[5]-1); // note that the y axis is inverted
    range.row_to = static_cast<int>((result_min_y - dem_max_y) / dem.geo_transform[5]+1);
    
    // ensure the index range is valid
    range.col_from = max(range.col_from, size_t(0));
    range.col_to = min(dem.cols, range.col_to);
    range.row_from = max(range.row_from, size_t(0));
    range.row_to = min(dem.rows, range.row_to);

    // debug output
    // cout << "DEM bounds: (" << dem_min_x << ", " << dem_min_y << ") to (" << dem_max_x << ", " << dem_max_y << ")" << endl;
    // cout << "Result bounds: (" << result_min_x << ", " << result_min_y << ") to (" << result_max_x << ", " << result_max_y << ")" << endl;
    // cout << "Index range: col[" << range.col_from << ", " << range.col_to << "], row[" << range.row_from << ", " << range.row_to << "]" << endl;
    // cout << "Result size: " << (range.col_to - range.col_from) << " x " << (range.row_to - range.row_from) << endl;
    
    return range;
}

void SunshineHours::calculate()
{
    CalculateSunshineHoursCuda calculateSunshineHoursCuda(result, dem, index_range, day_of_year, time_step, CUDA_DEVICE_ID);
    calculateSunshineHoursCuda.calculate();
}

Raster& SunshineHours::getResult()
{
    return result;
}

void SunshineHours::save(const string &file_path)
{
    result.save(file_path);
}

void SunshineHours::printCertainResult(int i, int j)
{
    double x, y;
    x = result.geo_transform[0] + j * result.geo_transform[1] + i * result.geo_transform[2] + 0.5 * result.geo_transform[1] + 0.5 * result.geo_transform[2];
    y = result.geo_transform[3] + j * result.geo_transform[4] + i * result.geo_transform[5] + 0.5 * result.geo_transform[4] + 0.5 * result.geo_transform[5];
    cout << "result.data[" << i << "][" << j << "] = " << result.data.get()[i * result.cols + j] << " at (" << x << ", " << y << ")" << endl;
}

void SunshineHours::printFirstNResult(int n)
{
    int count = 0;
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            if (!isnan(result.data.get()[i * result.cols + j]) && abs(result.data.get()[i * result.cols + j]) > 0.0001)
            {
                printCertainResult(i, j);
                count++;
            }
            if (count == n)
            {
                break;
            }
        }
        if (count == n)
        {
            break;
        }
    }
}
