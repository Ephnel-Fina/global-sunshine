#include "raster.h"
#include <iostream>
#include <filesystem>
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <algorithm>

#include "timer.h"

using namespace std;

Raster::Raster(const string &file_path)
{
    GDALAllRegister();

    timer.tick("Read raster file");

    this->file_path = file_path;

    GDALDataset *p_dataset = (GDALDataset *)GDALOpen(file_path.c_str(), GA_ReadOnly);
    if (p_dataset == NULL)
    {
        cerr << "ERROR: Cannot open file: \"" << file_path << "\"!" << endl;
        return;
    }

    rows = p_dataset->GetRasterYSize();
    cols = p_dataset->GetRasterXSize();
    int nBands = p_dataset->GetRasterCount();
    // cout << "Size: " << cols << " x " << rows << " x " << nBands << endl;

    projection = p_dataset->GetProjectionRef();
    p_dataset->GetGeoTransform(geo_transform);
    no_data_value = p_dataset->GetRasterBand(1)->GetNoDataValue();

    if (geo_transform[2] != 0 || geo_transform[4] != 0)
    {
        cerr << "ERROR: Not support rotation!" << endl;
        return;
    }

    if (abs(geo_transform[1] + geo_transform[5]) > 1e-6)
    {
        cerr << "ERROR: Not support different resolution in x and y direction!" << endl;
        return;
    }

    data = shared_ptr<float>(new float[size_t(rows) * size_t(cols)], default_delete<float[]>());

    CPLErr err = p_dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, cols, rows, data.get(), cols, rows, GDT_Float32, 0, 0);

    for (size_t i = 0; i < size_t(rows) * size_t(cols); i++)
    {
        if (data.get()[i] == no_data_value)
        {
            data.get()[i] = -NAN;
            continue;
        }
        max_value = max(max_value, data.get()[i]);
        min_value = min(min_value, data.get()[i]);
    }

    GDALClose(p_dataset);

    timer.tock();
}

Raster::Raster(size_t rows, size_t cols)
{
    this->rows = rows;
    this->cols = cols;
    data = shared_ptr<float>(new float[size_t(rows) * size_t(cols)], default_delete<float[]>());
}

void Raster::copyGeoTransformFrom(const Raster &raster)
{
    for (int i = 0; i < 6; i++)
    {
        geo_transform[i] = raster.geo_transform[i];
    }
}

size_t Raster::size() const
{
    return size_t(rows) * size_t(cols);
}

void Raster::printInfo()
{
    cout << "file_path: " << file_path << endl;
    cout << "rows: " << rows << " cols: " << cols << endl;
    cout << "projection: " << projection << endl;
    cout << "geo_transform: ";
    for (int i = 0; i < 6; i++)
    {
        cout << geo_transform[i] << " ";
    }
    cout << endl;
    cout << "no_data_value: " << no_data_value << endl;
    cout << "max_value: " << max_value << " min_value: " << min_value << endl;
}

void Raster::save(const string &file_path)
{
    filesystem::path save_file_path(file_path);
    filesystem::path save_dir = save_file_path.parent_path();
    if (!filesystem::exists(save_dir))
    {
        filesystem::create_directories(save_dir);
    }

    GDALDriver *p_driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!p_driver)
    {
        cerr << "ERROR: Cannot get driver!" << endl;
        return;
    }

    GDALDataset *p_dataset = p_driver->Create(file_path.c_str(), cols, rows, 1, GDT_Float32, NULL);
    if (!p_dataset)
    {
        cerr << "ERROR: Cannot create file: \"" << file_path << "\"!" << endl;
        return;
    }

    GDALSetProjection(p_dataset, projection.c_str());
    p_dataset->SetGeoTransform(geo_transform);

    GDALRasterBand *p_band = p_dataset->GetRasterBand(1);
    p_band->SetNoDataValue(no_data_value);
    
    CPLErr err = p_band->RasterIO(GF_Write, 0, 0, cols, rows, data.get(), cols, rows, GDT_Float32, 0, 0);

    GDALClose(p_dataset);
    return;
}
