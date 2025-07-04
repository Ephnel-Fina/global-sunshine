#include <iostream>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_core.cuh"
#include <iostream>
#include "timer.h"
#include <math_constants.h>
#include <algorithm>
#include "config.h"

#define CHECK_CUDA(call)                                \
    do                                                  \
    {                                                   \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)

__host__
double calculateSolarDeclination(const int day_of_year)
{
    double tau = 2 * M_PI * (day_of_year - 1) / 365.2422;
    double delta = 0.006894
                    - 0.399512 * cos(tau)
                    + 0.072075 * sin(tau)
                    - 0.006799 * cos(2 * tau)
                    + 0.000896 * sin(2 * tau)
                    - 0.002689 * cos(3 * tau)
                    + 0.001516 * sin(3 * tau);
    return delta;
}

__host__
void calculateSolarHourAngle(const double phi, const double delta, const int time_step, double &delta_omega, double &omega_r, double &omega_s, int &n_steps)
{
    omega_s = acos(-tan(delta) * tan(phi));
    omega_r = -omega_s;

    n_steps = int((omega_s - omega_r) / M_PI * 180) / 15.0 * 60.0 / time_step;
    delta_omega = (omega_s - omega_r) / n_steps;
}

__global__
void calculateSolarAltitudeAndAzimuthBatchKernel(
    const int    batch_size,
    const int    half_n_steps,
    const int    full_n_steps,
    const double delta,
    const int    time_step,
    const size_t row_base,
    const double* __restrict__ geo_transform,
    double* __restrict__ h_array,      // batch_size * full_n_steps
    double* __restrict__ A_array,      // batch_size * full_n_steps
    int*    __restrict__ n_steps_arr)  // batch_size
{
    const int row  = blockIdx.y;               // 每个block.y处理一行，blockDim.y =1
    const int step = blockIdx.x * blockDim.x + threadIdx.x;  // 时间步索引 (上午)

    if (row >= batch_size || step >= half_n_steps) return;

    // 通过 geoTransform 计算该行的纬度 (弧度)
    double lat_deg = geo_transform[3] + (double)(row_base + row) * geo_transform[5];
    double phi = lat_deg * M_PI / 180.0;

    // 计算该行的太阳时角相关参数
    double omega_s = acos(-tan(delta) * tan(phi));
    double omega_r = -omega_s;
    double tmp_deg = (omega_s - omega_r) * 180.0 / M_PI; // 角度差
    double hours   = tmp_deg / 15.0;                     // 小时差
    int    n_steps = int(hours * 60.0 / time_step);
    if(n_steps < 1) n_steps = 1; // 防止 0
    double delta_omega = (omega_s - omega_r) / n_steps;

    // 保存 n_steps (只在 step==0 写一次)
    if (step == 0) n_steps_arr[row] = n_steps;

    size_t idx_am  = (size_t)row * full_n_steps + step;      // 上午索引
    size_t idx_pm  = (size_t)row * full_n_steps + (n_steps-1-step); // 对称位

    h_array[idx_am] = asin(
        sin(phi) * sin(delta) + cos(phi) * cos(delta) * cos(omega_r + delta_omega * double(step)));

    double A_i = acos(cos(phi) * sin(omega_r + delta_omega * double(step)) / cos(h_array[idx_am]));
    A_i = (omega_r + delta_omega * double(step) < 0.0) ? (CUDART_PI - A_i) : (CUDART_PI + A_i);

    h_array[idx_pm] = h_array[idx_am];           // 高度角对称

    double A_sym = 2.0 * CUDART_PI - A_i; // 南向对称
    A_array[idx_am] = A_i;
    A_array[idx_pm] = A_sym;
}

__global__
void calculateRayAndStepHeightChangesBatchKernel(
    const int    batch_size,
    const int    max_n_steps,
    const int    max_search_steps,
    const size_t row_base,
    const double* __restrict__ geo_transform,
    const float  max_relative_height,
    const double* __restrict__ h_array,
    const double* __restrict__ A_array,
    const int*   __restrict__ n_steps_arr,

    int*   __restrict__ search_steps_flat, // batch_size * max_n_steps
    int*   __restrict__ dxs_flat,          // batch_size * max_n_steps * max_search_steps
    int*   __restrict__ dys_flat,
    float* __restrict__ height_changes_flat)
{
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int step = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;
    int n_steps = n_steps_arr[row];
    int half_n = (n_steps + 1) / 2;
    if (step >= half_n) return;  // 仅上午计算

    // geo 信息
    double lat_deg = geo_transform[3] + (double)(row_base + row) * geo_transform[5];
    double phi = lat_deg * M_PI / 180.0;

    double meters_per_deg_lat = 111132.92 - 559.82 * cos(2 * phi) + 1.175 * cos(4 * phi);
    double meters_per_deg_lon = 111412.84 * cos(phi) - 93.5 * cos(3 * phi);
    double meters_per_pixel_x = meters_per_deg_lon * geo_transform[1];
    double meters_per_pixel_y = -meters_per_deg_lat * geo_transform[5];
    double pixel_x_rad = geo_transform[1] * M_PI / 180.0;
    double pixel_y_rad = geo_transform[5] * M_PI / 180.0;

    const size_t base_idx = (size_t)row * max_n_steps + step;
    double h_i = h_array[base_idx];
    double A_i = A_array[base_idx];

    // 与旧版逻辑相同
    double delta_x_i = sin(A_i);
    double delta_y_i = -cos(A_i);
    double step_dist = sqrt((meters_per_pixel_x * delta_x_i) * (meters_per_pixel_x * delta_x_i) +
                            (meters_per_pixel_y * delta_y_i) * (meters_per_pixel_y * delta_y_i));

    double tan_h_i   = tan(h_i);
    double delta_h_i = tan_h_i * step_dist;
    int search_candidate = int(max_relative_height / delta_h_i);
    int search_step  = (search_candidate < max_search_steps) ? search_candidate : max_search_steps;

    // 写上午数据
    search_steps_flat[base_idx] = search_step;

    int dx_pix = int(delta_x_i * search_step);
    int dy_pix = int(delta_y_i * search_step);
    int dx = abs(dx_pix), dy = abs(dy_pix);
    int sx = dx_pix > 0 ? 1 : -1;
    int sy = dy_pix > 0 ? 1 : -1;
    int err = dx - dy;
    int x = 0, y = 0;

    int *dxs = dxs_flat + base_idx * max_search_steps;
    int *dys = dys_flat + base_idx * max_search_steps;
    float *height_changes = height_changes_flat + base_idx * max_search_steps;

    for (int k = 0; k < search_step; ++k) {
        int e2 = err * 2;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }

        dxs[k] = x;
        dys[k] = y;

        double delta_phi    = y * pixel_y_rad;
        double delta_lambda = x * pixel_x_rad;
        double sp2 = sin(delta_phi * 0.5);
        double sl2 = sin(delta_lambda * 0.5);
        double a   = sp2 * sp2 + cos(phi) * cos(phi + delta_phi) * sl2 * sl2;
        double c   = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));

        height_changes[k] = float((x * meters_per_pixel_x * delta_x_i +
                                   y * meters_per_pixel_y * delta_y_i) * tan_h_i +
                                  6371000.0 * (1.0 - cos(c)));
    }

    // ---- 写对称位（下午） ----
    size_t sym_base_idx = (size_t)row * max_n_steps + (n_steps - 1 - step);
    search_steps_flat[sym_base_idx] = search_step;

    int *dxs_sym = dxs_flat + sym_base_idx * max_search_steps;
    int *dys_sym = dys_flat + sym_base_idx * max_search_steps;
    float *hchg_sym = height_changes_flat + sym_base_idx * max_search_steps;

    for (int k = 0; k < search_step; ++k) {
        dxs_sym[k]  = -dxs[k];   // x 方向取反
        dys_sym[k]  =  dys[k];   // y 相同
        hchg_sym[k] =  height_changes[k];
    }
}

__global__
void calculateSunshineHoursBatchKernel(
    const int    batch_size,
    const size_t row_base,
    const size_t col0_in_dem,
    const size_t cols_per_dem_row,
    const size_t cols_per_result_row,
    const int    time_step,
    const int    max_n_steps,
    const int*   __restrict__ n_steps_arr,
    const int*   __restrict__ dxs,
    const int*   __restrict__ dys,
    const float* __restrict__ height_changes,
    const int*   __restrict__ search_steps,
    const int    max_search_steps,
    cudaTextureObject_t tex_dem_obj,
    float*        d_result_batch) // batch_size * cols_per_result_row
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size || col >= cols_per_result_row) return;

    size_t row_in_dem = row_base + row;
    float local_height = tex2D<float>(tex_dem_obj, col0_in_dem + col + 0.5, row_in_dem + 0.5);
    float result_time = 0.0f;

    int n_steps = n_steps_arr[row];
    for (int j = 0; j < n_steps; ++j) {
        int search_step = search_steps[row * max_n_steps + j];
        bool is_visible = true;
        for (int k = 0; k < search_step; ++k) {
            int dx = dxs[(row * max_n_steps + j) * max_search_steps + k];
            int dy = dys[(row * max_n_steps + j) * max_search_steps + k];
            float step_height_change = height_changes[(row * max_n_steps + j) * max_search_steps + k];
            float obstacle_height = tex2D<float>(tex_dem_obj, col0_in_dem + col + dx + 0.5, row_in_dem + dy + 0.5);
            if (obstacle_height > local_height + step_height_change) { is_visible = false; break; }
        }
        if (is_visible) result_time += time_step;
    }
    d_result_batch[row * cols_per_result_row + col] = result_time;
}


CalculateSunshineHoursCuda::CalculateSunshineHoursCuda(Raster &result, const Raster &dem, const IndexRange &target_index_range, const int day_of_year, const int time_step, const int cuda_device_id)
    : result(result), dem(dem), target_index_range(target_index_range), day_of_year(day_of_year), time_step(time_step), cuda_device_id(cuda_device_id)
{
    // set cuda device
    CHECK_CUDA(cudaSetDevice(cuda_device_id));

    // debug output
    std::cout << "cuda_device_id: " << cuda_device_id << std::endl;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, cuda_device_id));
    std::cout << "asyncEngineCount: " << prop.asyncEngineCount << std::endl;

    // create texture object for the dem
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); // create channel description for the texture
    CHECK_CUDA(cudaMallocArray(&d_dem_array, &desc, dem.cols, dem.rows)); // allocate device memory for the texture
    CHECK_CUDA(cudaMemcpy2DToArray(d_dem_array, 0, 0, dem.data.get(), dem.cols * sizeof(float), dem.cols * sizeof(float), dem.rows, cudaMemcpyHostToDevice)); // copy data to the texture

    cudaResourceDesc resDesc = {}; // create resource description for the texture
    resDesc.resType = cudaResourceTypeArray; // set resource type to array
    resDesc.res.array.array = d_dem_array; // set array to the texture

    cudaTextureDesc texDesc = {}; // create texture description for the texture
    texDesc.addressMode[0] = cudaAddressModeClamp; // when the texture is out of U bounds, clamp the value
    texDesc.addressMode[1] = cudaAddressModeClamp; // when the texture is out of V bounds, clamp the value
    texDesc.filterMode     = cudaFilterModePoint; // use point sampling
    texDesc.readMode       = cudaReadModeElementType; // read the texture as element type
    texDesc.normalizedCoords = 0; // use actual coordinates

    CHECK_CUDA(cudaCreateTextureObject(&tex_dem_obj, &resDesc, &texDesc, nullptr)); // create texture object

    // // debug output
    // std::cout << "tex_dem_obj created" << std::endl;

    // allocate device memory for the geo_transform
    CHECK_CUDA(cudaMalloc(&d_geo_transform, 6 * sizeof(double)));

    // copy geo_transform to device
    CHECK_CUDA(cudaMemcpy(d_geo_transform, dem.geo_transform, 6 * sizeof(double), cudaMemcpyHostToDevice));

    // // debug output
    // std::cout << "d_geo_transform created" << std::endl;

    // calculate grid size
    grid_size = dim3((result.cols + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                     (result.rows + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
}

CalculateSunshineHoursCuda::~CalculateSunshineHoursCuda()
{
    CHECK_CUDA(cudaDestroyTextureObject(tex_dem_obj)); // unbind the texture
    CHECK_CUDA(cudaFreeArray(d_dem_array));
    CHECK_CUDA(cudaFree(d_geo_transform));
}

void CalculateSunshineHoursCuda::calculate()
{
    const int batch = BATCH;              // 单次处理的行数
    // debug output
    std::cout << "calculate started" << std::endl;

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    std::vector<float*> d_result_row_buffers(NUM_STREAMS);
    std::vector<float*> h_result_row_buffers(NUM_STREAMS);

    // struct for async copy callback
    struct CopyCtx {
        float* dst;
        float* src;
        size_t bytes;
    };

    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        CHECK_CUDA(cudaStreamCreate(&streams[s]));
        CHECK_CUDA(cudaMalloc(&d_result_row_buffers[s], batch * result.cols * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_result_row_buffers[s], batch * result.cols * sizeof(float)));
    }

    // calculate solar declination
    double delta = calculateSolarDeclination(day_of_year);

    // debug output
    std::cout << "delta: " << delta << std::endl;

    int max_search_steps = int(sqrt(double(target_index_range.row_from) * target_index_range.row_from + double(target_index_range.col_from) * target_index_range.col_from) + 1);
    int max_n_steps = 24 * 60 / time_step;

    // debug output
    std::cout << "dem.cols: " << dem.cols << std::endl;
    std::cout << "dem.rows: " << dem.rows << std::endl;
    std::cout << "max_search_steps: " << max_search_steps << std::endl;

    std::vector<int*> h_dxs(NUM_STREAMS);
    std::vector<int*> h_dys(NUM_STREAMS);
    std::vector<float*> h_height_changes(NUM_STREAMS);
    std::vector<int*> h_search_steps(NUM_STREAMS);
    std::vector<int*> d_dxs(NUM_STREAMS);
    std::vector<int*> d_dys(NUM_STREAMS);
    std::vector<float*> d_height_changes(NUM_STREAMS);
    std::vector<int*> d_search_steps(NUM_STREAMS);
    std::vector<int*> d_n_steps(NUM_STREAMS);     
    std::vector<double*> d_h_array(NUM_STREAMS);
    std::vector<double*> d_A_array(NUM_STREAMS);

    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        CHECK_CUDA(cudaMallocHost(&h_dxs[s],     batch * max_search_steps * max_n_steps * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&h_dys[s],     batch * max_search_steps * max_n_steps * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&h_height_changes[s], batch * max_search_steps * max_n_steps * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_search_steps[s],   batch * max_n_steps * sizeof(int)));

        CHECK_CUDA(cudaMalloc(&d_dxs[s],         batch * max_search_steps * max_n_steps * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_dys[s],         batch * max_search_steps * max_n_steps * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_height_changes[s], batch * max_search_steps * max_n_steps * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_search_steps[s],   batch * max_n_steps * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_n_steps[s],        batch * sizeof(int)));

        CHECK_CUDA(cudaMalloc(&d_h_array[s], batch * max_n_steps * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_A_array[s], batch * max_n_steps * sizeof(double)));
    }

    // =================================================================================
    // CALCULATE SUNSHINE HOURS BY BATCH
    size_t total_rows = result.rows;
    for (size_t row_base = 0; row_base < total_rows; row_base += batch)
    {
        size_t cur_batch = std::min<size_t>(batch, total_rows - row_base);
        size_t s = (row_base / batch) % NUM_STREAMS;

        dim3 grid_size_batch((result.cols + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                             (cur_batch + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);

        // 1. 太阳高度角 & 方位角
        calculateSolarAltitudeAndAzimuthBatchKernel<<<grid_size_batch, BLOCK_SIZE, 0, streams[s]>>>(
            (int)cur_batch, max_n_steps, max_n_steps, delta, time_step,
            row_base + target_index_range.row_from, d_geo_transform,
            d_h_array[s], d_A_array[s], d_n_steps[s]);

        // 2. 视线 & 高程差
        calculateRayAndStepHeightChangesBatchKernel<<<grid_size_batch, BLOCK_SIZE, 0, streams[s]>>>(
            (int)cur_batch, max_n_steps, max_search_steps,
            row_base + target_index_range.row_from, d_geo_transform,
            dem.max_value - dem.min_value,
            d_h_array[s], d_A_array[s], d_n_steps[s],
            d_search_steps[s], d_dxs[s], d_dys[s], d_height_changes[s]);

        // 3. 累计日照时长
        calculateSunshineHoursBatchKernel<<<grid_size_batch, BLOCK_SIZE, 0, streams[s]>>>(
            (int)cur_batch,
            row_base + target_index_range.row_from,
            target_index_range.col_from,
            dem.cols, result.cols, time_step,
            max_n_steps,
            d_n_steps[s], d_dxs[s], d_dys[s],
            d_height_changes[s], d_search_steps[s], max_search_steps,
            tex_dem_obj, d_result_row_buffers[s]);

        // 将结果异步拷回主机
        CHECK_CUDA(cudaMemcpyAsync(h_result_row_buffers[s], d_result_row_buffers[s],
                                   cur_batch * result.cols * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));

        // 回调归并到最终结果
        CopyCtx* ctx = new CopyCtx{
            result.data.get() + row_base * result.cols,
            h_result_row_buffers[s],
            cur_batch * result.cols * sizeof(float)
        };
        CHECK_CUDA(cudaLaunchHostFunc(streams[s], [](void* userData) {
            CopyCtx* c = static_cast<CopyCtx*>(userData);
            std::memcpy(c->dst, c->src, c->bytes);
            delete c;
        }, ctx));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    // =================================================================================

    // free memory
    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        CHECK_CUDA(cudaFreeHost(h_dxs[s]));
        CHECK_CUDA(cudaFreeHost(h_dys[s]));
        CHECK_CUDA(cudaFreeHost(h_height_changes[s]));
        CHECK_CUDA(cudaFreeHost(h_search_steps[s]));
        CHECK_CUDA(cudaFree(d_dxs[s]));
        CHECK_CUDA(cudaFree(d_dys[s]));
        CHECK_CUDA(cudaFree(d_height_changes[s]));
        CHECK_CUDA(cudaFree(d_search_steps[s]));
        CHECK_CUDA(cudaFree(d_n_steps[s]));
        CHECK_CUDA(cudaFree(d_h_array[s]));
        CHECK_CUDA(cudaFree(d_A_array[s]));
    }

    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        CHECK_CUDA(cudaStreamDestroy(streams[s]));
        CHECK_CUDA(cudaFree(d_result_row_buffers[s]));
        CHECK_CUDA(cudaFreeHost(h_result_row_buffers[s]));
    }
}