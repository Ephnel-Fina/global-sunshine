#include "raster.h"
#include "sunshine_hours.h"
#include "timer.h"
#include "config.h"
#include <fstream>
#include <sstream>
#include <unordered_map>

// ===== 全局配置参数定义 =====
int BATCH         = 16;           // 默认值，可被配置文件覆盖
int NUM_STREAMS   = 16;
dim3 BLOCK_SIZE   = dim3(64, 4);
int CUDA_DEVICE_ID = 0;

// ===== Sunshine 计算相关 (默认值) =====
int DAY_OF_YEAR     = 15;
int TIME_STEP       = 5;
float PADDING_DEGREE = 1.0f;
// ===== 结束 =====

void load_config(const std::string& path)
{
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[Warn] unable to open config file " << path << ", fallback to default parameters\n";
        return;
    }
    std::string line;
    std::unordered_map<std::string, std::string> kv;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0]=='#') continue;
        size_t eq = line.find('=');
        if (eq==std::string::npos) continue;
        auto trim = [](std::string s){
            size_t b = s.find_first_not_of(" \t\r\n");
            size_t e = s.find_last_not_of(" \t\r\n");
            if (b==std::string::npos) return std::string();
            return s.substr(b, e-b+1);
        };
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq+1));
        kv[key] = val;
    }
    auto get_int = [&](const std::string& k, int &var){ if(kv.count(k)) var = std::stoi(kv[k]); };
    auto get_float = [&](const std::string& k, float &var){ if(kv.count(k)) var = std::stof(kv[k]); };

    get_int("BATCH", BATCH);
    get_int("NUM_STREAMS", NUM_STREAMS);
    if(kv.count("BLOCK_SIZE_X") && kv.count("BLOCK_SIZE_Y"))
        BLOCK_SIZE = dim3(std::stoi(kv["BLOCK_SIZE_X"]), std::stoi(kv["BLOCK_SIZE_Y"]));
    get_int("CUDA_DEVICE_ID", CUDA_DEVICE_ID);

    get_int("DAY_OF_YEAR", DAY_OF_YEAR);
    get_int("TIME_STEP", TIME_STEP);
    get_float("PADDING_DEGREE", PADDING_DEGREE);
}

int main()
{
    // 读取配置
    load_config("/home/ftx/yuanshen/shi/src/config.txt");

    timer.tick("total");

    timer.tick("Reading DEM");
    // string dem_file_path = "/home/lq/sunshine_hours/data/N30E080_FABDEM_V1-2.tif";
    // string result_file_path = "./result_N30E080.tif";
    string dem_file_path = "/home/ftx/yuanshen/data/验证样区/wgs84/赤道组/S10E030_FABDEM_V1-2.tif";
    string result_file_path = "./result_dem30_100.tif";
    Raster dem(dem_file_path);
    dem.printInfo();
    timer.tock();
 
    timer.tick("SunshineHours");
    SunshineHours sunshine_hours(dem, PADDING_DEGREE, DAY_OF_YEAR, TIME_STEP);
    timer.tick("SunshineHours::calculate");
    sunshine_hours.calculate();
    timer.tock();
    timer.tock();
    
    timer.tick("Save result");
    Raster result = sunshine_hours.getResult();
    result.save(result_file_path);
    timer.tock();
    
    timer.tock();
    timer.print_records();

    // result.print_info();

    return 0;
}