from osgeo import gdal, osr
import numpy as np
import cupy as cp
import multiprocessing
from dask import distributed
import dask.array as da
from dask import delayed
import dask
import time
import os
import logging
import calendar
import itertools
import sys
import re

# 配置gdal过滤器
gdal.UseExceptions()
osr.UseExceptions()
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"

# 配置日志过滤器
class NoWorkerCloseFilter(logging.Filter):
    def filter(self, record):
        return (
            "Closing Nanny" not in record.getMessage()
            and "Lost all workers" not in record.getMessage()
            and "Scheduler closing due to unknown reason..." not in record.getMessage()
            and "Scheduler closing all comms" not in record.getMessage()
            and "Remove worker" not in record.getMessage()
            and "Nanny asking worker" not in record.getMessage()
        )


distributed.nanny.logger.addFilter(NoWorkerCloseFilter())
distributed.scheduler.logger.addFilter(NoWorkerCloseFilter())


class DEMProcessor:

    def __init__(self, dem_file: str, chunk_size: int = 1024):
        """
        参数:
        dem_file -- DEM文件路径
        chunk_size -- 分块大小，默认512x512
        """
        self.dem_file = dem_file
        self.chunk_size = chunk_size

        # 初始化元数据
        self._load_metadata()

    def _load_metadata(self) -> None:
        """加载DEM文件的元数据"""
        ds = gdal.Open(self.dem_file)
        band = ds.GetRasterBand(1)

        # 基础元数据
        self.gt = ds.GetGeoTransform()  # 地理变换参数
        self.rows = band.YSize  # 行数
        self.cols = band.XSize  # 列数
        self.nodata = band.GetNoDataValue()  # 无效值

        # 坐标系定义
        self.src_wkt = ds.GetProjection()  # 原始坐标系
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.target_wkt = target_srs.ExportToWkt()

        ds = None  # 释放资源

    def _generate_chunk_params(self) -> list[tuple]:
        """生成分块参数列表"""
        params = []
        for y in range(0, self.rows, self.chunk_size):
            for x in range(0, self.cols, self.chunk_size):
                chunk_rows = min(self.chunk_size, self.rows - y)
                chunk_cols = min(self.chunk_size, self.cols - x)
                params.append((y, x, chunk_rows, chunk_cols))
        return params

    @staticmethod
    def _group_blocks(blocks: list, blocks_per_row: int) -> list[list]:
        """将块按原始行顺序分组"""
        return [
            blocks[i : i + blocks_per_row]
            for i in range(0, len(blocks), blocks_per_row)
        ]

    @delayed
    def _read_chunk_block(self, param: tuple) -> np.ndarray:
        """延迟读取单个数据块"""
        y_start, x_start, chunk_rows, chunk_cols = param

        ds = gdal.Open(self.dem_file)
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray(x_start, y_start, chunk_cols, chunk_rows)

        # 处理nodata并转换为float
        processed = np.where(data == self.nodata, np.nan, data.astype(np.float32))
        processed = np.round(processed, decimals=2)
        return processed  # 返回二维数组

    def to_dask_array(self) -> da.Array:
        """
        将DEM文件转换为包含高程值的Dask Array
        返回二维数组结构： 行 x 列
        仅包含：高程值
        """

        # 生成分块参数
        params = self._generate_chunk_params()

        # 创建延迟块列表
        blocks = [
            da.from_delayed(
                self._read_chunk_block(p), shape=(p[2], p[3]), dtype=np.float32
            )
            for p in params
        ]

        # 按原始行列结构重组
        blocks_per_row = (self.cols + self.chunk_size - 1) // self.chunk_size
        grouped = [
            blocks[i : i + blocks_per_row]
            for i in range(0, len(blocks), blocks_per_row)
        ]

        # 水平拼接同行块
        row_arrays = [da.concatenate(row, axis=1) for row in grouped]  # 沿列方向拼接

        # 垂直拼接所有行
        full_array = da.concatenate(row_arrays, axis=0)

        # 转换为numpy数组
        return full_array

    #从dem中心取出指定大小的dem  
    def extract_center_region(self, size_deg: float) -> tuple[np.ndarray, tuple]:
        """
        提取以中心为基准的size_deg x size_deg区域(单位:度)
        
        参数:
        size_deg -- 区域边长(单位为度,例如1表示1°x1°区域)

        返回:
        (子区域数据, 子区域的地理变换参数)
        """
        # 解码地理变换参数
        origin_x, pixel_width, _, origin_y, _, pixel_height = self.gt

        # 总影像的范围
        total_width_deg = self.cols * pixel_width
        total_height_deg = abs(self.rows * pixel_height)

        # DEM中心点坐标（左上角为起点）
        center_lon = origin_x + total_width_deg / 2
        center_lat = origin_y - total_height_deg / 2  # 注意Y方向是负的

        # 子区域左上角坐标
        half_size = size_deg / 2
        ulx = center_lon - half_size
        uly = center_lat + half_size

        # 转换为像素索引
        x_offset = int((ulx - origin_x) / pixel_width)
        y_offset = int((origin_y - uly) / abs(pixel_height))
        x_size = int(size_deg / pixel_width)
        y_size = int(size_deg / abs(pixel_height))

        # 读取子区域
        ds = gdal.Open(self.dem_file)
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray(x_offset, y_offset, x_size, y_size)
        data = np.where(data == self.nodata, np.nan, data.astype(np.float32))
        data = np.round(data, decimals=2)

        # 构造新的GeoTransform
        new_gt = (
            origin_x + x_offset * pixel_width,  # 左上角X
            pixel_width,
            0,
            origin_y + y_offset * pixel_height,  # 左上角Y
            0,
            pixel_height
        )

        ds = None  # 释放资源
        return data, new_gt
    
    #计算dem1在dem2中所处位置
    @staticmethod
    def get_geo_metadata(gt, data):
        """从 DEMProcessor 获取元数据"""
        return {
            "x_min": gt[0],
            "y_max": gt[3],
            "dx": gt[1],
            "dy": abs(gt[5]),
            "rows": data.shape[0],
            "cols": data.shape[1],
        }

    @staticmethod
    def calculate_overlap(dem_gt, dem, obstacle_gt, obstacle):
        """计算 dem1 在 dem2 中的行列范围"""
        dem_meta = DEMProcessor.get_geo_metadata(dem_gt, dem)
        obstacle_meta = DEMProcessor.get_geo_metadata(obstacle_gt, obstacle)

        # 计算 dem1 的右下角坐标
        x_max1 = dem_meta["x_min"] + dem_meta["cols"] * dem_meta["dx"]
        y_min1 = dem_meta["y_max"] - dem_meta["rows"] * dem_meta["dy"]

        # 计算行列偏移量
        col_start = int(
            (dem_meta["x_min"] - obstacle_meta["x_min"]) / obstacle_meta["dx"]
        )
        row_start = int(
            (obstacle_meta["y_max"] - dem_meta["y_max"]) / obstacle_meta["dy"]
        )

        # 计算结束位置
        col_end = int((x_max1 - obstacle_meta["x_min"]) / obstacle_meta["dx"])
        row_end = int((obstacle_meta["y_max"] - y_min1) / obstacle_meta["dy"])

        # 确保在 dem2 范围内
        col_start = max(0, col_start)
        row_start = max(0, row_start)
        col_end = min(obstacle_meta["cols"], col_end)
        row_end = min(obstacle_meta["rows"], row_end)

        return (row_start, row_end, col_start, col_end)

    def analyze_memory(self):
        """内存分析"""
        demo = self.to_dask_array()
        print("内存分析报告:")
        print(f"数组维度: {demo.shape}")
        print(f"分块结构: {demo.chunks}")
        print(f"单元素大小: {demo.dtype.itemsize} bytes")
        print(f"理论内存总量: {demo.nbytes / 1024**3:.3f} GB")

    @staticmethod
    def analyze_nan(arr):
        nan_count = np.count_nonzero(np.isnan(arr))  # 统计NaN数量
        total = arr.size  # 总元素数
        ratio = nan_count / total  # 计算占比
        print(f"NaN的占比为: {ratio:.2%}")

    def get_geo_transform(self):
        return self.gt

    def get_geo_wkt(self):
        return self.src_wkt

    
class SolarAsg:
    """
    输入数组dask.array,天数,时间步长(分钟)，分块大小
    返回（太阳参数矩阵[太阳时角, 高度角, 方位角]，日落时角，太阳时角步长）
    """

    def __init__(self, N_day, time_step, chunk_size=1000):
        self.N_day = N_day
        self.time_step = time_step
        self.chunk_size = chunk_size
        
        # 初始化缓存标志
        self._calculated = {
            'solar_declination': False,
            'sunrise_sunset': False,
            'time_params': False,
            'elevation': False,
            'azimuth': False
        }

    @staticmethod
    def safe_acos(value):
        return cp.arccos(cp.clip(value, -1.0, 1.0))

    @staticmethod
    def safe_asin(value):
        return cp.arcsin(cp.clip(value, -1.0, 1.0))

    # 太阳赤纬角
    def calculate_solar_declination(self):
        if self._calculated['solar_declination']:
            return self.solar_declination
            
        t = self.N_day - 1
        theta = 2 * cp.pi * t / 365.2422
        self.solar_declination = (
            0.006894
            - 0.399512 * cp.cos(theta)
            + 0.072075 * cp.sin(theta)
            - 0.006799 * cp.cos(2 * theta)
            + 0.000896 * cp.sin(2 * theta)
            - 0.002689 * cp.cos(3 * theta)
            + 0.001516 * cp.sin(3 * theta)
        )
        self._calculated['solar_declination'] = True
        return self.solar_declination


class Latprocessor:
    def __init__(self,file_path):
        self.file_path=file_path
        self.ds=gdal.Open(self.file_path)
        self.wkt = self.ds.GetProjection()
        self.gt = self.ds.GetGeoTransform()
        self.rows = self.ds.RasterYSize
        self.cols = self.ds.RasterXSize
        self.semi_major,self.semi_minor,self.scale_factor,self.false_northing,self.false_easting,self.central_meridian=self.get_utm_params()

    def get_utm_params(self) -> dict:
        """从WKT中提取UTM投影参数"""
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.wkt) 
        return srs.GetSemiMajor(),srs.GetSemiMinor(),srs.GetProjParm(osr.SRS_PP_SCALE_FACTOR, 0.9996),srs.GetProjParm(osr.SRS_PP_FALSE_NORTHING),srs.GetProjParm(osr.SRS_PP_FALSE_EASTING),srs.GetProjParm(osr.SRS_PP_CENTRAL_MERIDIAN)
    
    def utm_lat(self) -> cp.ndarray:

        f = 1 - (self.semi_minor / self.semi_major)  # 扁率

        # 预处理坐标
        northing = cp.asarray(self.gt[3] + cp.arange(self.rows) * self.gt[5] + 0.5 * self.gt[5])
        y = (northing - self.false_northing) / self.scale_factor

        # 椭球参数
        e = cp.sqrt(2*f - f**2)
        e1 = (1 - cp.sqrt(1 - e**2)) / (1 + cp.sqrt(1 - e**2))

        # 计算基准纬度
        M = y  # 近似子午线弧长
        mu = M / (self.semi_major * (1 - e**2/4 - 3*e**4/64 - 5*e**6/256))

        phi = mu + (3*e1/2 - 27*e1**3/32)*cp.sin(2*mu) \
                + (21*e1**2/16 - 55*e1**4/32)*cp.sin(4*mu) \
                + (151*e1**3/96)*cp.sin(6*mu)

        # 最终纬度计算
        sin_phi = cp.sin(phi)
        N = self.semi_major / cp.sqrt(1 - e**2 * sin_phi**2)
        R = self.semi_major * (1 - e**2) / (1 - e**2 * sin_phi**2)**1.5

        phi_rad = phi - (N * cp.tan(phi) / R) * (0**2/2)  

        return phi_rad
    
    def wgs84_lat_lon(self) -> cp.ndarray:
        lon_out = cp.empty((self.rows, self.cols), dtype=cp.float64)
        lat_out = cp.empty((self.rows, self.cols), dtype=cp.float64)

        threads_per_block = (16, 16)
        blocks_per_grid_x = (self.cols + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (self.rows + threads_per_block[1] - 1) // threads_per_block[1]
        grid = (blocks_per_grid_x, blocks_per_grid_y)
        module = cp.RawModule(path='/home/ftx/yuanshen/cuda_kernel/wgslat.ptx')
        kernel = module.get_function('wgs84_grid_latlon')

        kernel(
            grid, threads_per_block,
            (self.rows, self.cols,
            self.gt[0], self.gt[1], self.gt[3], self.gt[5],
            lon_out, lat_out)
        )
        lon = lon_out.reshape(self.rows, self.cols)
        lat = lat_out.reshape(self.rows, self.cols)
        return cp.deg2rad(lon), cp.deg2rad(lat)

def save_to_geotiff(
        output_path: str,
        data: np.ndarray,
        gt: tuple,
        projection: str,
        nodata: float = np.nan,
    ) -> None:
        """
        将二维数组保存为GeoTIFF文件

        参数:
        output_path -- 输出文件路径
        data -- 二维numpy数组（行列顺序需与地理变换参数匹配）
        gt -- 地理变换参数 (geotransform)
        projection -- 坐标系WKT字符串
        nodata -- 无效值标记（默认为np.nan）
        """
        # 验证输入数据维度
        if len(data.shape) != 2:
            raise ValueError("输入数据必须是二维数组")

        # 获取GDAL驱动
        driver = gdal.GetDriverByName("GTiff")
        if driver is None:
            raise RuntimeError("无法获取GTiff驱动，请检查GDAL安装")

        # 创建数据集（注意行列顺序）
        rows, cols = data.shape
        dataset = driver.Create(
            output_path,
            xsize=cols,
            ysize=rows,
            bands=1,
            eType=gdal.GDT_Float32,  # 使用Float32节省空间
        )
        if dataset is None:
            raise RuntimeError(f"文件创建失败: {gdal.GetLastErrorMsg()}")

        try:
            # 设置地理参考
            dataset.SetGeoTransform(gt)
            dataset.SetProjection(projection)

            # 获取波段并写入数据
            band = dataset.GetRasterBand(1)
            band.WriteArray(data)

            # 设置无效值（处理nan）
            if np.isnan(nodata):
                band.SetNoDataValue(nodata)
                # 将numpy的nan转换为GDAL可识别的无效值
                data = np.where(np.isnan(data), nodata, data)
            else:
                band.SetNoDataValue(nodata)

            # 强制写入磁盘
            band.FlushCache()

        finally:
            # 释放资源
            dataset = None

def spinner(msg, stop_event):
    for char in itertools.cycle('哈基米叮咚鸡胖宝宝搞核算带手机一段一段'):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\r{msg} {char}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * (len(msg) + 2) + '\r')  # 清除行


def _compute_sunshine_for_day(args):
    days,file, device_id, dem_np, dem_gt, output_path,name = args

    demprocessor = DEMProcessor(file, chunk_size=1000)
    dem_da = demprocessor.to_dask_array()
    current_dem_np = dem_da.compute()
    current_dem_gt = demprocessor.get_geo_transform()
    current_dem_wkt = demprocessor.get_geo_wkt()

    with cp.cuda.Device(device_id):
        st=time.time()
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()


        rows, cols = current_dem_np.shape
        current_dem_flat = cp.asarray(current_dem_np.ravel(), dtype=cp.float64)
        geotransform = cp.asarray(dem_gt, dtype=cp.float64)
        region_bounds=cp.asarray(DEMProcessor.calculate_overlap(dem_gt,dem_np,current_dem_gt,current_dem_np),dtype=cp.int32)

        sunshine = cp.zeros_like(current_dem_flat)
        solarasg = SolarAsg(days, time_step=5)
        solar_decl = cp.float64(solarasg.calculate_solar_declination())

        latproc = Latprocessor(file)
        lon_gpu, lat_gpu = latproc.wgs84_lat_lon()
        lon_flat = lon_gpu.ravel()
        lat_flat = lat_gpu.ravel()
        module = cp.RawModule(path='/home/ftx/yuanshen/cuda_kernel/ceshi.ptx')
        kernel = module.get_function('calculateSunshineHoursCudaKernel')
        block=(16,16)
        grid = ((cols + block[0] - 1) // block[0], (rows + block[1] - 1) // block[1])
        kernel((grid[0], grid[1]), block,
            (sunshine, current_dem_flat, lat_flat, lon_flat,
                cp.int32(rows), cp.int32(cols), geotransform,
                solar_decl, region_bounds,cp.int32(5)))




        # rows, cols = current_dem_np.shape
        # current_dem_flat = cp.asarray(current_dem_np.ravel(), dtype=cp.float64)
        # geotransform = cp.asarray(current_dem_gt, dtype=cp.float64)
        # region_bounds=cp.asarray(DEMProcessor.calculate_overlap(dem_gt,dem_np,current_dem_gt,current_dem_np),dtype=cp.int32)

        # sunshine = cp.zeros_like(current_dem_flat,dtype=cp.int32)
        # solarasg = SolarAsg(15, time_step=5)
        # solar_decl = cp.float64(solarasg.calculate_solar_declination())

        # latproc = Latprocessor(file)
        # lon_gpu, lat_gpu = latproc.wgs84_lat_lon()
        # lon_flat = lon_gpu.ravel()
        # lat_flat = lat_gpu.ravel()

        # # module = cp.RawModule(path='/home/ftx/yuanshen/better_cuda/t2.ptx')
        # module = cp.RawModule(path='/home/ftx/yuanshen/better_cuda/t3.ptx')
        # kernel = module.get_function('calculateSunshineHoursCudaKernel')
        # block = (16, 16,0)
        # grid = ((cols + block[0] - 1) // block[0], (rows + block[1] - 1) // block[1])
        # shared_size = 3 * block[0] * block[1] * cp.dtype(cp.float64).itemsize

        # kernel((grid[0], grid[1]), block,
        #     (sunshine, current_dem_flat, lat_flat, lon_flat,
        #         cp.int32(rows), cp.int32(cols), geotransform,
        #         solar_decl,region_bounds ,cp.int32(5)),shared_mem=shared_size)
        cp.cuda.runtime.deviceSynchronize()
        cp.cuda.Stream.null.synchronize()

        result = cp.asnumpy(sunshine).reshape(rows, cols)
        row_start, row_end, col_start, col_end = cp.asnumpy(region_bounds)
        center_result=result[row_start:row_end,col_start:col_end]
        save_to_geotiff(output_path, center_result, dem_gt, current_dem_wkt)
        et=time.time()
        print(f"[GPU{device_id}] 完成{name}的计算，输出: {output_path}")
        print(f"[GPU{device_id}] 完成{name}的计算,计算耗时：{et-st}")

        # 删除变量引用
        del current_dem_flat
        del geotransform
        del sunshine
        del latproc
        del lat_gpu
        del lon_gpu
        del lon_flat
        del lat_flat
        del module
        del kernel
        cp.get_default_memory_pool().free_all_blocks()  # 释放显存
        cp.cuda.Stream.null.synchronize()
        import gc
        gc.collect()

        return output_path

def main(file_path,  file_path_list, base_outname,chunk_size):
    demprocessor = DEMProcessor(file_path, chunk_size=chunk_size)
    dem_da = demprocessor.to_dask_array()
    dem_np = dem_da.compute()
    dem_gt = demprocessor.get_geo_transform()
    
    days = []
    for m in range(1, 13):
        day = sum(calendar.monthrange(2025, mm)[1] for mm in range(1, m)) + 15
        days.append(day)

    gpu_ids = [4,5,6,7]
    tasks = []
    for day in days:
        for idx, file in enumerate(file_path_list):
            gpu = gpu_ids[idx % len(gpu_ids)]
            match = re.search(r'/.*/([^/]+)\.tif', file)
            result = match.group(1)
            outname = f"/home/ftx/yuanshen/data/测试结果/{base_outname}_in_dem{idx+10}_t4.tif"
            tasks.append((day,file, gpu, dem_np, dem_gt,  outname,result))

    # 4) 并行执行
    with multiprocessing.Pool(processes=len(gpu_ids)) as pool:
        results = pool.map(_compute_sunshine_for_day, tasks)

    print("所有任务完成:", results)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # 修复 Windows 多进程问题
    multiprocessing.set_start_method("spawn")  # 明确启动方式
    # file_path1='/home/ftx/yuanshen/data/边缘测试/赤道组/切割部分/1°cut_dem.tif'
    # file_path_list=[f'/home/ftx/yuanshen/data/边缘测试/赤道组/切割部分/{i}°cut_dem.tif' for i in range(3,4)]
    file_path1='/home/ftx/yuanshen/data/边缘测试/甘肃组/切割部分/9+cut_dem.tif'
    # file_path_list=[f'/home/ftx/yuanshen/data/边缘测试/赤道组/切割部分/{i}°cut_dem.tif' for i in range(10,11)]
    file_path_list=['/home/ftx/yuanshen/data/边缘测试/甘肃组/切割部分/10+cut_dem.tif']


    main(file_path=file_path1,file_path_list=file_path_list,base_outname="dem9",chunk_size=1024)




