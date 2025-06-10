import numpy as np
from osgeo import gdal, osr
import numpy as np
import cupy as cp
import dask.array as da
from dask import delayed
import os
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.font_manager import FontProperties
from natsort import natsorted

# 配置gdal过滤器
gdal.UseExceptions()
osr.UseExceptions()
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"

font = FontProperties(fname='/home/ftx/.fonts/SimHei.ttf')  # 绝对路径


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
        for data_b in range(0, self.rows, self.chunk_size):
            for x in range(0, self.cols, self.chunk_size):
                chunk_rows = min(self.chunk_size, self.rows - data_b)
                chunk_cols = min(self.chunk_size, self.cols - x)
                params.append((data_b, x, chunk_rows, chunk_cols))
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

class Correlation_Analysis:
    def __init__(self,data_a,data_b,groupname=None,outpath=None,filename=None):
        self.data_a=data_a
        self.data_b=data_b
        self.groupname=groupname
        self.outpath=outpath
        self.filename=filename
    
    #皮尔逊相关系数
    def Pearson_Correlation_Coefficient(self):

        # 将影像展开为一维数组
        data_a = self.data_a.ravel()
        values_b = self.data_b.ravel()

        # 计算皮尔逊相关系数和 p-value（此处 p 为显著性检验的 p 值）
        pearson_coef, p_value = pearsonr(data_a, values_b)
        print(f"Pearson相关系数: {pearson_coef:.6f}, 显著性p值: {p_value:.6e}")

    #互信息
    def Mutual_Information(self, bins=256):
        """
        计算两个数组 a 和 b 之间的互信息（使用二维直方图离散方法）
        """
        # 将输入展开为1D，并排除NaN
        data_a = self.data_a.ravel()
        data_b = self.data_b.ravel()
        # 计算联合直方图，范围可以根据数据范围调整
        jh, x_edges, y_edges = np.histogram2d(data_a, data_b, bins=bins)
        # 将计数转换为联合概率分布
        pxy = jh / jh.sum()
        # 计算边缘分布
        px = pxy.sum(axis=1)  # 对y求和得到x的边缘分布
        py = pxy.sum(axis=0)  # 对x求和得到y的边缘分布
        # 仅在pxy>0的区域计算互信息以避免log(0)
        nz = pxy > 0
        mi = np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz]))
        print(f"互信息:{mi:.6f}")

    #结构相似性指数与绘图
    def Structural_Similarity_Index(self):
        data_range=self.data_b.max() - self.data_b.min()
        ssim_index ,ssim_map= structural_similarity(self.data_a,self.data_b,data_range=data_range,full=True)
        print(f"结构相似性指数:{ssim_index:.6f}")
        # 绘制图像
        plt.figure(figsize=(10, 10))
        im = plt.imshow(ssim_map, 
                        cmap='coolwarm', 
                        vmin=0,          
                        vmax=1,           
                        origin='lower'    
                    )

        # 添加颜色条
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('Value', fontsize=12)

        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title(f'{self.groupname}SSIM相关性',fontproperties=font,fontsize=16)

        # 显示图像
        plt.savefig(os.path.join(self.outpath,f'{self.filename}_SSIM.png'), bbox_inches='tight', dpi=300)

    #滑动窗口局部皮尔逊相关性绘图
    
    def Sliding_Window_Local_Pearson_Correlation(self):
        from matplotlib.colors import LinearSegmentedColormap  # 新增导入
        
        imgA = cp.asarray(self.data_a)
        imgB = cp.asarray(self.data_b)
        H, W = imgA.shape 

        win = 31
        half = win // 2

        module = cp.RawModule(path='/home/ftx/yuanshen/data/代码/kernel/local_corr.ptx')
        kernel = module.get_function('local_corr')

        block = (16, 16)
        grid = ((W + block[0] - 1) // block[0],
                (H + block[1] - 1) // block[1])
        shared_mem_size = (block[0] + 2*half) * (block[1] + 2*half) * 2 * 4

        out_gpu = cp.zeros((H, W), dtype=cp.float32)
        kernel(grid, block, (imgA, imgB, out_gpu, H, W, win), shared_mem=shared_mem_size)

        out_np = out_gpu.get()

        # 创建自定义颜色映射：红色到白色
        colors = [(1, 0, 0), (1, 1, 1)]  # RGB格式，红色(1,0,0)到白色(1,1,1)
        cmap_red_white = LinearSegmentedColormap.from_list('RedWhite', colors)

        plt.figure(figsize=(10, 10))
        im = plt.imshow(out_np, 
                        cmap=cmap_red_white,  # 修改颜色映射
                        vmin=0,          
                        vmax=1,           
                        origin='lower'    
                    )

        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('Value', fontsize=12)
        plt.gca().invert_yaxis()

        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title(f'{self.groupname}局部相关性',fontproperties=font,fontsize=16)

        plt.savefig(os.path.join(self.outpath,f'{self.filename}_SWLPC.png'), bbox_inches='tight', dpi=300)
    
    #绘制差值图
    def diff_fig(self):
        diff = self.data_a - self.data_b
        abs_diff = np.abs(diff[~np.isnan(diff)])
        
        if len(abs_diff) > 0:
            # 计算实际最大值和99%分位值
            real_max = np.nanmax(abs_diff)
            clip_val = np.percentile(abs_diff, 99)
        else:
            real_max = clip_val = 0

        plt.figure(figsize=(10,10))
        img = plt.imshow(diff, cmap='RdBu', 
                        vmin=-clip_val,  # 颜色范围截止到99%分位
                        vmax=clip_val,
                        alpha=0.7)  # 添加透明度
        
        # 叠加显示极端值（超过99%分位部分）
        if real_max > clip_val:
            mask = (np.abs(diff) > clip_val)
            plt.imshow(np.where(mask, diff, np.nan), 
                    cmap='RdBu', 
                    vmin=-real_max,
                    vmax=real_max,
                    alpha=1.0)  # 极端值不透明

        # 颜色条设置
        cbar = plt.colorbar(img, label='diff')
        cbar.set_ticks([-clip_val, 0, clip_val])

        # 顶部说明文本保持原样
        cbar.ax.text(0.5, 1.05, f'差值范围: ±{real_max:.1f}',
                    transform=cbar.ax.transAxes, 
                    ha='center',
                    fontproperties=font)
        
        plt.title(f'{self.groupname}差值强调图', fontproperties=font,fontsize=16)
        
        plt.savefig(os.path.join(self.outpath, f'{self.filename}_diff.png'), 
                bbox_inches='tight', dpi=300)
        plt.close()

    def sandian(self):
        values_a=self.data_a
        values_b=self.data_b
        plt.figure(figsize=(5,5))
        plt.scatter(values_a, values_b, s=0.5, alpha=0.5)
        plt.plot([values_a.min(), values_a.max()], [values_a.min(), values_a.max()], 'r--')  # 对角线
        plt.xlabel('图像 A 像素值', fontproperties=font)
        plt.ylabel('图像 B 像素值', fontproperties=font)
        plt.title('像素值相关散点图', fontproperties=font)
        plt.savefig(os.path.join(self.outpath, f'{self.filename}_散点.png'), 
                    bbox_inches='tight', dpi=300)

    def Statistics_error(self):

        data=self.data_a-self.data_b
        abs_arr = np.abs(data)
        
        total_count = abs_arr.size
        
        count_0 = np.sum(abs_arr == 0)
        count_5 = np.sum(abs_arr == 5)
        count_10 = np.sum(abs_arr == 10)
        count_15 = np.sum(abs_arr == 15)
        count_20 = np.sum(abs_arr == 20)
        count_25 = np.sum(abs_arr == 25)
        count_gt25 = np.sum(abs_arr > 25)
        
        counts = [count_0, count_5, count_10, count_15, count_20, count_25, count_gt25]
        percentages = [count / total_count * 100 for count in counts]
        
        labels = ['0', '5', '10', '15', '20', '25', '>25']
        
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(labels, percentages, color='skyblue', edgecolor='black')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.5,  # 稍微高于条形顶部
                    f'{height:.3f}%',  # 显示百分比，保留1位小数
                    ha='center', 
                    va='bottom',
                    fontsize=12)
        
        plt.title('吻合值与各个误差值占总体值百分比',fontproperties=font, fontsize=16)
        plt.xlabel('数值', fontproperties=font, fontsize=16)
        plt.ylabel('百分比 (%)', fontproperties=font, fontsize=16)
        
        # 设置y轴范围
        plt.ylim(0, max(percentages) * 1.25)  # 留出空间显示标签
        
        # 添加网格线
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 优化布局
        plt.tight_layout()
        plt.savefig(os.path.join(self.outpath, f'{self.filename}_统计.png'), 
                    bbox_inches='tight', dpi=300)



def cal_error(mainname,filepath):
    filename=[f"{mainname}_day{i}.tif" for i in []]
    for i in filename:
        path=os.path.join(filepath,i)
        print(path)

def main():
    #组别名称
    groupname='极地组'

    #存储文件夹
    filepath=f'/home/ftx/yuanshen/data/边缘测试/{groupname}/日照时间'
    folder_path = Path(filepath)
    filenames = natsorted([entry.stem for entry in folder_path.iterdir() if entry.is_file()])
    files=natsorted([entry.name for entry in folder_path.iterdir() if entry.is_file()])
    file_paths=[os.path.join(filepath,name) for name in files ]
    standard_dem=file_paths[-1]
    compare_dem=file_paths[:-1]

    #输出文件夹
    # outpath=f"/home/ftx/yuanshen/data/绘制图像/{groupname}/SSIM"
    outpath=f"/home/ftx/yuanshen/data/绘制图像/{groupname}/差值图"
    # outpath=f"/home/ftx/yuanshen/data/绘制图像/{groupname}/像素散点图"
    # outpath=f"/home/ftx/yuanshen/data/绘制图像/{groupname}/SWLPC"

    #数据读取并计算
    dems=DEMProcessor(standard_dem)
    dems_v=dems.to_dask_array().compute()
    for path,name in zip(compare_dem,filenames[:-1]):
        demcompare=DEMProcessor(path)
        demc_v=demcompare.to_dask_array().compute()
        aly=Correlation_Analysis(dems_v,demc_v,groupname,outpath,name)
        # aly.Pearson_Correlation_Coefficient()
        # aly.Sliding_Window_Local_Pearson_Correlation()
        # aly.Structural_Similarity_Index()
        aly.diff_fig()
        # aly.sandian()


def main2():
    #组别名称
    groupname='山区组'

    #存储文件夹
    filepath1=f'/home/ftx/yuanshen/data/边缘测试/{groupname}/包含曲率计算'
    folder_path1 = Path(filepath1)
    filenames1 = natsorted([entry.stem for entry in folder_path1.iterdir() if entry.is_file()])
    files1=natsorted([entry.name for entry in folder_path1.iterdir() if entry.is_file()])
    file_paths1=[os.path.join(filepath1,name) for name in files1 ]

    filepath2=f'/home/ftx/yuanshen/data/边缘测试/{groupname}/不含曲率计算'
    folder_path2 = Path(filepath2)
    files2=natsorted([entry.name for entry in folder_path2.iterdir() if entry.is_file()])
    file_paths2=[os.path.join(filepath2,name) for name in files1 ]

    #输出文件夹
    outpath=f'/home/ftx/yuanshen/data/绘制图像/{groupname}/曲率比较'
    
    for path1,path2,name in zip(file_paths1,file_paths2,filenames1):
        DEM1=DEMProcessor(path1)
        data1,_=DEM1.extract_center_region(9)
        DEM2=DEMProcessor(path2)
        data2,_=DEM2.extract_center_region(9)
        aly=Correlation_Analysis(data1,data2,groupname,outpath,name)
        aly.diff_fig()
        aly.Statistics_error()

# main2()
day=15
DEM1=DEMProcessor(f'/home/ftx/yuanshen/data/边缘测试/旬阳组/result/包含曲率计算/N30E080_day{day}.tif')
data1=DEM1.to_dask_array().compute()
DEM2=DEMProcessor(f'/home/ftx/yuanshen/data/边缘测试/旬阳组/result/不含曲率计算/N30E080_day{day}.tif')
data2=DEM2.to_dask_array().compute()
aly=Correlation_Analysis(data1,data2,'接边1°','/home/ftx/yuanshen/data/边缘测试/旬阳组/result','接边1°')
aly.Sliding_Window_Local_Pearson_Correlation()
aly.diff_fig()
aly.Statistics_error()
