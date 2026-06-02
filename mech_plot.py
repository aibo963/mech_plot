#!/usr/bin/env python3
"""
mech_plot — 拉伸/压缩数据批量可视化工具
========================================
支持格式: .lst (WDT-20), .csv, .txt
功能: 应力-应变曲线叠加/子图、力学性能标注、批量导出

用法:
  python mech_plot.py <数据目录或文件> [选项]
  python mech_plot.py D:\实验数据\拉伸 --overlay --output results.png
  python mech_plot.py D:\实验数据\压缩 --mode compression --subplot
"""

import os
import sys
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# ============================================================
# 字体配置: 中文=宋体(SimSun), 英文=Times New Roman
# 必须在任何 plt 调用之前
# ============================================================
def setup_fonts():
    """配置 matplotlib 字体: 中文宋体 + 英文 Times New Roman"""
    # 注册字体文件
    font_files = {
        'SimSun': r'C:\Windows\Fonts\simsun.ttc',
        'Times New Roman': r'C:\Windows\Fonts\times.ttf',
    }
    registered = {}
    for name, path in font_files.items():
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            registered[name] = True

    # 设置 serif 字体族: SimSun 覆盖中文, Times New Roman 覆盖英文
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
    return registered

_registered_fonts = setup_fonts()

# 全局轴线和刻度样式 (所有图生效，包括诊断图)
plt.rcParams.update({
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.top': False,      # 上轴不加刻度线
    'ytick.right': False,    # 右轴不加刻度线
})

# ============================================================
# 配色方案
# ============================================================
COLOR_SCHEMES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173'],
    'warm': ['#e41a1c', '#ff7f00', '#ffcc00', '#ff6666', '#cc3300',
             '#ff9933', '#ffdd44', '#ff8080', '#e65c00', '#b3b300'],
    'cool': ['#377eb8', '#4daf4a', '#984ea3', '#a65628', '#f781bf',
             '#3366cc', '#00cc66', '#6633cc', '#996633', '#cc3399'],
    'grayscale': ['#000000', '#333333', '#555555', '#777777', '#999999',
                  '#aaaaaa', '#bbbbbb', '#cccccc', '#dddddd', '#eeeeee'],
    'journal': ['#c44e52', '#4c72b0', '#55a868', '#8172b2', '#ccb974',
                '#64b5cd', '#dd8452', '#da8bc3', '#8c8c8c', '#cdcd57'],
}

# ============================================================
# 数据类
# ============================================================
@dataclass
class SpecimenData:
    """单个试样的测试数据"""
    name: str                         # 试样名称 (从文件名提取)
    filepath: str                     # 原始文件路径
    load: np.ndarray                  # 载荷 (N)
    displacement: np.ndarray          # 位移 (mm)
    time: np.ndarray                  # 时间 (s)
    stress: Optional[np.ndarray] = None    # 应力 (MPa)
    strain: Optional[np.ndarray] = None    # 应变 (%, 工程应变)
    # 试样参数
    width: float = 0.0                # 宽度 (mm)
    thickness: float = 0.0            # 厚度 (mm)
    gauge_length: float = 0.0         # 标距 (mm)
    cross_section: float = 0.0        # 截面积 (mm²)
    # 力学性能
    uts: Optional[float] = None       # 抗拉强度 (MPa)
    elongation: Optional[float] = None # 断后伸长率 (%)
    # 元数据
    mode: str = 'tensile'             # tensile 或 compression
    composition: str = ''             # 成分
    treatment: str = ''               # 处理工艺
    metadata: Dict = field(default_factory=dict)


# ============================================================
# 文件解析器
# ============================================================
class DataParser:
    """通用数据文件解析器"""

    @staticmethod
    def detect_format(filepath: str) -> str:
        """检测文件格式"""
        ext = Path(filepath).suffix.lower()
        if ext == '.lst':
            return 'lst'
        elif ext == '.csv':
            return 'csv'
        elif ext in ('.txt', '.dat', '.tsv'):
            return 'auto'
        else:
            return 'auto'

    @staticmethod
    def parse_lst(filepath: str) -> np.ndarray:
        """
        解析 WDT-20 .lst 文件
        策略：找最长的连续4列数值数据块
        返回: ndarray, shape (N, 4) → [load, disp1, disp2, time]
        """
        with open(filepath, 'r', encoding='gbk', errors='replace') as f:
            lines = [l.strip() for l in f.readlines()]

        # 找最长连续4列数值块
        blocks = []
        start = None
        length = 0
        for i, line in enumerate(lines):
            parts = line.split(',')
            if len(parts) == 4:
                try:
                    [float(p) for p in parts]
                    if start is None:
                        start = i
                    length += 1
                except ValueError:
                    if length > 0:
                        blocks.append((start, length))
                    start = None
                    length = 0
            else:
                if length > 0:
                    blocks.append((start, length))
                start = None
                length = 0
        if length > 0:
            blocks.append((start, length))

        if not blocks:
            raise ValueError(f"无法在 {filepath} 中找到有效数据块")

        # 过滤全零填充块
        real_blocks = []
        for s, l in blocks:
            for j in range(s, min(s + l, len(lines))):
                try:
                    if float(lines[j].split(',')[0]) > 0:
                        real_blocks.append((s, l))
                        break
                except:
                    pass

        if not real_blocks:
            raise ValueError(f"{filepath} 中所有数据块的载荷均为零")

        best_start, best_len = max(real_blocks, key=lambda x: x[1])
        data = []
        for j in range(best_start, best_start + best_len):
            if j >= len(lines):
                break
            parts = lines[j].split(',')
            if len(parts) == 4:
                try:
                    data.append([float(p) for p in parts])
                except ValueError:
                    break

        if not data:
            raise ValueError(f"{filepath} 数据解析失败")

        return np.array(data)

    @staticmethod
    def parse_csv(filepath: str) -> np.ndarray:
        """
        解析 CSV 文件
        自动检测: 分隔符(逗号/制表符/空格) 和 列数
        """
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # 跳过标题行 (非数值行)
        data_lines = []
        header = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            # 检测分隔符
            for sep in [',', '\t', ';']:
                parts = stripped.split(sep)
                if len(parts) >= 2:
                    try:
                        [float(p.strip()) for p in parts]
                        data_lines.append([float(p.strip()) for p in parts])
                        break
                    except ValueError:
                        if header is None:
                            header = stripped  # 保存第一个非数值行作为标题
                        continue
            else:
                # 尝试空格分隔
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        [float(p) for p in parts]
                        data_lines.append([float(p) for p in parts])
                    except ValueError:
                        if header is None:
                            header = stripped

        if not data_lines:
            raise ValueError(f"无法解析 {filepath}")

        arr = np.array(data_lines)
        return arr

    @staticmethod
    def parse_auto(filepath: str) -> np.ndarray:
        """自动检测格式并解析"""
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            first_lines = [f.readline() for _ in range(10)]

        # 检查是否为 .lst 格式 (含非数值行和4列数据)
        has_header = any(not re.match(r'^[\d\s,.\-+eE]+$', l.strip()) and l.strip()
                        for l in first_lines if l.strip())
        col_counts = []
        for l in first_lines:
            l = l.strip()
            if not l:
                continue
            for sep in [',', '\t', ' ']:
                parts = l.split(sep)
                if len(parts) >= 2:
                    try:
                        [float(p) for p in parts]
                        col_counts.append(len(parts))
                        break
                    except ValueError:
                        continue

        if col_counts and max(set(col_counts), key=col_counts.count) == 4 and has_header:
            return DataParser.parse_lst(filepath)
        else:
            return DataParser.parse_csv(filepath)

    @staticmethod
    def detect_columns(filepath: str) -> Dict[str, int]:
        """
        从文件头部行检测列含义

        Returns: {列名: 列索引}，如 {'load': 4, 'stress': 5, 'disp': 2, 'strain': 3, 'time': 1}
        """
        col_map = {}
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = [f.readline() for _ in range(20)]
        except Exception:
            try:
                with open(filepath, 'r', encoding='gbk', errors='replace') as f:
                    lines = [f.readline() for _ in range(20)]
            except Exception:
                return col_map

        # 找表头行: 包含已知列名的行
        field_aliases = {
            'load': ['load(n)', 'load', '载荷', 'force(n)', 'force', 'f(n)'],
            'stress': ['stress(mpa)', 'stress', '应力', 'σ(mpa)', 'sigma'],
            'disp': ['disp(mm)', 'disp', 'displacement', '位移', 'δ(mm)'],
            'strain': ['strain(%)', 'strain', '应变', 'ε(%)', 'elongation'],
            'time': ['time(s)', 'time', '时间', 't(s)'],
        }

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 尝试不同分隔符
            for sep in ['\t', ',', ';', '  ']:
                parts = [p.strip().strip('"').strip() for p in line.split(sep) if p.strip()]
                if len(parts) < 3:
                    continue
                # 检查是否大部分是非数值 (表头行)
                non_numeric = sum(1 for p in parts if not re.match(r'^[\d.\-+eE()]+$', p))
                if non_numeric < len(parts) * 0.3:
                    continue  # 大部分是数值，不是表头

                found = {}
                for i, part in enumerate(parts):
                    part_lower = part.lower().strip()
                    for field, aliases in field_aliases.items():
                        if any(a in part_lower for a in aliases):
                            found[field] = i
                            break

                if len(found) >= 2:  # 至少识别出2列
                    col_map = found
                    return col_map

        return col_map

    @staticmethod
    def extract_header_info(filepath: str) -> Dict:
        """从文件头部提取试样信息 (支持 .lst / .txt / .csv 的表头)"""
        info = {}
        try:
            with open(filepath, 'r', encoding='gbk', errors='replace') as f:
                header_lines = [f.readline() for _ in range(50)]

            # 排除导出的CSV (含 # 注释行 或 应变(%),应力(MPa) 表头)
            text = ''.join(header_lines[:10])
            if '# ' in text and ('力学性能' in text or 'MPa' in text):
                return info
            if '应变(%)' in text and '应力(MPa)' in text:
                return info

            for i, line in enumerate(header_lines):
                line = line.strip()
                line_lower = line.lower()

                # 跳过注释行
                if line.startswith('#'):
                    continue

                # 标距: 严格匹配 "GaugeLength = 30.0 mm" 或 "标距: 20mm"
                if 'gaugelength' in line_lower or '标距' in line:
                    nums = re.findall(r'[\d.]+', line)
                    if nums:
                        val = float(nums[0])
                        if 5 < val < 500:  # 合理标距范围
                            info['gauge_length'] = val

                # SampleSize = 12.36,1.73 mm (宽×厚)
                if 'samplesize' in line_lower:
                    nums = re.findall(r'[\d.]+', line)
                    if len(nums) >= 2:
                        w, t = float(nums[0]), float(nums[1])
                        if 0.1 < w < 100 and 0.1 < t < 50:  # 合理尺寸范围
                            info['width'] = w
                            info['thickness'] = t

                # 宽度/厚度: 只匹配明确的标签行
                if any(k in line_lower for k in ['宽度', 'width', '直径']):
                    nums = re.findall(r'[\d.]+', line)
                    if nums:
                        val = float(nums[0])
                        if 0.1 < val < 100:
                            info['width'] = val

                if any(k in line_lower for k in ['厚度', 'thickness']):
                    nums = re.findall(r'[\d.]+', line)
                    if nums:
                        val = float(nums[0])
                        if 0.01 < val < 50:
                            info['thickness'] = val
        except:
            pass
        return info

    @staticmethod
    def parse_specimen_info_csv(filepath: str) -> Dict[str, Dict]:
        """
        解析试样信息 CSV 文件

        支持的列名 (自动匹配):
          试样名称/name, 宽度/width, 厚度/thickness, 标距/gauge,
          截面积/area, 成分/composition, 处理工艺/treatment

        CSV 示例:
          试样名称,宽度,厚度,标距,成分,处理工艺
          1#,3.56,1.69,20,Fe-17Mn-0.4Si,冷轧退火
          2#,3.56,1.69,20,Fe-17Mn-0.4Si,固溶

        Returns: {试样名: {width, thickness, gauge_length, cross_section, composition, treatment}}
        """
        result = {}
        try:
            with open(filepath, 'r', encoding='utf-8-sig', errors='replace') as f:
                lines = f.readlines()
        except Exception:
            try:
                with open(filepath, 'r', encoding='gbk', errors='replace') as f:
                    lines = f.readlines()
            except Exception:
                return result

        if not lines:
            return result

        # 解析表头，映射列名
        header = lines[0].strip()
        # 检测分隔符
        for sep in [',', '\t', ';']:
            cols = header.split(sep)
            if len(cols) >= 2:
                break
        else:
            cols = header.split()

        col_map = {}
        field_aliases = {
            'name': ['试样名称', '名称', 'name', '编号', '试样编号'],
            'width': ['宽度', 'width', 'w'],
            'thickness': ['厚度', 'thickness', 't'],
            'gauge_length': ['标距', 'gauge', 'gauge_length', 'l0', '长度'],
            'cross_section': ['截面积', 'area', 'cross_section', 's'],
            'composition': ['成分', 'composition', '合金'],
            'treatment': ['处理工艺', '工艺', 'treatment', '状态', '处理'],
        }
        for i, col_name in enumerate(cols):
            col_clean = col_name.strip().lower().replace('_mm', '').replace('(mm)', '').replace('（mm）', '')
            for field, aliases in field_aliases.items():
                if col_clean in [a.lower() for a in aliases]:
                    col_map[field] = i
                    break

        if 'name' not in col_map:
            print("  ⚠ 试样信息CSV: 未找到试样名称列，跳过")
            return result

        # 解析数据行
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(sep) if p.strip()]
            if len(parts) <= col_map['name']:
                continue

            name = parts[col_map['name']]
            info = {}

            def get_float(field):
                idx = col_map.get(field)
                if idx is not None and idx < len(parts):
                    try:
                        return float(parts[idx])
                    except ValueError:
                        pass
                return 0.0

            def get_str(field):
                idx = col_map.get(field)
                if idx is not None and idx < len(parts):
                    return parts[idx]
                return ''

            w = get_float('width')
            t = get_float('thickness')
            g = get_float('gauge_length')
            a = get_float('cross_section')

            if w > 0:
                info['width'] = w
            if t > 0:
                info['thickness'] = t
            if g > 0:
                info['gauge_length'] = g
            if a > 0:
                info['cross_section'] = a

            comp = get_str('composition')
            treat = get_str('treatment')
            if comp:
                info['composition'] = comp
            if treat:
                info['treatment'] = treat

            if info:
                result[name] = info

        return result


# ============================================================
# 数据处理器
# ============================================================
class DataProcessor:
    """数据处理：载荷→应力，位移→应变"""

    @staticmethod
    def convert_to_stress_strain(specimen: SpecimenData) -> SpecimenData:
        """将原始数据转换为应力-应变"""
        # 确定截面积
        if specimen.width > 0 and specimen.thickness > 0:
            specimen.cross_section = specimen.width * specimen.thickness
        elif specimen.cross_section <= 0:
            raise ValueError(f"[{specimen.name}] 未提供试样尺寸，无法计算应力")

        # 计算应力 (MPa)
        valid_load = specimen.load > 0
        specimen.stress = specimen.load[valid_load] / specimen.cross_section

        # 对应的位移和时间
        disp_valid = specimen.displacement[valid_load]
        time_valid = specimen.time[valid_load]

        # 计算工程应变 (%)
        if specimen.gauge_length > 0:
            specimen.strain = (disp_valid / specimen.gauge_length) * 100
        else:
            # 无标距时，使用位移作为名义应变
            specimen.strain = disp_valid
            print(f"  ⚠ [{specimen.name}] 未提供标距，应变轴为位移(mm)")

        specimen.displacement = disp_valid
        specimen.time = time_valid
        specimen.load = specimen.load[valid_load]

        # 基本力学性能
        uts_idx = np.argmax(specimen.stress)
        specimen.uts = float(specimen.stress[uts_idx])

        if len(specimen.strain) > 0 and specimen.strain[-1] > 0:
            specimen.elongation = float(specimen.strain[-1])

        return specimen

    @staticmethod
    def normalize_strain_start(specimen: SpecimenData) -> SpecimenData:
        """将应变起点归零（如果从负值开始）"""
        if specimen.strain is not None and len(specimen.strain) > 0:
            if specimen.strain[0] < 0:
                specimen.strain = specimen.strain - specimen.strain[0]
        return specimen

    @staticmethod
    def correct_elastic_modulus(specimen: SpecimenData, E_true: float) -> SpecimenData:
        """
        根据用户指定的真实弹性模量修正应变/位移

        原理: 如果测量的E偏低（系统柔度、引伸计滑移等），
        说明应变被高估了。修正: ε_true = ε_measured × (E_measured / E_true)

        Args:
            specimen: 已转换为应力-应变的试样数据
            E_true: 用户指定的真实弹性模量 (MPa)
        """
        if specimen.stress is None or specimen.strain is None:
            return specimen
        if len(specimen.stress) < 20:
            return specimen

        stress = specimen.stress
        strain = specimen.strain

        # 滑动窗口找弹性段: 斜率×R² 最高
        uts = np.max(stress)
        W = max(15, len(stress) // 100)  # 窗口大小
        best_score = 0
        best_slope = 0

        for i in range(0, len(stress) - W):
            s = stress[i:i+W]
            e = strain[i:i+W]

            # 跳过高应力区 (>60% UTS)
            if np.max(s) > uts * 0.6:
                continue

            # 跳过非单调区
            if np.sum(np.diff(s) < 0) > W * 0.15:
                continue

            if np.std(e) < 1e-6:
                continue

            coeffs = np.polyfit(e, s, 1)
            slope = coeffs[0]
            if slope <= 0:
                continue

            y_pred = np.polyval(coeffs, e)
            ss_res = np.sum((s - y_pred)**2)
            ss_tot = np.sum((s - np.mean(s))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            score = slope * (r2 ** 2)
            if score > best_score and r2 > 0.95:
                best_score = score
                best_slope = slope

        if best_slope <= 0:
            print(f"  ⚠ [{specimen.name}] 未找到弹性段，跳过E修正")
            return specimen

        # E = dσ/dε, 应变是%单位, E = slope × 100
        E_measured = best_slope * 100  # MPa

        if E_measured < 1000 or E_measured > 500000:
            print(f"  ⚠ [{specimen.name}] 测量E={E_measured:.0f}MPa不合理，跳过修正")
            return specimen

        # 修正因子: 应变被高估了 ratio 倍，需要缩小
        ratio = E_measured / E_true

        if abs(ratio - 1.0) < 0.02:
            print(f"  ✅ [{specimen.name}] E={E_measured/1000:.1f}GPa ≈ {E_true/1000:.1f}GPa，无需修正")
            return specimen

        # 修正应变和位移: ε_true = ε_measured × (E_measured / E_true)
        specimen.strain = strain * ratio
        if specimen.gauge_length > 0:
            specimen.displacement = specimen.displacement * ratio

        # 重新计算力学性能
        uts_idx = np.argmax(specimen.stress)
        specimen.uts = float(specimen.stress[uts_idx])
        if len(specimen.strain) > 0 and specimen.strain[-1] > 0:
            specimen.elongation = float(specimen.strain[-1])

        print(f"  🔧 [{specimen.name}] E修正: {E_measured/1000:.1f}→{E_true/1000:.1f}GPa, 比值={ratio:.4f}")

        return specimen


# ============================================================
# 弹性段自动检测与修正
# ============================================================
@dataclass
class ElasticDiagnostic:
    """弹性段诊断结果"""
    specimen_name: str
    issues: List[str]           # 发现的问题描述
    true_elastic_start: int     # 真正弹性段起始索引
    true_elastic_end: int       # 真正弹性段结束索引
    elastic_modulus: float      # 真正弹性模量 (MPa, 基于应力-应变)
    original_E: float           # 原始前10点斜率
    corrected: bool             # 是否进行了修正
    fit_k: float = 0.0          # 拟合直线斜率 (load/disp)
    fit_b: float = 0.0          # 拟合直线截距
    n_replaced: int = 0         # 被替换的数据点数
    cut_idx: int = 0            # 截断点索引


class ElasticRegionCorrector:
    def get_mechanical_properties(self, specimen: SpecimenData, diagnostic: ElasticDiagnostic) -> Dict:
        """
        提取完整力学性能指标
        
        返回: {
            'E': 弹性模量 (MPa),
            'UTS': 抗拉强度 (MPa),
            'yield': 屈服强度 (MPa),
            'yield_type': 'R0.2' / '上屈服' / '下屈服',
            'elongation': 延伸率 (%),
            'upper_yield': 上屈服 (MPa, 仅上下屈服模式),
            'lower_yield': 下屈服 (MPa, 仅上下屈服模式),
        }
        """
        props = {}
        
        load = specimen.load
        disp = specimen.displacement
        stress = specimen.stress
        strain = specimen.strain
        area = specimen.cross_section
        gauge = specimen.gauge_length
        
        es_s = diagnostic.true_elastic_start
        es_e = diagnostic.true_elastic_end
        k = diagnostic.fit_k
        cut_idx = diagnostic.cut_idx
        
        # 截断后的数据
        if cut_idx > 0 and cut_idx < len(load):
            load_cut = load[cut_idx:]
            disp_cut = disp[cut_idx:]
            stress_cut = stress[cut_idx:]
            strain_cut = (disp_cut - disp_cut[0]) / gauge * 100 if gauge > 0 else disp_cut
            es_s_new = es_s - cut_idx
            es_e_new = es_e - cut_idx
        else:
            load_cut = load
            disp_cut = disp
            stress_cut = stress
            strain_cut = strain
            es_s_new = es_s
            es_e_new = es_e
        
        # 弹性模量
        if k > 0 and gauge > 0 and area > 0:
            props['E'] = k * gauge / area  # MPa (应变为无量纲)
        elif diagnostic.elastic_modulus > 0:
            props['E'] = diagnostic.elastic_modulus
        else:
            props['E'] = 0
        
        # UTS
        uts_idx = np.argmax(stress_cut)
        props['UTS'] = float(stress_cut[uts_idx])
        
        # 延伸率
        props['elongation'] = float(strain_cut[-1]) if len(strain_cut) > 0 else 0
        
        # 屈服强度
        E_elastic = props['E']
        
        # 尝试找上下屈服
        yield_found = False
        if es_e_new >= 0 and es_e_new < len(stress_cut):
            post_elastic = stress_cut[es_e_new:]
            post_strain = strain_cut[es_e_new:]
            if len(post_elastic) > 10:
                search_end = min(len(post_elastic), max(int(len(post_elastic) * 0.2), 20))
                upper_idx = np.argmax(post_elastic[:search_end])
                upper_val = post_elastic[upper_idx]
                
                if upper_idx + 1 < len(post_elastic):
                    # 只在上屈服点后的前30%范围内找下屈服（避免找到曲线末端的下降段）
                    lower_search_end = min(len(post_elastic), upper_idx + max(20, int(len(post_elastic) * 0.3)))
                    lower_search = post_elastic[upper_idx:lower_search_end]
                    lower_idx = np.argmin(lower_search) + upper_idx
                    lower_val = post_elastic[lower_idx]
                    
                    # 上屈服比下屈服高10%以上且差值>20MPa
                    if upper_val > lower_val * 1.1 and (upper_val - lower_val) > 20:
                        props['yield'] = float(lower_val)
                        props['yield_type'] = '下屈服'
                        props['upper_yield'] = float(upper_val)
                        props['lower_yield'] = float(lower_val)
                        yield_found = True
        
        # 没有上下屈服，用R0.2
        if not yield_found and E_elastic > 1000:
            offset_line = E_elastic * (strain_cut / 100 - 0.002)
            search_start = max(0, es_s_new) if es_s_new >= 0 else 0
            for i in range(search_start, len(stress_cut) - 1):
                diff_curr = stress_cut[i] - offset_line[i]
                diff_next = stress_cut[i + 1] - offset_line[i + 1]
                if diff_curr >= 0 and diff_next < 0:
                    ratio = diff_curr / (diff_curr - diff_next)
                    props['yield'] = float(stress_cut[i] + ratio * (stress_cut[i + 1] - stress_cut[i]))
                    props['yield_type'] = 'R0.2'
                    yield_found = True
                    break
        
        if not yield_found:
            props['yield'] = None
            props['yield_type'] = None
        
        return props

    """
    弹性段自动检测与修正

    算法:
    1. 滑动窗口计算局部斜率 d(load)/d(disp)
    2. 找到最大一致斜率区域 = 真正弹性段
    3. 检测异常: 载荷下降、低斜率、高初始载荷
    4. 用真正弹性段拟合直线回推，替换低质量数据
    """

    def __init__(self,
                 slope_window: int = 5,
                 slope_threshold_ratio: float = 0.5,
                 load_drop_threshold: float = 0.85,
                 min_elastic_points: int = 8):
        """
        Args:
            slope_window: 计算局部斜率的滑动窗口大小(数据点数)
            slope_threshold_ratio: 斜率阈值 = max_slope * 此比例，低于此为异常
            load_drop_threshold: 载荷下降判定阈值 (当前/前一点 < 此值)
            min_elastic_points: 真正弹性段最少需要的数据点数
        """
        self.slope_window = max(3, slope_window)
        self.slope_threshold_ratio = slope_threshold_ratio
        self.load_drop_threshold = load_drop_threshold
        self.min_elastic_points = min_elastic_points

    def _compute_local_slopes(self, load: np.ndarray, disp: np.ndarray) -> np.ndarray:
        """计算每个点的局部斜率 d(load)/d(disp)，用滑动窗口线性拟合"""
        n = len(load)
        slopes = np.zeros(n)
        w = self.slope_window
        for i in range(n):
            lo = max(0, i - w)
            hi = min(n, i + w + 1)
            if hi - lo < 3:
                slopes[i] = 0
                continue
            x = disp[lo:hi]
            y = load[lo:hi]
            dx = x[-1] - x[0]
            if abs(dx) < 1e-10:
                slopes[i] = 0
            else:
                # 用最小二乘拟合斜率，比端点差分更稳健
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                ss_xx = np.sum((x - x_mean) ** 2)
                if ss_xx < 1e-15:
                    slopes[i] = 0
                else:
                    slopes[i] = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
        return slopes

    def _find_load_drops(self, load: np.ndarray, search_range: int) -> List[int]:
        """在前 search_range 个点中找载荷下降位置"""
        drops = []
        for i in range(1, min(search_range, len(load))):
            if load[i] < load[i-1] * self.load_drop_threshold:
                drops.append(i)
        return drops

    def _find_uts_index(self, stress: np.ndarray) -> int:
        """找UTS对应的索引"""
        return int(np.argmax(stress))

    def _find_true_elastic_region(self, slopes: np.ndarray, load: np.ndarray,
                                   disp: np.ndarray, stress: np.ndarray) -> Tuple[int, int, float]:
        """
        找到真正弹性段的起止索引和斜率

        核心思路 (v8):
        - 全范围扫描，找R²最高且斜率最大的连续段 = 弹性段核心
        - 弹性段特征: 高R²(>0.99) + 高斜率 + 载荷单调递增
        - 弹性斜率 > 塑性斜率，弹性区在塑性区之前
        - 从核心向两侧扩展(左:滑动→弹性过渡, 右:弹性→塑性过渡)
        """
        n = len(load)
        W = max(self.min_elastic_points, 15)  # 窗口大小

        # 全范围扫描: 计算每个位置的R²和斜率
        best_score = -1
        best_idx = 0
        best_slope = 0.0
        best_r2 = 0.0

        for i in range(0, n - W):
            x = disp[i:i+W]
            y = load[i:i+W]

            # 载荷必须基本单调递增
            diffs = np.diff(y)
            if np.sum(diffs < 0) > W * 0.15:  # 允许少量波动
                continue

            # 线性拟合
            if np.std(x) < 1e-10:
                continue
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            if slope <= 0:
                continue

            # R²
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # 评分: 斜率 × R²² (高斜率且高线性度)
            score = slope * (r2 ** 2)

            if score > best_score and r2 > 0.98:
                best_score = score
                best_idx = i
                best_slope = slope
                best_r2 = r2

        if best_slope <= 0:
            return 0, min(self.min_elastic_points, n), 0.0

        # 弹性段核心
        core_start = best_idx
        core_end = best_idx + W

        # 向左扩展: 找弹性段真正的起点
        # 条件: R² > 0.95 且 斜率 > 核心的50%
        elastic_start = core_start
        left_slope_threshold = best_slope * 0.5
        for i in range(core_start - 1, max(0, core_start - 500) - 1, -1):
            if i < W:
                break
            x = disp[i:i+W]
            y = load[i:i+W]
            if np.std(x) < 1e-10:
                break
            coeffs = np.polyfit(x, y, 1)
            s = coeffs[0]
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r2 < 0.95 or s < left_slope_threshold:
                break
            elastic_start = i

        # 向右扩展: 找屈服点(斜率下降)
        elastic_end = core_end
        right_slope_threshold = best_slope * 0.5
        for i in range(core_end, min(n - W, core_end + 500)):
            x = disp[i:i+W]
            y = load[i:i+W]
            if np.std(x) < 1e-10:
                break
            coeffs = np.polyfit(x, y, 1)
            s = coeffs[0]
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r2 < 0.95 or s < right_slope_threshold:
                elastic_end = i
                break
            elastic_end = i + W

        # 确保范围合理
        elastic_end = max(elastic_end, elastic_start + self.min_elastic_points)
        elastic_end = min(elastic_end, n)

        # 精炼: 在已找到的弹性段内，找R²最高的子段
        # 弹性段前半可能还在滑动→弹性的过渡区，后半更纯
        if elastic_end - elastic_start > self.min_elastic_points * 2:
            sub_W = max(self.min_elastic_points, (elastic_end - elastic_start) // 3)
            best_sub_r2 = 0
            best_sub_start = elastic_start
            for i in range(elastic_start, elastic_end - sub_W):
                x = disp[i:i+sub_W]
                y = load[i:i+sub_W]
                if np.std(x) < 1e-10:
                    continue
                coeffs = np.polyfit(x, y, 1)
                if coeffs[0] <= 0:
                    continue
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                if r2 > best_sub_r2:
                    best_sub_r2 = r2
                    best_sub_start = i
                    best_slope = coeffs[0]

            # 用精炼后的子段作为最终弹性段
            elastic_start = best_sub_start
            elastic_end = best_sub_start + sub_W

        return elastic_start, elastic_end, best_slope

    def diagnose(self, specimen: SpecimenData) -> ElasticDiagnostic:
        """诊断弹性段质量，不修改数据"""
        load = specimen.load.copy()
        disp = specimen.displacement.copy()
        stress = specimen.stress.copy()
        strain = specimen.strain.copy()
        n = len(load)

        # 计算局部斜率 (load/disp 单位)
        slopes = self._compute_local_slopes(load, disp)

        # 应力-应变斜率
        if specimen.gauge_length > 0 and specimen.cross_section > 0:
            stress_strain_slopes = slopes * specimen.gauge_length / (specimen.cross_section * 100)  # → MPa per %strain
        else:
            stress_strain_slopes = slopes

        # 原始前10点斜率
        if n > 10:
            coeffs_orig = np.polyfit(strain[:10], stress[:10], 1)
            original_E = coeffs_orig[0] * 100  # MPa
        else:
            original_E = 0

        # 找真正弹性段
        es_start, es_end, max_slope = self._find_true_elastic_region(slopes, load, disp, stress)

        # 真正弹性模量 (应力-应变)
        if es_end > es_start + 2:
            coeffs_true = np.polyfit(strain[es_start:es_end], stress[es_start:es_end], 1)
            true_E = coeffs_true[0] * 100  # MPa
        else:
            true_E = 0

        # 检测问题
        issues = []
        # 1. 载荷下降
        drops = self._find_load_drops(load, n // 3)
        if drops:
            for d in drops:
                drop_pct = (1 - load[d] / load[d-1]) * 100
                issues.append(f'载荷下降: pt{d} ({load[d-1]:.0f}→{load[d]:.0f}N, ↓{drop_pct:.0f}%)')

        # 2. 低斜率
        if original_E < 2000:
            issues.append(f'初始E过低: {original_E:.0f} MPa')

        # 3. 高初始载荷
        if load[0] > 50:
            issues.append(f'初始载荷过高: {load[0]:.0f}N')

        # 4. 真正弹性段与起点有偏移
        if es_start > 3:
            issues.append(f'低质量段持续到pt{es_start}')

        return ElasticDiagnostic(
            specimen_name=specimen.name,
            issues=issues,
            true_elastic_start=es_start,
            true_elastic_end=es_end,
            elastic_modulus=true_E,
            original_E=original_E,
            corrected=False,
            n_replaced=0,
        )

    def correct(self, specimen: SpecimenData, diagnostic: ElasticDiagnostic = None) -> Tuple[SpecimenData, ElasticDiagnostic]:
        """
        修正弹性段数据 — 完整流程

        步骤:
        1. 拟合弹性段斜率 k
        2. 弹性趋势线外推到载荷=0 → 截断点
        3. 从截断点截断，设为位移/应变原点
        4. 用趋势线+模板残差替换截断点到弹性段之间的滑动段
        """
        if diagnostic is None:
            diagnostic = self.diagnose(specimen)

        if not diagnostic.issues:
            diagnostic.corrected = False
            return specimen, diagnostic

        load = specimen.load.copy()
        disp = specimen.displacement.copy()
        es_start = diagnostic.true_elastic_start
        es_end = diagnostic.true_elastic_end
        area = specimen.cross_section
        gauge = specimen.gauge_length

        if es_end <= es_start + 2:
            diagnostic.corrected = False
            return specimen, diagnostic

        # === 第1步: 弹性段斜率 ===
        fit_x = disp[es_start:es_end]
        fit_y = load[es_start:es_end]
        k = np.polyfit(fit_x, fit_y, 1)[0]

        # === 第2步: 截断点 = 弹性趋势线 load=0 处 ===
        b = load[es_start] - k * disp[es_start]
        disp_at_zero = -b / k
        cut_idx = np.searchsorted(disp, disp_at_zero)
        cut_idx = max(0, min(cut_idx, len(disp) - 1))

        # === 第3步: 截断，设为原点 ===
        load_cut = load[cut_idx:]
        disp_cut = disp[cut_idx:]
        es_start_new = es_start - cut_idx
        es_end_new = es_end - cut_idx

        # === 第4步: 波动模板 ===
        W = 20
        best_r2 = 0
        best_i = es_start
        for i in range(es_start, es_end - W):
            x = disp[i:i+W]
            y = load[i:i+W]
            c = np.polyfit(x, y, 1)
            yp = np.polyval(c, x)
            ss_r = np.sum((y - yp)**2)
            ss_t = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
            if r2 > best_r2:
                best_r2 = r2
                best_i = i

        tmpl_disp = disp[best_i:best_i+W]
        tmpl_load = load[best_i:best_i+W]
        tc = np.polyfit(tmpl_disp, tmpl_load, 1)
        tmpl_residuals = tmpl_load - np.polyval(tc, tmpl_disp)

        # === 第5步: 替换滑动段 ===
        load_new = load_cut.copy()
        offset = W - (es_start_new % W)
        for i in range(es_start_new):
            j = (i + offset) % W
            trend = k * disp_cut[i] + b
            load_new[i] = trend + tmpl_residuals[j]

        # 更新 specimen
        specimen.load = load_new
        specimen.displacement = disp_cut
        if area > 0:
            specimen.stress = load_new / area
        if gauge > 0:
            specimen.strain = (disp_cut - disp_cut[0]) / gauge * 100

        # 更新力学性能
        uts_idx = np.argmax(specimen.stress)
        specimen.uts = float(specimen.stress[uts_idx])
        if len(specimen.strain) > 0 and specimen.strain[-1] > 0:
            specimen.elongation = float(specimen.strain[-1])

        diagnostic.corrected = True
        diagnostic.fit_k = k
        diagnostic.fit_b = b
        diagnostic.n_replaced = es_start_new
        diagnostic.true_elastic_start = es_start_new
        diagnostic.true_elastic_end = es_end_new
        diagnostic.cut_idx = 0  # 数据已被截断，不再需要二次截断
        diagnostic._original_cut_idx = cut_idx  # 保存原始截断点供诊断图使用
        diagnostic._original_es_s = es_start  # 原始弹性段起始
        diagnostic._original_es_e = es_end    # 原始弹性段结束
        if gauge > 0 and area > 0:
            diagnostic.elastic_modulus = k * gauge / (area * 100)

        return specimen, diagnostic

    def plot_diagnostic(self, specimen_orig: SpecimenData, diagnostic: ElasticDiagnostic) -> plt.Figure:
        """生成四阶段诊断图: 原始 → 弹性段检测 → 截断 → 替换"""
        sp = specimen_orig
        load = sp.load
        disp = sp.displacement
        stress = sp.stress
        strain = sp.strain
        area = sp.cross_section
        gauge = sp.gauge_length

        es_s = diagnostic.true_elastic_start
        es_e = diagnostic.true_elastic_end
        k = diagnostic.fit_k
        b = diagnostic.fit_b
        cut_idx = getattr(diagnostic, '_original_cut_idx', diagnostic.cut_idx)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # --- 阶段1: 原始曲线 ---
        ax = axes[0, 0]
        ax.plot(strain, stress, 'b-', linewidth=0.8)
        ax.set_xlabel('应变 (%)')
        ax.set_ylabel('应力 (MPa)')
        ax.autoscale(True)
        ax.margins(x=0.05, y=0.05)
        ax.set_title('① 原始曲线')

        # --- 阶段2: 弹性段检测 ---
        ax = axes[0, 1]
        ax.plot(strain, stress, 'b-', linewidth=0.8, alpha=0.5)
        if es_e > es_s:
            ax.plot(strain[es_s:es_e], stress[es_s:es_e], 'r-', linewidth=2.5,
                    label=f'弹性段 E={diagnostic.elastic_modulus:.0f}MPa')
        ax.set_xlabel('应变 (%)')
        ax.set_ylabel('应力 (MPa)')
        ax.autoscale(True)
        ax.margins(x=0.05, y=0.05)
        ax.set_title('② 弹性段检测')
        ax.legend(fontsize=9)

        # --- 阶段3: 截断 (从载荷=0处) ---
        ax = axes[1, 0]
        if cut_idx > 0:
            # 截断后的数据
            strain_cut = (disp[cut_idx:] - disp[cut_idx]) / gauge * 100
            stress_cut = stress[cut_idx:]
            es_s_new = es_s - cut_idx
            es_e_new = es_e - cut_idx

            ax.plot(strain_cut, stress_cut, 'b-', linewidth=0.8, alpha=0.5)
            if es_s_new >= 0 and es_e_new <= len(strain_cut):
                ax.plot(strain_cut[es_s_new:es_e_new], stress_cut[es_s_new:es_e_new],
                        'r-', linewidth=2.5, label='弹性段')

            # 弹性趋势线延伸到0
            strain_ext = np.linspace(0, strain_cut[es_s_new], 50)
            disp_ext = strain_ext / 100 * gauge + disp[cut_idx]
            stress_ext = (k * disp_ext + b) / area
            ax.plot(strain_ext, stress_ext, 'r--', linewidth=1, alpha=0.5, label='趋势线')

            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlim(left=-0.5)

            # === 力学性能标注 ===
            # UTS
            uts_idx = np.argmax(stress_cut)
            uts_val = stress_cut[uts_idx]
            uts_strain = strain_cut[uts_idx]
            ax.plot(uts_strain, uts_val, 'rv', markersize=8)
            ax.annotate(f'UTS={uts_val:.0f}MPa', xy=(uts_strain, uts_val),
                        xytext=(10, 10), textcoords='offset points', fontsize=8,
                        color='red', fontweight='bold')

            # 屈服强度: 自动判断R0.2还是上下屈服
            E_elastic = k * gauge / area  # 弹性模量 (MPa, 应变为无量纲)
            yield_val = None
            yield_label = ''

            # 在弹性段之后找上下屈服点
            # 上屈服: 弹性段后第一个应力峰值
            # 下屈服: 上屈服后的应力谷值
            es_s_new = es_s - cut_idx
            es_e_new = es_e - cut_idx
            if es_s_new >= 0 and es_e_new < len(stress_cut):
                post_elastic = stress_cut[es_e_new:]
                post_strain = strain_cut[es_e_new:]
                if len(post_elastic) > 10:
                    # 找上屈服: 前20%内的最大值
                    search_end = min(len(post_elastic), max(int(len(post_elastic) * 0.2), 20))
                    upper_idx = np.argmax(post_elastic[:search_end])
                    upper_val = post_elastic[upper_idx]

                    # 找下屈服: 上屈服后的最小值
                    if upper_idx + 1 < len(post_elastic):
                        lower_search = post_elastic[upper_idx:]
                        lower_idx = np.argmin(lower_search) + upper_idx
                        lower_val = post_elastic[lower_idx]

                        # 判断: 如果上屈服比下屈服高10%以上，认为有上下屈服
                        if upper_val > lower_val * 1.1 and (upper_val - lower_val) > 20:
                            yield_val = lower_val
                            yield_strain = post_strain[lower_idx]
                            yield_label = f'下屈服={lower_val:.0f}MPa'
                            # 标注上屈服
                            ax.plot(post_strain[upper_idx], upper_val, 'g^', markersize=8)
                            ax.annotate(f'上屈服={upper_val:.0f}MPa',
                                        xy=(post_strain[upper_idx], upper_val),
                                        xytext=(10, -15), textcoords='offset points',
                                        fontsize=8, color='green')

            # 如果没有上下屈服，用R0.2
            if yield_val is None and E_elastic > 1000:  # E太低时R0.2不可靠
                # 0.2%偏移线: stress = E * (strain - 0.002)
                # 与应力-应变曲线的交点
                offset_line = E_elastic * (strain_cut / 100 - 0.002)
                # 只在弹性段之后找交点
                if es_s_new >= 0:
                    search_start = es_s_new
                else:
                    search_start = 0
                for i in range(search_start, len(stress_cut) - 1):
                    diff_curr = stress_cut[i] - offset_line[i]
                    diff_next = stress_cut[i + 1] - offset_line[i + 1]
                    if diff_curr >= 0 and diff_next < 0:
                        # 线性插值
                        ratio = diff_curr / (diff_curr - diff_next)
                        yield_val = stress_cut[i] + ratio * (stress_cut[i + 1] - stress_cut[i])
                        yield_strain = strain_cut[i] + ratio * (strain_cut[i + 1] - strain_cut[i])
                        yield_label = f'R0.2={yield_val:.0f}MPa'
                        # 画0.2%偏移线
                        offset_x = np.linspace(0.2, strain_cut[min(i + 20, len(strain_cut) - 1)], 50)
                        offset_y = E_elastic * (offset_x / 100 - 0.002) * 100 / 100
                        # 简化: 直接画从0.2%开始的直线
                        ax.plot([0.2, strain_cut[min(i + 20, len(strain_cut) - 1)]],
                                [0, E_elastic * (strain_cut[min(i + 20, len(strain_cut) - 1)] / 100 - 0.002)],
                                'k--', linewidth=0.8, alpha=0.5)
                        break

            # 标注屈服点
            if yield_val is not None:
                ax.plot(yield_strain, yield_val, 'bs', markersize=8)
                ax.annotate(yield_label, xy=(yield_strain, yield_val),
                            xytext=(10, -15), textcoords='offset points',
                            fontsize=8, color='blue', fontweight='bold')

            # 延伸率
            elong = strain_cut[-1]
            ax.annotate(f'延伸率={elong:.1f}%', xy=(strain_cut[-1], stress_cut[-1]),
                        xytext=(-60, 10), textcoords='offset points',
                        fontsize=8, color='purple', fontweight='bold')

        else:
            ax.plot(strain, stress, 'b-', linewidth=0.8)
            ax.autoscale(True)
        ax.set_xlabel('应变 (%)')
        ax.set_ylabel('应力 (MPa)')
        ax.margins(x=0.05, y=0.05)
        ax.set_title(f'③ 截断 (pt{cut_idx})')
        ax.legend(fontsize=9)

        # --- 阶段4: 替换滑动段 ---
        ax = axes[1, 1]
        if cut_idx > 0 and diagnostic.corrected:
            strain_cut = (disp[cut_idx:] - disp[cut_idx]) / gauge * 100
            es_s_new = es_s - cut_idx

            # 重新生成替换数据
            load_cut = load[cut_idx:]
            disp_cut = disp[cut_idx:]

            # 波动模板
            W = 20
            best_r2 = 0
            best_i = es_s
            for i in range(es_s, es_e - W):
                x = disp[i:i+W]
                y = load[i:i+W]
                c = np.polyfit(x, y, 1)
                yp = np.polyval(c, x)
                ss_r = np.sum((y - yp)**2)
                ss_t = np.sum((y - np.mean(y))**2)
                r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
                if r2 > best_r2:
                    best_r2 = r2
                    best_i = i

            tmpl_d = disp[best_i:best_i+W]
            tmpl_l = load[best_i:best_i+W]
            tc = np.polyfit(tmpl_d, tmpl_l, 1)
            tmpl_res = tmpl_l - np.polyval(tc, tmpl_d)

            load_new = load_cut.copy()
            offset = W - (es_s_new % W)
            for i in range(es_s_new):
                j = (i + offset) % W
                trend = k * disp_cut[i] + b
                load_new[i] = trend + tmpl_res[j]

            stress_new = load_new / area

            ax.plot(strain_cut, stress_new, 'g-', linewidth=0.8)
            ax.autoscale(True)
        else:
            ax.plot(strain, stress, 'g-', linewidth=0.8)
            ax.autoscale(True)
        ax.set_xlabel('应变 (%)')
        ax.set_ylabel('应力 (MPa)')
        ax.margins(x=0.05, y=0.05)
        ax.set_title('④ 替换滑动段')

        fig.suptitle(f'弹性段诊断: {sp.name}', fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig

    def plot_diagnostic_with_props(self, specimen_orig: SpecimenData, 
                                    specimen_corrected: SpecimenData,
                                    diagnostic: ElasticDiagnostic,
                                    props: Dict) -> plt.Figure:
        """
        生成带完整力学性能标注的四阶段诊断图
        
        Args:
            specimen_orig: 原始试样数据
            specimen_corrected: 修正后的试样数据
            diagnostic: 诊断结果
            props: 力学性能字典 (从 get_mechanical_properties 获取)
        """
        sp = specimen_orig
        load = sp.load
        disp = sp.displacement
        stress = sp.stress
        strain = sp.strain
        area = sp.cross_section
        gauge = sp.gauge_length

        es_s = diagnostic.true_elastic_start
        es_e = diagnostic.true_elastic_end
        k = diagnostic.fit_k
        b = diagnostic.fit_b
        cut_idx = getattr(diagnostic, '_original_cut_idx', diagnostic.cut_idx)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # --- 阶段1: 原始曲线 ---
        ax = axes[0, 0]
        ax.plot(strain, stress, 'b-', linewidth=0.8)
        ax.set_xlabel('应变 (%)')
        ax.set_ylabel('应力 (MPa)')
        ax.autoscale(True)
        ax.margins(x=0.05, y=0.05)
        ax.set_title('① 原始曲线', fontsize=12, fontweight='bold')
        
        # 标注问题区域
        if diagnostic.issues:
            issue_text = '\n'.join(diagnostic.issues[:3])  # 最多显示3个问题
            ax.text(0.02, 0.98, f'问题:\n{issue_text}', transform=ax.transAxes,
                   fontsize=8, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                   color='red')

        # --- 阶段2: 弹性段检测 ---
        ax = axes[0, 1]
        ax.plot(strain, stress, 'b-', linewidth=0.8, alpha=0.5)
        if es_e > es_s:
            ax.plot(strain[es_s:es_e], stress[es_s:es_e], 'r-', linewidth=2.5,
                    label=f'弹性段 (E={props.get("E", 0):.0f} MPa)')
        ax.set_xlabel('应变 (%)')
        ax.set_ylabel('应力 (MPa)')
        ax.autoscale(True)
        ax.margins(x=0.05, y=0.05)
        ax.set_title('② 弹性段检测', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')

        # --- 阶段3: 截断 + 力学性能标注 ---
        ax = axes[1, 0]
        if cut_idx > 0:
            strain_cut = (disp[cut_idx:] - disp[cut_idx]) / gauge * 100
            stress_cut = stress[cut_idx:]
            es_s_new = es_s - cut_idx
            es_e_new = es_e - cut_idx

            ax.plot(strain_cut, stress_cut, 'b-', linewidth=0.8, alpha=0.5)
            if es_s_new >= 0 and es_e_new <= len(strain_cut):
                ax.plot(strain_cut[es_s_new:es_e_new], stress_cut[es_s_new:es_e_new],
                        'r-', linewidth=2.5, label='弹性段')

            # 弹性趋势线延伸到0
            strain_ext = np.linspace(0, strain_cut[es_s_new], 50)
            disp_ext = strain_ext / 100 * gauge + disp[cut_idx]
            stress_ext = (k * disp_ext + b) / area
            ax.plot(strain_ext, stress_ext, 'r--', linewidth=1, alpha=0.5, label='趋势线')

            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlim(left=-0.5)

            # === 力学性能标注 ===
            # UTS
            uts_val = props.get('UTS', 0)
            uts_idx = np.argmax(stress_cut)
            uts_strain = strain_cut[uts_idx]
            ax.plot(uts_strain, uts_val, 'rv', markersize=10, zorder=5)
            ax.annotate(f'UTS = {uts_val:.0f} MPa', 
                       xy=(uts_strain, uts_val),
                       xytext=(15, 15), textcoords='offset points',
                       fontsize=10, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

            # 屈服强度
            yield_val = props.get('yield')
            yield_type = props.get('yield_type', '')
            if yield_val is not None:
                # 找到屈服点对应的应变
                if yield_type == 'R0.2':
                    # R0.2: 画偏移线
                    E_elastic = props.get('E', 0)
                    offset_x = np.linspace(0.2, strain_cut[-1] * 0.8, 100)
                    offset_y = E_elastic * (offset_x / 100 - 0.002)
                    valid = offset_y >= 0
                    ax.plot(offset_x[valid], offset_y[valid], 'k--', linewidth=1, alpha=0.6, label='0.2%偏移线')
                    
                    # 找交点位置
                    for i in range(len(stress_cut) - 1):
                        line_val = E_elastic * (strain_cut[i] / 100 - 0.002)
                        line_val_next = E_elastic * (strain_cut[i + 1] / 100 - 0.002)
                        if stress_cut[i] >= line_val and stress_cut[i + 1] < line_val_next:
                            yield_strain = strain_cut[i]
                            break
                    else:
                        yield_strain = 0.2
                    
                    ax.plot(yield_strain, yield_val, 'bs', markersize=10, zorder=5)
                    ax.annotate(f'R0.2 = {yield_val:.0f} MPa',
                               xy=(yield_strain, yield_val),
                               xytext=(15, -20), textcoords='offset points',
                               fontsize=10, color='blue', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
                else:
                    # 上下屈服
                    upper_yield = props.get('upper_yield')
                    lower_yield = props.get('lower_yield')
                    if upper_yield and lower_yield:
                        # 找上屈服点应变
                        if es_e_new >= 0 and es_e_new < len(stress_cut):
                            post_strain = strain_cut[es_e_new:]
                            post_stress = stress_cut[es_e_new:]
                            search_end = min(len(post_stress), max(int(len(post_stress) * 0.2), 20))
                            upper_idx = np.argmax(post_stress[:search_end])
                            upper_strain = post_strain[upper_idx]
                            
                            # 下屈服点应变
                            lower_search = post_stress[upper_idx:]
                            lower_idx = np.argmin(lower_search) + upper_idx
                            lower_strain = post_strain[lower_idx]
                            
                            ax.plot(upper_strain, upper_yield, 'g^', markersize=10, zorder=5)
                            ax.annotate(f'上屈服 = {upper_yield:.0f} MPa',
                                       xy=(upper_strain, upper_yield),
                                       xytext=(15, 15), textcoords='offset points',
                                       fontsize=10, color='green', fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                                       arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
                            
                            ax.plot(lower_strain, lower_yield, 'bs', markersize=10, zorder=5)
                            ax.annotate(f'下屈服 = {lower_yield:.0f} MPa',
                                       xy=(lower_strain, lower_yield),
                                       xytext=(15, -20), textcoords='offset points',
                                       fontsize=10, color='blue', fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                                       arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

            # 延伸率
            elong = props.get('elongation', 0)
            ax.annotate(f'延伸率 = {elong:.1f}%',
                       xy=(strain_cut[-1], stress_cut[-1]),
                       xytext=(-80, 20), textcoords='offset points',
                       fontsize=10, color='purple', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

        else:
            ax.plot(strain, stress, 'b-', linewidth=0.8)
            ax.autoscale(True)
        ax.set_xlabel('应变 (%)')
        ax.set_ylabel('应力 (MPa)')
        ax.margins(x=0.05, y=0.05)
        ax.set_title(f'③ 截断 (pt{cut_idx}) + 力学性能', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        # 限制纵坐标范围为最大应力的120%
        if cut_idx > 0:
            max_stress = np.max(stress_cut)
            ax.set_ylim(0, max_stress * 1.2)

        # --- 阶段4: 替换滑动段 + 性能汇总 ---
        ax = axes[1, 1]
        if specimen_corrected is not None and specimen_corrected.stress is not None:
            ax.plot(specimen_corrected.strain, specimen_corrected.stress, 'g-', linewidth=1.2)
            ax.autoscale(True)
        else:
            ax.plot(strain, stress, 'g-', linewidth=0.8)
            ax.autoscale(True)
        ax.set_xlabel('应变 (%)')
        ax.set_ylabel('应力 (MPa)')
        ax.margins(x=0.05, y=0.05)
        ax.set_title('④ 替换滑动段 (最终曲线)', fontsize=12, fontweight='bold')
        
        # 右下角显示力学性能汇总
        E_val = props.get('E', 0)
        uts_val = props.get('UTS', 0)
        yield_val = props.get('yield')
        yield_type = props.get('yield_type', '')
        elong = props.get('elongation', 0)
        
        summary_lines = [f'弹性模量 E = {E_val:.0f} MPa']
        summary_lines.append(f'抗拉强度 UTS = {uts_val:.0f} MPa')
        if yield_val is not None:
            summary_lines.append(f'屈服强度 {yield_type} = {yield_val:.0f} MPa')
        summary_lines.append(f'延伸率 = {elong:.1f}%')
        
        summary_text = '\n'.join(summary_lines)
        ax.text(0.98, 0.03, summary_text, transform=ax.transAxes,
               fontsize=11, va='bottom', ha='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

        fig.suptitle(f'弹性段诊断与力学性能: {sp.name}', fontsize=16, fontweight='bold')
        fig.tight_layout()
        return fig


# ============================================================
# 绘图引擎
# ============================================================
class PlotEngine:
    """绘图引擎"""

    def __init__(self, style: str = 'default', dpi: int = 300):
        self.colors = COLOR_SCHEMES.get(style, COLOR_SCHEMES['default'])
        self.dpi = dpi
        self._setup_style()

    def _setup_style(self):
        """配置绘图样式"""
        plt.rcParams.update({
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'font.size': 11,
            'axes.labelsize': 13,
            'axes.titlesize': 14,
            'legend.fontsize': 9,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'lines.linewidth': 1.5,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            # 四边轴线全部显示
            'axes.spines.top': True,
            'axes.spines.right': True,
            # 主次刻度朝内
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'xtick.top': False,      # 上轴不加刻度线
            'ytick.right': False,    # 右轴不加刻度线
        })

    def _get_color(self, index: int) -> str:
        return self.colors[index % len(self.colors)]

    def _get_linestyle(self, index: int, mode: str = 'tensile') -> str:
        """不同试样用不同线型增加区分度"""
        if mode == 'compression':
            styles = ['-', '--', '-.', ':']
        else:
            styles = ['-'] * 20  # 拉伸默认全实线
        return styles[index % len(styles)]

    def plot_overlay(self, specimens: List[SpecimenData],
                     title: str = '应力-应变曲线',
                     xlabel: str = '工程应变 (%)',
                     ylabel: str = '应力 (MPa)',
                     show_uts: bool = False,
                     show_legend: bool = True,
                     figsize: Tuple[float, float] = (10, 7),
                     xlim: Optional[Tuple[float, float]] = None,
                     ylim: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """叠加所有曲线到一张图"""
        fig, ax = plt.subplots(figsize=figsize)

        for i, sp in enumerate(specimens):
            if sp.stress is None or sp.strain is None:
                print(f"  ⚠ 跳过 {sp.name}（无应力-应变数据）")
                continue

            color = self._get_color(i)
            ls = self._get_linestyle(i, sp.mode)
            label = sp.name
            if sp.composition:
                label = f"{sp.composition} - {sp.name}"
            if sp.treatment:
                label += f" ({sp.treatment})"

            ax.plot(sp.strain, sp.stress, color=color, linestyle=ls,
                    label=label, alpha=0.85)

            # 标注 UTS
            if show_uts and sp.uts:
                uts_idx = np.argmax(sp.stress)
                ax.annotate(f'{sp.uts:.0f}',
                           xy=(sp.strain[uts_idx], sp.stress[uts_idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color=color, fontweight='bold')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if show_legend:
            ax.legend(loc='best', framealpha=0.9, edgecolor='gray')

        # 自动轴范围
        if xlim:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(left=0)
        if ylim:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)

        fig.tight_layout()
        return fig

    def plot_subplots(self, specimens: List[SpecimenData],
                      title: str = '应力-应变曲线',
                      cols: int = 3,
                      figsize_per: Tuple[float, float] = (4, 3.5),
                      show_uts: bool = True,
                      show_info: bool = True,
                      props_dict: Dict = None) -> plt.Figure:
        """
        每个试样一个子图

        Args:
            props_dict: {试样名: {'E', 'UTS', 'yield', 'yield_type', 'elongation'}}
                        如果提供，在右下角显示完整力学性能标注
        """
        n = len(specimens)
        rows = (n + cols - 1) // cols
        figsize = (figsize_per[0] * cols, figsize_per[1] * rows)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, sp in enumerate(specimens):
            ax = axes[i]
            if sp.stress is None or sp.strain is None:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title(sp.name)
                continue

            color = self._get_color(i)
            ax.plot(sp.strain, sp.stress, color=color, linewidth=1.5)

            ax.set_xlabel('应变 (%)')
            ax.set_ylabel('应力 (MPa)')
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

            # 标题
            title_text = sp.name
            if sp.composition:
                title_text = f"{sp.composition} - {sp.name}"
            ax.set_title(title_text, fontsize=10)

            # 右下角力学性能标注
            props = props_dict.get(sp.name) if props_dict else None
            if props:
                info_lines = []
                E_val = props.get('E', 0)
                if E_val > 0:
                    info_lines.append(f'E = {E_val:.0f} MPa')
                uts_val = props.get('UTS', 0)
                if uts_val > 0:
                    info_lines.append(f'UTS = {uts_val:.0f} MPa')
                yield_val = props.get('yield')
                yield_type = props.get('yield_type', '')
                if yield_val is not None:
                    info_lines.append(f'{yield_type} = {yield_val:.0f} MPa')
                elong = props.get('elongation', 0)
                if elong > 0:
                    info_lines.append(f'ε = {elong:.1f}%')

                if info_lines:
                    info_text = '\n'.join(info_lines)
                    ax.text(0.97, 0.03, info_text, transform=ax.transAxes,
                           fontsize=8, va='bottom', ha='right',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.85))
            elif show_info and sp.uts:
                # 降级: 无详细props时只显示UTS和延伸率
                info_lines = [f'UTS = {sp.uts:.1f} MPa']
                if sp.elongation:
                    info_lines.append(f'ε = {sp.elongation:.1f}%')
                info_text = '\n'.join(info_lines)
                ax.text(0.97, 0.03, info_text, transform=ax.transAxes,
                       fontsize=8, va='bottom', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

            # UTS 标注点
            if show_uts and sp.uts:
                uts_idx = np.argmax(sp.stress)
                ax.plot(sp.strain[uts_idx], sp.stress[uts_idx], 'v',
                       color=color, markersize=6)

        # 隐藏多余子图
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

        fig.tight_layout()
        return fig

    def plot_comparison(self, specimens: List[SpecimenData],
                        title: str = '力学性能对比',
                        properties: List[str] = None) -> plt.Figure:
        """柱状图对比力学性能"""
        if properties is None:
            properties = ['uts']
            if any(s.elongation for s in specimens):
                properties.append('elongation')

        prop_labels = {
            'uts': '抗拉强度 (MPa)',
            'elongation': '断后伸长率 (%)',
            'yield_strength': '屈服强度 (MPa)',
        }

        n_props = len(properties)
        fig, axes = plt.subplots(1, n_props, figsize=(5 * n_props, 5))
        if n_props == 1:
            axes = [axes]

        names = [sp.name for sp in specimens]
        x = np.arange(len(names))

        for pi, prop in enumerate(properties):
            ax = axes[pi]
            values = []
            for sp in specimens:
                val = getattr(sp, prop, None)
                values.append(val if val is not None else 0)

            bars = ax.bar(x, values, color=[self._get_color(i) for i in range(len(specimens))],
                         alpha=0.8, edgecolor='white', linewidth=0.5)

            # 数值标注
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(prop_labels.get(prop, prop))
            ax.set_ylim(bottom=0)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig


# ============================================================
# 批量处理引擎
# ============================================================
class BatchProcessor:
    """批量数据处理"""

    SUPPORTED_EXTENSIONS = {'.lst', '.csv', '.txt', '.dat', '.tsv'}

    def __init__(self, mode: str = 'tensile',
                 width: float = 0, thickness: float = 0,
                 gauge_length: float = 0,
                 cross_section: float = 0,
                 specimen_params: Dict = None):
        """
        Args:
            mode: 'tensile' 或 'compression'
            width: 默认宽度 (mm)，所有试样共用
            thickness: 默认厚度 (mm)
            gauge_length: 默认标距 (mm)
            cross_section: 默认截面积 (mm²)，优先级低于 width*thickness
            specimen_params: 每个试样的独立参数 {文件名: {width, thickness, ...}}
        """
        self.mode = mode
        self.default_width = width
        self.default_thickness = thickness
        self.default_gauge_length = gauge_length
        self.default_cross_section = cross_section
        self.specimen_params = specimen_params or {}

    def find_data_files(self, path: str) -> List[str]:
        """查找目录下所有支持的数据文件"""
        path = Path(path)
        if path.is_file():
            return [str(path)]

        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(glob.glob(str(path / f'*{ext}')))

        # 排序：自然排序 (1, 2, ..., 10 而非 1, 10, 2)
        def natural_sort_key(s):
            return [int(c) if c.isdigit() else c.lower()
                    for c in re.split(r'(\d+)', str(s))]

        files.sort(key=natural_sort_key)
        return files

    def load_specimen(self, filepath: str, specimen_name: str = None) -> SpecimenData:
        """加载单个试样数据"""
        filepath = str(filepath)
        fmt = DataParser.detect_format(filepath)

        # 解析
        if fmt == 'lst':
            raw_data = DataParser.parse_lst(filepath)
            header_info = DataParser.extract_header_info(filepath)
        elif fmt == 'csv':
            raw_data = DataParser.parse_csv(filepath)
            header_info = DataParser.extract_header_info(filepath)
        else:
            raw_data = DataParser.parse_auto(filepath)
            header_info = DataParser.extract_header_info(filepath)

        if specimen_name is None:
            specimen_name = Path(filepath).stem

        ncols = raw_data.shape[1]

        # === 检测列含义 ===
        col_map = DataParser.detect_columns(filepath)

        has_stress = 'stress' in col_map
        has_strain = 'strain' in col_map
        has_load = 'load' in col_map
        has_disp = 'disp' in col_map

        if has_load and has_disp:
            # 有载荷和位移列
            load = raw_data[:, col_map['load']]
            displacement = raw_data[:, col_map['disp']]
            time_col = col_map.get('time')
            time = raw_data[:, time_col] if time_col is not None else np.arange(len(load))
        elif ncols >= 4 and not has_stress:
            # .lst 格式: load, disp1, disp2, time
            load = raw_data[:, 0]
            displacement = raw_data[:, 1]
            time = raw_data[:, 3]
        elif ncols >= 2:
            # 降级: 假设前两列为 load, displacement
            load = raw_data[:, 0]
            displacement = raw_data[:, 1]
            time = np.arange(len(load))
        else:
            raise ValueError(f"数据列数不足: {ncols}")

        # 试样参数
        sp_params = self.specimen_params.get(specimen_name, {})
        width = sp_params.get('width', self.default_width) or header_info.get('width', 0)
        thickness = sp_params.get('thickness', self.default_thickness) or header_info.get('thickness', 0)
        gauge = sp_params.get('gauge_length', self.default_gauge_length) or header_info.get('gauge_length', 0)
        cross = sp_params.get('cross_section', self.default_cross_section)

        specimen = SpecimenData(
            name=specimen_name,
            filepath=filepath,
            load=load,
            displacement=displacement,
            time=time,
            width=width,
            thickness=thickness,
            gauge_length=gauge,
            cross_section=cross if cross > 0 else 0,
            mode=self.mode,
            composition=sp_params.get('composition', ''),
            treatment=sp_params.get('treatment', ''),
        )

        return specimen

    def load_batch(self, path: str) -> List[SpecimenData]:
        """批量加载"""
        files = self.find_data_files(path)
        if not files:
            raise FileNotFoundError(f"在 {path} 中未找到数据文件")

        print(f"📂 找到 {len(files)} 个数据文件:")
        specimens = []
        for f in files:
            name = Path(f).stem
            print(f"  📄 {Path(f).name}", end='')
            try:
                sp = self.load_specimen(f, name)
                sp = DataProcessor.convert_to_stress_strain(sp)
                sp = DataProcessor.normalize_strain_start(sp)
                specimens.append(sp)
                print(f"  ✅ {len(sp.load)} 点, UTS={sp.uts:.1f} MPa")
            except Exception as e:
                print(f"  ❌ {e}")

        if not specimens:
            raise RuntimeError("所有文件加载失败")

        return specimens


# ============================================================
# 数据导出
# ============================================================
def export_summary_csv(specimens: List[SpecimenData], output_path: str):
    """导出力学性能汇总 CSV"""
    lines = ['试样名称,成分,处理工艺,抗拉强度(MPa),断后伸长率(%),数据点数,文件路径']
    for sp in specimens:
        lines.append(f'{sp.name},{sp.composition},{sp.treatment},'
                    f'{sp.uts:.1f},{sp.elongation:.1f},{len(sp.load)},{sp.filepath}')
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(lines))
    print(f"📊 汇总表已保存: {output_path}")


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='mech_plot — 拉伸/压缩数据批量可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 拉伸数据叠加图
  python mech_plot.py D:\\实验数据\\拉伸 --overlay -o 拉伸汇总.png

  # 压缩数据子图模式
  python mech_plot.py D:\\实验数据\\压缩 --mode compression --subplot

  # 指定试样尺寸 (所有试样共用)
  python mech_plot.py data/ --width 3.5 --thickness 1.7 --gauge 20

  # 带 UTS 标注和性能对比图
  python mech_plot.py data/ --overlay --show-uts --comparison

  # 导出数据汇总表
  python mech_plot.py data/ --overlay --csv-summary

  # 自动修正低质量弹性段
  python mech_plot.py data/ --fix-elastic --fix-elastic-plot
        """
    )

    parser.add_argument('path', help='数据文件或目录路径')
    parser.add_argument('--mode', choices=['tensile', 'compression'], default='tensile',
                       help='测试类型 (默认: tensile)')
    parser.add_argument('--overlay', action='store_true', default=True,
                       help='叠加模式 (默认)')
    parser.add_argument('--subplot', action='store_true',
                       help='子图模式 (每个试样一个子图)')
    parser.add_argument('--comparison', action='store_true',
                       help='生成力学性能柱状对比图')
    parser.add_argument('-o', '--output', default=None,
                       help='输出文件路径 (默认: 数据目录/mech_plot_output.png)')
    parser.add_argument('--cols', type=int, default=3,
                       help='子图模式每行列数 (默认: 3)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='输出分辨率 (默认: 300)')
    parser.add_argument('--style', choices=list(COLOR_SCHEMES.keys()), default='default',
                       help='配色方案 (默认: default)')
    parser.add_argument('--title', default=None,
                       help='图表标题')
    parser.add_argument('--show-uts', action='store_true',
                       help='标注抗拉强度')
    parser.add_argument('--no-legend', action='store_true',
                       help='不显示图例')
    parser.add_argument('--xlim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='X轴范围')
    parser.add_argument('--ylim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='Y轴范围')
    parser.add_argument('--csv-summary', action='store_true',
                       help='导出力学性能汇总 CSV')

    # 弹性段修正
    parser.add_argument('--fix-elastic', action='store_true',
                       help='自动检测并修正低质量弹性段 (载荷下降、低斜率等)')
    parser.add_argument('--fix-elastic-plot', action='store_true',
                       help='生成弹性段修正诊断图 (每个异常试样一张)')
    parser.add_argument('--preview', action='store_true',
                       help='实时预览四阶段诊断图 (交互模式，逐个显示)')

    # 试样参数
    parser.add_argument('--width', type=float, default=0,
                       help='试样宽度 (mm)')
    parser.add_argument('--thickness', type=float, default=0,
                       help='试样厚度 (mm)')
    parser.add_argument('--gauge', type=float, default=0,
                       help='标距 (mm)')
    parser.add_argument('--area', type=float, default=0,
                       help='截面积 (mm²)，优先于 width*thickness')
    parser.add_argument('--elastic-modulus', type=float, default=0,
                       help='真实弹性模量 (GPa)，用于修正应变/位移')
    parser.add_argument('--specimen-info', default=None,
                       help='试样信息CSV文件路径 (自动匹配列名)')
    parser.add_argument('--interactive', action='store_true',
                       help='交互模式: 无试样信息时手动输入参数')

    args = parser.parse_args()

    # 处理参数
    path = args.path
    if not os.path.exists(path):
        print(f"❌ 路径不存在: {path}")
        sys.exit(1)

    # 输出路径
    if args.output:
        output_path = args.output
    else:
        if os.path.isdir(path):
            output_dir = path
        else:
            output_dir = str(Path(path).parent)
        output_path = os.path.join(output_dir, 'mech_plot_output.png')

    # 标题
    mode_cn = '拉伸' if args.mode == 'tensile' else '压缩'
    default_title = f'{mode_cn}应力-应变曲线'
    title = args.title or default_title

    print(f"🔧 mech_plot — {mode_cn}数据批量可视化")
    print(f"   路径: {path}")
    print(f"   尺寸: {args.width}×{args.thickness} mm, 标距: {args.gauge} mm")
    print()

    # === 加载试样信息 ===
    specimen_info = {}
    info_path = args.specimen_info

    # 自动查找试样信息.csv
    if info_path is None and os.path.isdir(path):
        auto_candidates = ['试样信息.csv', 'specimen_info.csv', '试样信息.txt']
        for candidate in auto_candidates:
            auto_path = os.path.join(path, candidate)
            if os.path.exists(auto_path):
                info_path = auto_path
                break

    if info_path and os.path.exists(info_path):
        specimen_info = DataParser.parse_specimen_info_csv(info_path)
        if specimen_info:
            print(f"📋 已加载试样信息: {info_path} ({len(specimen_info)} 条)")
        else:
            print(f"⚠ 试样信息文件为空或解析失败: {info_path}")

    # 交互模式: 无试样信息且未指定尺寸时，手动输入
    if args.interactive and not specimen_info and args.width == 0 and args.thickness == 0:
        print("\n📝 交互模式: 请输入试样参数 (回车跳过)")
        try:
            w_input = input("  宽度 (mm): ").strip()
            t_input = input("  厚度 (mm): ").strip()
            g_input = input("  标距 (mm): ").strip()
            if w_input:
                args.width = float(w_input)
            if t_input:
                args.thickness = float(t_input)
            if g_input:
                args.gauge = float(g_input)
            print(f"  → 尺寸: {args.width}×{args.thickness} mm, 标距: {args.gauge} mm\n")
        except (EOFError, KeyboardInterrupt):
            print("\n  跳过手动输入。")

    # 批量加载
    processor = BatchProcessor(
        mode=args.mode,
        width=args.width,
        thickness=args.thickness,
        gauge_length=args.gauge,
        cross_section=args.area,
        specimen_params=specimen_info,
    )
    specimens = processor.load_batch(path)

    print(f"\n✅ 成功加载 {len(specimens)} 个试样")

    # 弹性模量修正
    E_true = args.elastic_modulus * 1000  # GPa → MPa
    if E_true > 0:
        print(f"\n🔧 弹性模量修正: 目标 E = {args.elastic_modulus} GPa ({E_true:.0f} MPa)")
        for sp in specimens:
            DataProcessor.correct_elastic_modulus(sp, E_true)

    # 弹性段检测与修正
    diagnostics = {}
    if args.fix_elastic or args.fix_elastic_plot or args.preview:
        print("\n🔍 弹性段质量检测...")
        corrector = ElasticRegionCorrector()
        specimens_orig = []  # 保存原始副本用于诊断图
        for i, sp in enumerate(specimens):
            sp_copy = SpecimenData(
                name=sp.name, filepath=sp.filepath,
                load=sp.load.copy(), displacement=sp.displacement.copy(),
                time=sp.time.copy(),
                stress=sp.stress.copy() if sp.stress is not None else None,
                strain=sp.strain.copy() if sp.strain is not None else None,
                width=sp.width, thickness=sp.thickness,
                gauge_length=sp.gauge_length, cross_section=sp.cross_section,
                mode=sp.mode, composition=sp.composition, treatment=sp.treatment,
            )
            specimens_orig.append(sp_copy)

            diag = corrector.diagnose(sp)
            diagnostics[sp.name] = diag  # 保存所有诊断结果
            if diag.issues:
                print(f"  ⚠ {sp.name}: {'; '.join(diag.issues)}")
                if args.fix_elastic:
                    sp, diag = corrector.correct(sp, diag)
                    diagnostics[sp.name] = diag
                    print(f"    → 修正: 替换{diag.n_replaced}点, E={diag.elastic_modulus:.0f} MPa")
            else:
                print(f"  ✅ {sp.name}: 弹性段正常 (E={diag.elastic_modulus:.0f} MPa)")

        if args.fix_elastic:
            n_fixed = sum(1 for d in diagnostics.values() if d.corrected)
            print(f"\n  修正了 {n_fixed}/{len(specimens)} 个试样")

        # 实时预览模式
        if args.preview:
            print("\n📊 实时预览四阶段诊断图...")
            print("   (按 Enter 显示下一个试样，输入 q 退出预览)")
            
            import matplotlib
            matplotlib.use('TkAgg')  # 使用交互式后端
            import matplotlib.pyplot as plt_interactive
            plt_interactive.ion()  # 开启交互模式
            
            for i, sp_orig in enumerate(specimens_orig):
                name = sp_orig.name
                diag = diagnostics.get(name)
                if diag is None:
                    continue
                
                # 获取修正后的试样数据
                sp_corrected = specimens[i] if i < len(specimens) else None
                
                # 提取力学性能
                props = corrector.get_mechanical_properties(sp_orig, diag)
                
                # 修正数据后再计算一次性能
                if diag.issues and args.fix_elastic and sp_corrected:
                    props_corrected = corrector.get_mechanical_properties(sp_corrected, diag)
                    # 用修正后的性能覆盖
                    props.update(props_corrected)
                
                # 绘制诊断图
                fig = corrector.plot_diagnostic_with_props(sp_orig, sp_corrected, diag, props)
                plt_interactive.show()
                
                # 打印性能汇总
                print(f"\n  【{name}】力学性能:")
                print(f"    弹性模量 E = {props.get('E', 0):.0f} MPa")
                print(f"    抗拉强度 UTS = {props.get('UTS', 0):.0f} MPa")
                yield_val = props.get('yield')
                yield_type = props.get('yield_type', '')
                if yield_val is not None:
                    print(f"    屈服强度 {yield_type} = {yield_val:.0f} MPa")
                print(f"    延伸率 = {props.get('elongation', 0):.1f}%")
                
                # 等待用户输入
                if i < len(specimens_orig) - 1:
                    user_input = input(f"\n  按 Enter 查看下一个试样 (q退出): ").strip().lower()
                    if user_input == 'q':
                        break
                else:
                    print("\n  所有试样预览完毕。")
                    input("  按 Enter 关闭预览窗口...")
            
            plt_interactive.ioff()  # 关闭交互模式
            plt_interactive.close('all')

        # 生成诊断图
        if args.fix_elastic_plot:
            print("\n📊 生成弹性段诊断图...")
            for i, sp_orig in enumerate(specimens_orig):
                name = sp_orig.name
                diag = diagnostics.get(name)
                if diag is None:
                    continue
                
                # 获取修正后的试样数据
                sp_corrected = specimens[i] if i < len(specimens) else None
                
                # 提取力学性能
                props = corrector.get_mechanical_properties(sp_orig, diag)
                if diag.issues and args.fix_elastic and sp_corrected:
                    props_corrected = corrector.get_mechanical_properties(sp_corrected, diag)
                    props.update(props_corrected)
                
                # 绘制带性能标注的诊断图
                fig_diag = corrector.plot_diagnostic_with_props(sp_orig, sp_corrected, diag, props)
                diag_path = str(Path(output_path).parent / f'elastic_diag_{name}.png')
                fig_diag.savefig(diag_path, bbox_inches='tight', facecolor='white')
                print(f"  📊 {name}: {diag_path}")
                
                # 打印性能
                print(f"     E={props.get('E', 0):.0f} MPa, UTS={props.get('UTS', 0):.0f} MPa, "
                      f"延伸率={props.get('elongation', 0):.1f}%")
            plt.close('all')

    # 绘图
    plotter = PlotEngine(style=args.style, dpi=args.dpi)
    xlim = tuple(args.xlim) if args.xlim else None
    ylim = tuple(args.ylim) if args.ylim else None

    # 生成叠加图
    print("\n🎨 正在绘图...")
    fig1 = plotter.plot_overlay(
        specimens,
        title=title,
        show_uts=args.show_uts,
        show_legend=not args.no_legend,
        xlim=xlim,
        ylim=ylim,
    )
    overlay_path = output_path
    fig1.savefig(overlay_path, bbox_inches='tight', facecolor='white')
    print(f"  📈 叠加图: {overlay_path}")

    # 子图模式
    if args.subplot:
        subplot_path = Path(output_path).stem + '_subplots' + Path(output_path).suffix
        subplot_path = str(Path(output_path).parent / subplot_path)
        # 如果有弹性段修正的力学性能，传递给子图
        props_for_subplots = None
        if diagnostics:
            corrector = ElasticRegionCorrector()
            props_for_subplots = {}
            for sp in specimens:
                diag = diagnostics.get(sp.name)
                if diag:
                    props = corrector.get_mechanical_properties(sp, diag)
                    props_for_subplots[sp.name] = props
        fig2 = plotter.plot_subplots(specimens, title=title, cols=args.cols,
                                     show_uts=args.show_uts, props_dict=props_for_subplots)
        fig2.savefig(subplot_path, bbox_inches='tight', facecolor='white')
        print(f"  📊 子图: {subplot_path}")

    # 对比图
    if args.comparison:
        comp_path = Path(output_path).stem + '_comparison' + Path(output_path).suffix
        comp_path = str(Path(output_path).parent / comp_path)
        fig3 = plotter.plot_comparison(specimens, title=f'{mode_cn}力学性能对比')
        fig3.savefig(comp_path, bbox_inches='tight', facecolor='white')
        print(f"  📊 对比图: {comp_path}")

    # CSV 汇总
    if args.csv_summary:
        csv_path = str(Path(output_path).with_suffix('.csv'))
        export_summary_csv(specimens, csv_path)

    plt.close('all')
    print("\n✅ 完成!")


if __name__ == '__main__':
    main()
