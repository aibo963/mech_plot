# mech_plot

拉伸/压缩数据批量可视化工具，支持弹性段自动检测与修正。

## 功能

- **多格式支持**：`.lst`（WDT-20）、`.csv`、`.txt`，自动识别列名和表头
- **弹性段自动修正**（v8算法）：全范围扫描 → 精炼 → 截断 → 残差替换
- **四阶段诊断图**：原始曲线 → 弹性段检测 → 截断+力学性能 → 最终曲线
- **力学性能自动提取**：弹性模量 E、抗拉强度 UTS、屈服强度（R0.2 / 上下屈服）、延伸率
- **试样信息CSV**：自动解析试样尺寸、成分、工艺
- **GUI 界面**：四面板诊断图 + 一键处理 + 批量穿透子目录 + 图片/数据导出
- **字体**：中文宋体 + 英文 Times New Roman

## 安装

```bash
pip install numpy matplotlib
pip install tkinterdnd2  # 可选，拖拽支持
```

## 使用

### CLI

```bash
# 基本用法（自动从表头提取尺寸）
python mech_plot.py D:\实验数据\拉伸

# 指定试样尺寸
python mech_plot.py D:\实验数据\拉伸 --width 3.5 --thickness 1.7 --gauge 20

# 弹性段修正 + 诊断图
python mech_plot.py D:\实验数据\拉伸 --fix-elastic --fix-elastic-plot

# 子图模式 + 性能对比
python mech_plot.py D:\实验数据\拉伸 --subplot --comparison --show-uts

# 导出CSV汇总
python mech_plot.py D:\实验数据\拉伸 --csv-summary
```

### GUI

```bash
python mech_plot_gui.py
```

或双击 `启动GUI.bat`。

## 试样信息CSV

在数据目录下放置 `试样信息.csv`，自动匹配列名：

```csv
编号,宽度_mm,厚度_mm,长度_mm,成分,处理
1,3.56,1.69,20,Fe-17Mn-0.4Si,冷轧退火
2,3.95,1.53,20,Fe-17Mn-0.4Si,固溶
```

支持的列名：`试样名称/编号`、`宽度/width`、`厚度/thickness`、`标距/长度/gauge`、`成分/composition`、`处理工艺/处理/treatment`

## 弹性段修正算法

针对拉伸数据中常见的夹具滑移、引伸计打滑等问题：

1. **全范围扫描**：滑动窗口找 R²>0.98 且 slope×R²² 最高的段 = 弹性段核心
2. **精炼**：核心内找 R² 最高的子段（排除滑动→弹性过渡区）
3. **截断**：弹性趋势线外推到 load=0 处截断，设为应变零点
4. **替换**：趋势线 + 模板残差替换滑动段

## 目录结构

```
mech_plot/
├── mech_plot.py           # CLI 主程序
├── mech_plot_gui.py       # GUI 界面
├── specimens_template.yaml # 试样信息模板
├── 启动GUI.bat            # Windows 快捷启动
└── README.md
```

## 许可

MIT
