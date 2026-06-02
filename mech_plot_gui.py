#!/usr/bin/env python3
"""
mech_plot GUI v4 — 简洁布局 + 固定图框 + 作图按钮修复
"""
import os
import sys
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mech_plot import (BatchProcessor, PlotEngine, COLOR_SCHEMES,
                       DataParser, ElasticRegionCorrector, DataProcessor,
                       SpecimenData, setup_fonts)

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

try:
    from tkinterdnd2 import TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False


def find_all_data_files(root_path):
    exts = {'.lst', '.csv', '.txt', '.dat', '.tsv'}
    found = {}
    for dirpath, _, filenames in os.walk(root_path):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts and fn.lower() not in ('试样信息.csv', 'specimen_info.csv'):
                full = os.path.join(dirpath, fn)
                key = (fn, os.path.getsize(full))
                if key not in found:
                    found[key] = full
    def nsort(fname):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', fname)]
    return [v for (fn, _), v in sorted(found.items(), key=lambda x: nsort(x[0][0]))]


class DiagPanel:
    """单个诊断子图面板"""
    def __init__(self, parent, title, row, col, on_plot=None):
        self.title = title
        self.fig = None
        self._data_cache = None
        self._on_plot = on_plot

        self.frame = ttk.LabelFrame(parent, text=title, padding=3)
        self.frame.grid(row=row, column=col, sticky='nsew', padx=2, pady=2)
        self.frame.grid_propagate(False)

        # 按钮栏
        bar = ttk.Frame(self.frame)
        bar.pack(fill='x')
        ttk.Button(bar, text='作图', command=self._do_plot, width=5).pack(side='left', padx=1)
        ttk.Button(bar, text='保存', command=self._save, width=5).pack(side='left', padx=1)
        ttk.Button(bar, text='导出数据', command=self._export, width=7).pack(side='left', padx=1)

        # 画布容器
        self._canvas_frame = ttk.Frame(self.frame)
        self._canvas_frame.pack(fill='both', expand=True)
        self._canvas = None
        self._placeholder = ttk.Label(self._canvas_frame, text='等待数据...', foreground='gray')
        self._placeholder.pack(fill='both', expand=True)

    def show(self, fig, data=None):
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self.fig:
            plt.close(self.fig)
        self._placeholder.pack_forget()

        self.fig = fig
        self._data_cache = data
        self._canvas = FigureCanvasTkAgg(fig, master=self._canvas_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

    def clear(self):
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self.fig:
            plt.close(self.fig)
            self.fig = None
        self._data_cache = None
        self._placeholder.pack(fill='both', expand=True)

    def _do_plot(self):
        if self._on_plot:
            threading.Thread(target=self._on_plot, daemon=True).start()

    def _save(self):
        if not self.fig:
            return
        path = filedialog.asksaveasfilename(
            title=f'保存 {self.title}', defaultextension='.png',
            filetypes=[('PNG','*.png'),('PDF','*.pdf'),('SVG','*.svg')],
            initialfile=f'{self.title}.png')
        if path:
            self.fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=200)

    def _export(self):
        if not self._data_cache:
            messagebox.showinfo('提示', '无可导出数据')
            return
        path = filedialog.asksaveasfilename(
            title=f'导出 {self.title}', defaultextension='.csv',
            filetypes=[('CSV','*.csv'),('TXT','*.txt')],
            initialfile=f'{self.title}.csv')
        if path:
            strain, stress = self._data_cache
            with open(path, 'w', encoding='utf-8-sig') as f:
                f.write('应变(%),应力(MPa)\n')
                for s, st in zip(strain, stress):
                    f.write(f'{s:.4f},{st:.2f}\n')


class MechPlotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('mech_plot v4')
        self.root.geometry('1300x850')
        self.root.resizable(True, True)

        self.path_var = tk.StringVar()
        self.mode_var = tk.StringVar(value='tensile')
        self.width_var = tk.StringVar()
        self.thickness_var = tk.StringVar()
        self.gauge_var = tk.StringVar()
        self.E_var = tk.StringVar()  # 弹性模量 (GPa)
        self.dpi_var = tk.StringVar(value='150')
        self.style_var = tk.StringVar(value='default')
        self.show_uts_var = tk.BooleanVar(value=False)
        self.fix_elastic_var = tk.BooleanVar(value=True)
        self.export_dir_var = tk.StringVar()
        self.status_var = tk.StringVar(value='就绪')

        self.specimens = []
        self.specimens_orig = []
        self.diagnostics = {}
        self.props_dict = {}
        self.corrector = ElasticRegionCorrector()

        self._build_ui()

        # 自动加载
        self._auto_load_job = None
        self.path_var.trace_add('write', self._on_path_changed)

    def _build_ui(self):
        root = self.root

        paned = ttk.PanedWindow(root, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=4, pady=4)

        # ========== 左侧 (不用滚动，直接pack) ==========
        left = ttk.Frame(paned, width=300)
        paned.add(left, weight=0)

        pad = {'padx': 4, 'pady': 2}

        # 路径
        frm = ttk.LabelFrame(left, text='数据路径', padding=4)
        frm.pack(fill='x', **pad)
        self.path_entry = ttk.Entry(frm, textvariable=self.path_var)
        self.path_entry.pack(fill='x')
        if HAS_DND:
            try:
                self.path_entry.drop_target_register('DND_Files')
                self.path_entry.dnd_bind('<<Drop>>', self._on_drop)
            except: pass
        r = ttk.Frame(frm); r.pack(fill='x', pady=2)
        ttk.Button(r, text='浏览文件', command=self._browse_file).pack(side='left', padx=1)
        ttk.Button(r, text='浏览目录', command=self._browse_dir).pack(side='left', padx=1)

        # 试样参数 (紧凑一行)
        frm = ttk.LabelFrame(left, text='试样参数 (留空=自动提取)', padding=4)
        frm.pack(fill='x', **pad)
        r = ttk.Frame(frm); r.pack(fill='x')
        ttk.Label(r, text='宽(mm)').pack(side='left')
        ttk.Entry(r, textvariable=self.width_var, width=6).pack(side='left', padx=1)
        ttk.Label(r, text='厚(mm)').pack(side='left', padx=(4,0))
        ttk.Entry(r, textvariable=self.thickness_var, width=6).pack(side='left', padx=1)
        ttk.Label(r, text='标距(mm)').pack(side='left', padx=(4,0))
        ttk.Entry(r, textvariable=self.gauge_var, width=6).pack(side='left', padx=1)
        r2 = ttk.Frame(frm); r2.pack(fill='x', pady=2)
        ttk.Label(r2, text='E(GPa)').pack(side='left')
        ttk.Entry(r2, textvariable=self.E_var, width=6).pack(side='left', padx=1)
        ttk.Label(r2, text='(留空不修正)', foreground='gray').pack(side='left', padx=4)
        ttk.Button(r2, text='确认更新', command=self._update_params).pack(side='right', padx=2)

        # 选项 (紧凑)
        frm = ttk.LabelFrame(left, text='选项', padding=4)
        frm.pack(fill='x', **pad)
        r = ttk.Frame(frm); r.pack(fill='x')
        ttk.Checkbutton(r, text='弹性段修正', variable=self.fix_elastic_var).pack(side='left')
        ttk.Checkbutton(r, text='标注UTS', variable=self.show_uts_var).pack(side='left', padx=6)
        r = ttk.Frame(frm); r.pack(fill='x', pady=1)
        ttk.Label(r, text='配色').pack(side='left')
        ttk.Combobox(r, textvariable=self.style_var, values=list(COLOR_SCHEMES.keys()),
                     state='readonly', width=8).pack(side='left', padx=2)
        ttk.Label(r, text='DPI').pack(side='left', padx=(4,0))
        ttk.Entry(r, textvariable=self.dpi_var, width=4).pack(side='left', padx=1)

        # 导出路径
        frm = ttk.LabelFrame(left, text='导出目录', padding=4)
        frm.pack(fill='x', **pad)
        r = ttk.Frame(frm); r.pack(fill='x')
        ttk.Entry(r, textvariable=self.export_dir_var).pack(side='left', fill='x', expand=True)
        ttk.Button(r, text='选择', command=self._browse_export).pack(side='left', padx=2)

        # 试样选择
        frm = ttk.LabelFrame(left, text='当前试样', padding=4)
        frm.pack(fill='x', **pad)
        self.specimen_combo = ttk.Combobox(frm, state='readonly')
        self.specimen_combo.pack(fill='x')
        self.specimen_combo.bind('<<ComboboxSelected>>', lambda e: self._show_diagnostic())

        # 操作
        frm = ttk.LabelFrame(left, text='操作', padding=4)
        frm.pack(fill='x', **pad)
        self.btn_run = ttk.Button(frm, text='▶ 一键处理', command=self._run)
        self.btn_run.pack(fill='x', pady=1)
        self.btn_batch = ttk.Button(frm, text='▶ 批量处理 (穿透子目录)', command=self._batch)
        self.btn_batch.pack(fill='x', pady=1)
        self.found_label = ttk.Label(frm, text='', foreground='blue', wraplength=280, justify='left')
        self.found_label.pack(anchor='w')

        # 日志 (填充剩余空间)
        frm = ttk.LabelFrame(left, text='日志', padding=4)
        frm.pack(fill='both', expand=True, **pad)
        self.log = tk.Text(frm, height=4, font=('Consolas', 9), wrap='word')
        lsb = ttk.Scrollbar(frm, command=self.log.yview)
        self.log.configure(yscrollcommand=lsb.set)
        lsb.pack(side='right', fill='y')
        self.log.pack(fill='both', expand=True)

        # ========== 右侧: 2x2 诊断图 ==========
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        grid = ttk.Frame(right)
        grid.pack(fill='both', expand=True)
        grid.grid_rowconfigure(0, weight=1)
        grid.grid_rowconfigure(1, weight=1)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)

        self.panels = [
            DiagPanel(grid, '① 原始曲线', 0, 0, on_plot=lambda: self._refresh_panel(0)),
            DiagPanel(grid, '② 弹性段检测', 0, 1, on_plot=lambda: self._refresh_panel(1)),
            DiagPanel(grid, '③ 截断 + 力学性能', 1, 0, on_plot=lambda: self._refresh_panel(2)),
            DiagPanel(grid, '④ 最终曲线 (修正后)', 1, 1, on_plot=lambda: self._refresh_panel(3)),
        ]

        ttk.Label(root, textvariable=self.status_var, relief='sunken', anchor='w').pack(fill='x', side='bottom')

    # === 路径 ===
    def _browse_file(self):
        p = filedialog.askopenfilename(filetypes=[('数据','*.lst *.csv *.txt *.dat'),('所有','*.*')])
        if p: self.path_var.set(p)

    def _browse_dir(self):
        p = filedialog.askdirectory()
        if p: self.path_var.set(p)

    def _browse_export(self):
        p = filedialog.askdirectory()
        if p: self.export_dir_var.set(p)

    def _update_params(self):
        """确认更新试样参数后重新计算"""
        name = self.specimen_combo.get()
        if not self.specimens:
            messagebox.showinfo('提示', '请先加载数据文件')
            return
        
        # 获取新参数
        try:
            w = float(self.width_var.get()) if self.width_var.get().strip() else None
            t = float(self.thickness_var.get()) if self.thickness_var.get().strip() else None
            g = float(self.gauge_var.get()) if self.gauge_var.get().strip() else None
            E = float(self.E_var.get()) if self.E_var.get().strip() else None
        except ValueError:
            messagebox.showerror('错误', '参数格式错误，请输入数字')
            return
        
        if not any([w, t, g]):
            messagebox.showinfo('提示', '请至少输入一个参数')
            return
        
        # 查找试样（列表结构）
        sp = None
        idx = None
        for i, s in enumerate(self.specimens):
            if s.name == name:
                sp = s
                idx = i
                break
        
        if sp is None:
            messagebox.showinfo('提示', f'未找到试样: {name}')
            return
        
        # 更新参数
        if w: sp.width = w
        if t: sp.thickness = t
        if g: sp.gauge_length = g
        if w and t: sp.cross_section = w * t
        
        # 重新计算应力应变
        from mech_plot import DataProcessor
        if sp.width and sp.thickness and sp.gauge_length:
            sp = DataProcessor.convert_to_stress_strain(sp)
            sp = DataProcessor.normalize_strain_start(sp)
            self.specimens[idx] = sp
            
            self._log(f'✅ {name} 参数已更新: {sp.width}×{sp.thickness}mm, 标距{sp.gauge_length}mm')
            # 重新显示
            self._show_diagnostic()
        

    def _on_drop(self, event):
        raw = event.data.strip().strip('{}')
        if os.path.exists(raw):
            self.path_var.set(raw)

    def _get_export_dir(self):
        d = self.export_dir_var.get().strip()
        if d and os.path.isdir(d): return d
        p = self.path_var.get().strip()
        return p if os.path.isdir(p) else os.path.dirname(p)

    def _log(self, msg):
        self.log.insert('end', msg + '\n')
        self.log.see('end')
        self.root.update_idletasks()

    def _get_panel_figsize(self, idx):
        """获取面板容器的实际像素尺寸，转换为figsize(英寸)"""
        try:
            w_px = self.panels[idx]._canvas_frame.winfo_width()
            h_px = self.panels[idx]._canvas_frame.winfo_height()
        except Exception:
            w_px, h_px = 0, 0
        if w_px < 50 or h_px < 50:
            # widget还未渲染，用窗口大小估算
            try:
                win_w = self.root.winfo_width()
                win_h = self.root.winfo_height()
                w_px = max(win_w // 2 - 30, 300)
                h_px = max(win_h // 2 - 50, 200)
            except Exception:
                w_px, h_px = 500, 350
        h_px = max(h_px - 35, 100)
        w_px = max(w_px - 10, 100)
        dpi = int(self.dpi_var.get() or 150)
        return (w_px / dpi, h_px / dpi)

    # === 自动加载 ===
    def _on_path_changed(self, *args):
        if self._auto_load_job:
            self.root.after_cancel(self._auto_load_job)
        self._auto_load_job = self.root.after(500, self._auto_load_preview)

    def _auto_load_preview(self):
        path = self.path_var.get().strip()
        if not path or not os.path.exists(path):
            return

        def worker():
            try:
                self._log(f'📂 {os.path.basename(path)}')
                specimens = self._load(path)
                if not specimens:
                    return
                self.specimens = specimens

                # 自动弹性段修正（跳过没有应力应变数据的试样）
                if self.fix_elastic_var.get():
                    self._log('🔍 弹性段修正...')
                    valid_specimens = [sp for sp in specimens if sp.stress is not None]
                    if valid_specimens:
                        self.diagnostics, self.props_dict, self.specimens_orig = self._correct(valid_specimens)

                names = [sp.name for sp in specimens]
                self.root.after(0, lambda: self.specimen_combo.configure(values=names))
                if names:
                    self.root.after(0, lambda: self.specimen_combo.set(names[0]))

                # 生成四张诊断图（只对有应力应变数据的试样）
                if names:
                    # 获取容器尺寸
                    self.root.after(0, lambda: None)  # 同步布局
                    fs0 = self._get_panel_figsize(0)
                    # 找到第一个有应力应变数据的试样
                    first_valid = next((sp for sp in specimens if sp.stress is not None), None)
                    if first_valid:
                        figs_data = self._make_figs(first_valid.name, figsize=fs0)
                        for i, (fig, data) in enumerate(figs_data):
                            if fig:
                                self.root.after(0, lambda f=fig, d=data, p=self.panels[i]: p.show(f, d))
                        self._log(f'  ✅ {first_valid.name}: {len(first_valid.load)}点 UTS={first_valid.uts:.0f}MPa')
                    else:
                        self._log(f'  ⚠️ 已加载 {len(specimens)} 个试样，请输入试样尺寸后点击"确认更新"')
                if len(specimens) > 1:
                    self._log(f'  ... 共{len(specimens)}个试样')
            except Exception as e:
                self._log(f'  ❌ {e}')

        threading.Thread(target=worker, daemon=True).start()

    # === 加载 ===
    def _load(self, path):
        w = float(self.width_var.get() or 0)
        t = float(self.thickness_var.get() or 0)
        g = float(self.gauge_var.get() or 0)

        specimen_info = {}
        if os.path.isdir(path):
            for c in ['试样信息.csv', 'specimen_info.csv']:
                fp = os.path.join(path, c)
                if os.path.exists(fp):
                    specimen_info = DataParser.parse_specimen_info_csv(fp)
                    if specimen_info:
                        self._log(f'📋 试样信息: {len(specimen_info)}条')
                    break

        processor = BatchProcessor(mode=self.mode_var.get(), width=w, thickness=t,
                                   gauge_length=g, specimen_params=specimen_info)
        files = [path] if os.path.isfile(path) else processor.find_data_files(path)

        specimens = []
        for f in files:
            name = Path(f).stem
            try:
                sp = processor.load_specimen(f, name)
                try:
                    sp = DataProcessor.convert_to_stress_strain(sp)
                    sp = DataProcessor.normalize_strain_start(sp)
                    specimens.append(sp)
                    self._log(f'  ✅ {name}: {len(sp.load)}点 UTS={sp.uts:.0f}MPa')
                except ValueError as e:
                    # 缺少尺寸信息，保存原始数据，等用户输入参数后再计算
                    specimens.append(sp)
                    self._log(f'  ⚠️ {name}: {len(sp.load)}点 (需要输入试样尺寸)')
            except Exception as e:
                self._log(f'  ❌ {name}: {e}')

        # 弹性模量修正
        E_gpa = float(self.E_var.get() or 0)
        if E_gpa > 0 and specimens:
            E_mpa = E_gpa * 1000
            self._log(f'🔧 E修正: 目标 {E_gpa} GPa')
            for sp in specimens:
                DataProcessor.correct_elastic_modulus(sp, E_mpa)
        return specimens

    def _correct(self, specimens):
        diagnostics = {}
        props_dict = {}
        specimens_orig = []
        for sp in specimens:
            orig = SpecimenData(
                name=sp.name, filepath=sp.filepath,
                load=sp.load.copy(), displacement=sp.displacement.copy(),
                time=sp.time.copy(),
                stress=sp.stress.copy() if sp.stress is not None else None,
                strain=sp.strain.copy() if sp.strain is not None else None,
                width=sp.width, thickness=sp.thickness,
                gauge_length=sp.gauge_length, cross_section=sp.cross_section,
                mode=sp.mode, composition=sp.composition, treatment=sp.treatment)
            specimens_orig.append(orig)

            diag = self.corrector.diagnose(sp)
            if diag.issues:
                self._log(f'  ⚠ {sp.name}: {"; ".join(diag.issues[:2])}')
                sp, diag = self.corrector.correct(sp, diag)
            diagnostics[sp.name] = diag
            props_dict[sp.name] = self.corrector.get_mechanical_properties(sp, diag)
        return diagnostics, props_dict, specimens_orig

    # === 生成诊断图 ===
    def _make_figs(self, name, figsize=None):
        idx = next((i for i, s in enumerate(self.specimens) if s.name == name), None)
        if idx is None:
            return [(None, None)] * 4

        sp = self.specimens[idx]
        sp_orig = self.specimens_orig[idx] if idx < len(self.specimens_orig) else sp
        diag = self.diagnostics.get(name)
        props = self.props_dict.get(name, {})
        if diag is None:
            return [(None, None)] * 4

        dpi = int(self.dpi_var.get() or 150)
        if figsize is None:
            figsize = (5, 4)
        cut_idx = getattr(diag, '_original_cut_idx', diag.cut_idx)
        es_s = getattr(diag, '_original_es_s', diag.true_elastic_start)
        es_e = getattr(diag, '_original_es_e', diag.true_elastic_end)

        load = sp_orig.load
        disp = sp_orig.displacement
        stress_o = sp_orig.stress
        strain_o = sp_orig.strain
        area = sp_orig.cross_section
        gauge = sp_orig.gauge_length

        results = []

        # ① 原始曲线
        fig1 = Figure(dpi=dpi, figsize=figsize)
        ax1 = fig1.add_subplot(111)
        ax1.plot(strain_o, stress_o, 'b-', lw=0.8)
        ax1.set_xlabel('应变 (%)'); ax1.set_ylabel('应力 (MPa)')
        ax1.autoscale(True); ax1.margins(0.05)
        if diag.issues:
            ax1.text(0.02, 0.98, '\n'.join(diag.issues[:3]), transform=ax1.transAxes,
                    fontsize=7, va='top', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8), color='red')
        fig1.tight_layout()
        results.append((fig1, (strain_o, stress_o)))

        # ② 弹性段检测
        fig2 = Figure(dpi=dpi, figsize=figsize)
        ax2 = fig2.add_subplot(111)
        ax2.plot(strain_o, stress_o, 'b-', lw=0.8, alpha=0.5)
        if es_e > es_s:
            ax2.plot(strain_o[es_s:es_e], stress_o[es_s:es_e], 'r-', lw=2.5,
                    label=f'E={props.get("E", 0):.0f} MPa')
        ax2.set_xlabel('应变 (%)'); ax2.set_ylabel('应力 (MPa)')
        ax2.autoscale(True); ax2.margins(0.05); ax2.legend(fontsize=9)
        fig2.tight_layout()
        results.append((fig2, (strain_o, stress_o)))

        # ③ 截断 + 力学性能
        fig3 = Figure(dpi=dpi, figsize=figsize)
        ax3 = fig3.add_subplot(111)
        if cut_idx > 0 and cut_idx < len(disp):
            sc = (disp[cut_idx:] - disp[cut_idx]) / gauge * 100
            stc = stress_o[cut_idx:]
            es_sn = es_s - cut_idx
            es_en = es_e - cut_idx

            ax3.plot(sc, stc, 'b-', lw=0.8, alpha=0.5)
            if 0 <= es_sn < es_en <= len(sc):
                ax3.plot(sc[es_sn:es_en], stc[es_sn:es_en], 'r-', lw=2.5, label='弹性段')
            ax3.axvline(0, color='gray', ls=':', alpha=0.5)
            ax3.set_xlim(left=-0.5)

            ui = np.argmax(stc)
            ax3.plot(sc[ui], stc[ui], 'rv', ms=8)
            ax3.annotate(f'UTS={stc[ui]:.0f}', xy=(sc[ui], stc[ui]), xytext=(10, 10),
                        textcoords='offset points', fontsize=9, color='red', fontweight='bold')

            yld = props.get('yield')
            yld_t = props.get('yield_type', '')
            if yld:
                for j in range(len(stc) - 1):
                    if (stc[j] - yld) * (stc[j + 1] - yld) <= 0:
                        ys = sc[j] + (sc[j + 1] - sc[j]) * (yld - stc[j]) / (stc[j + 1] - stc[j] + 1e-10)
                        break
                else:
                    ys = 0.2
                ax3.plot(ys, yld, 'bs', ms=8)
                ax3.annotate(f'{yld_t}={yld:.0f}', xy=(ys, yld), xytext=(10, -15),
                            textcoords='offset points', fontsize=9, color='blue', fontweight='bold')

            elong = props.get('elongation', 0)
            ax3.annotate(f'ε={elong:.1f}%', xy=(sc[-1], stc[-1]), xytext=(-60, 10),
                        textcoords='offset points', fontsize=9, color='purple', fontweight='bold')
        else:
            ax3.plot(strain_o, stress_o, 'b-', lw=0.8)
            sc, stc = strain_o, stress_o
        ax3.set_xlabel('应变 (%)'); ax3.set_ylabel('应力 (MPa)')
        ax3.autoscale(True); ax3.margins(0.05)
        fig3.tight_layout()
        results.append((fig3, (sc, stc)))

        # ④ 最终曲线
        fig4 = Figure(dpi=dpi, figsize=figsize)
        ax4 = fig4.add_subplot(111)
        if sp.stress is not None and sp.strain is not None:
            ax4.plot(sp.strain, sp.stress, 'g-', lw=1.2)
        ax4.set_xlabel('应变 (%)'); ax4.set_ylabel('应力 (MPa)')
        ax4.autoscale(True); ax4.margins(0.05)
        lines = []
        E = props.get('E', 0)
        if E > 0: lines.append(f'E = {E:.0f} MPa')
        uts = props.get('UTS', 0)
        if uts > 0: lines.append(f'UTS = {uts:.0f} MPa')
        yld = props.get('yield'); yt = props.get('yield_type', '')
        if yld: lines.append(f'{yt} = {yld:.0f} MPa')
        el = props.get('elongation', 0)
        if el > 0: lines.append(f'ε = {el:.1f}%')
        if lines:
            ax4.text(0.97, 0.03, '\n'.join(lines), transform=ax4.transAxes,
                    fontsize=9, va='bottom', ha='right',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.8))
        fig4.tight_layout()
        results.append((fig4, (sp.strain, sp.stress)))

        return results

    # === 一键处理 ===
    def _run(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning('提示', '请先选择数据路径')
            return
        self.btn_run.configure(state='disabled')
        self.log.delete('1.0', 'end')
        self.status_var.set('处理中...')

        def worker():
            try:
                self._log('🔧 加载数据...')
                self.specimens = self._load(path)
                if not self.specimens:
                    self._log('❌ 无数据'); return
                self._log(f'\n✅ {len(self.specimens)} 个试样')

                if self.fix_elastic_var.get():
                    self._log('\n🔍 弹性段修正...')
                    self.diagnostics, self.props_dict, self.specimens_orig = self._correct(self.specimens)
                    n = sum(1 for d in self.diagnostics.values() if d.corrected)
                    self._log(f'  修正 {n}/{len(self.specimens)}')

                names = [sp.name for sp in self.specimens]
                self.root.after(0, lambda: self.specimen_combo.configure(values=names))
                if names:
                    self.root.after(0, lambda: self.specimen_combo.set(names[0]))

                # 保存叠加图
                dpi = int(self.dpi_var.get() or 150)
                plotter = PlotEngine(style=self.style_var.get(), dpi=dpi)
                fig_overlay = plotter.plot_overlay(self.specimens, show_uts=self.show_uts_var.get())
                out_dir = self._get_export_dir()
                overlay_path = os.path.join(out_dir, '拉伸应力应变曲线.png')
                fig_overlay.savefig(overlay_path, bbox_inches='tight', facecolor='white', dpi=dpi)
                plt.close(fig_overlay)
                self._log(f'\n💾 {overlay_path}')

                # 四张诊断图
                if names:
                    self._log(f'\n🎨 诊断图: {names[0]}')
                    fs = self._get_panel_figsize(0)
                    figs_data = self._make_figs(names[0], figsize=fs)
                    for i, (fig, data) in enumerate(figs_data):
                        if fig:
                            self.root.after(0, lambda f=fig, d=data, p=self.panels[i]: p.show(f, d))

                self._log('\n✅ 完成!')
                self.status_var.set(f'完成 — {len(self.specimens)} 个试样')
            except Exception as e:
                import traceback
                self._log(f'\n❌ {e}\n{traceback.format_exc()}')
                self.status_var.set('出错')
            finally:
                self.root.after(0, lambda: self.btn_run.configure(state='normal'))

        threading.Thread(target=worker, daemon=True).start()

    # === 作图/试样切换 ===
    def _refresh_panel(self, idx):
        name = self.specimen_combo.get()
        if not name or not self.specimens:
            self.root.after(0, lambda: messagebox.showinfo('提示', '请先加载数据'))
            return
        try:
            self._log(f'🎨 {name} → 面板{idx+1}')
            figsize = self._get_panel_figsize(idx)
            figs_data = self._make_figs(name, figsize=figsize)
            fig, data = figs_data[idx]
            if fig:
                self.root.after(0, lambda: self.panels[idx].show(fig, data))
        except Exception as e:
            self._log(f'❌ {e}')

    def _show_diagnostic(self):
        name = self.specimen_combo.get()
        if not name or not self.specimens:
            return
        self._log(f'🎨 {name}')

        def worker():
            try:
                # 用第一个面板的尺寸作为基准
                figsize = self._get_panel_figsize(0)
                figs_data = self._make_figs(name, figsize=figsize)
                for i, (fig, data) in enumerate(figs_data):
                    if fig:
                        self.root.after(0, lambda f=fig, d=data, p=self.panels[i]: p.show(f, d))
            except Exception as e:
                self._log(f'❌ {e}')

        threading.Thread(target=worker, daemon=True).start()

    # === 批量处理 ===
    def _batch(self):
        path = self.path_var.get().strip()
        if not path or not os.path.isdir(path):
            messagebox.showwarning('提示', '请选择目录')
            return

        files = find_all_data_files(path)
        names = [Path(f).name for f in files]
        self.found_label.configure(text=f'找到 {len(files)} 个:\n' + '\n'.join(names[:10]) +
                                    (f'\n...共{len(files)}个' if len(files) > 10 else ''))
        self._log(f'\n📂 {len(files)} 个文件')

        if not files:
            return

        self.btn_batch.configure(state='disabled')
        self.status_var.set('批量处理中...')

        def worker():
            try:
                w = float(self.width_var.get() or 0)
                t = float(self.thickness_var.get() or 0)
                g = float(self.gauge_var.get() or 0)

                specimen_info = {}
                for c in ['试样信息.csv', 'specimen_info.csv']:
                    fp = os.path.join(path, c)
                    if os.path.exists(fp):
                        specimen_info = DataParser.parse_specimen_info_csv(fp)
                        break

                processor = BatchProcessor(mode=self.mode_var.get(), width=w, thickness=t,
                                           gauge_length=g, specimen_params=specimen_info)
                all_sp = []
                for f in files:
                    name = Path(f).stem
                    try:
                        sp = processor.load_specimen(f, name)
                        sp = DataProcessor.convert_to_stress_strain(sp)
                        sp = DataProcessor.normalize_strain_start(sp)
                        if self.fix_elastic_var.get():
                            diag = self.corrector.diagnose(sp)
                            if diag.issues:
                                sp, diag = self.corrector.correct(sp, diag)
                        all_sp.append(sp)
                        self._log(f'  ✅ {name}')
                    except Exception as e:
                        self._log(f'  ❌ {name}: {e}')

                if all_sp:
                    self.specimens = all_sp
                    dpi = int(self.dpi_var.get() or 150)
                    plotter = PlotEngine(style=self.style_var.get(), dpi=dpi)
                    fig = plotter.plot_overlay(all_sp, show_uts=self.show_uts_var.get())
                    out = os.path.join(self._get_export_dir(), '批量_叠加图.png')
                    fig.savefig(out, bbox_inches='tight', facecolor='white', dpi=dpi)
                    plt.close(fig)
                    self._log(f'\n💾 {out}')

                self._log(f'\n✅ 批量完成: {len(all_sp)}个')
                self.status_var.set(f'批量完成 — {len(all_sp)}个')
            except Exception as e:
                self._log(f'\n❌ {e}')
                self.status_var.set('出错')
            finally:
                self.root.after(0, lambda: self.btn_batch.configure(state='normal'))

        threading.Thread(target=worker, daemon=True).start()


def main():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    MechPlotGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
