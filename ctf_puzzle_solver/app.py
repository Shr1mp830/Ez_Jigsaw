from __future__ import annotations

import argparse
import gc
import queue
import shutil
import traceback
import threading
import tkinter as tk
from pathlib import Path
from statistics import median
from time import strftime
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .image_tools import lsb_bitplane, split_scrambled_image
from .solver import (
    Piece,
    SolveResult,
    compose_image,
    load_pieces,
    open_image,
    scatter_pieces,
    solve_by_edges,
    solve_with_reference,
    with_opacity,
)


class PuzzleApp:
    EDGE_SOLVE_WARNING_LIMIT = 260
    MIN_CANVAS_ZOOM = 0.08
    CANVAS_PIXEL_BUDGET = 6_500_000
    RENDER_BATCH_SIZE = 24

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.version = "v1.0"
        self.root.title(f"Ez_Jigsaw {self.version}")
        self.root.geometry("1440x880")
        self.root.minsize(1040, 680)
        self.colors = {
            "app": "#090d14",
            "panel": "#101722",
            "panel_2": "#182233",
            "panel_3": "#202b3d",
            "canvas": "#05080d",
            "canvas_grid": "#111827",
            "canvas_major": "#243247",
            "text": "#edf2f7",
            "muted": "#94a3b8",
            "subtle": "#64748b",
            "accent": "#18b6d4",
            "accent_hover": "#38d5f2",
            "success": "#22c55e",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "border": "#2b384c",
        }

        self.pieces: list[Piece] = []
        self.original = None
        self.original_path: Path | None = None
        self.pieces_folder: Path | None = None
        self.scrambled_image_path: Path | None = None
        self.result: SolveResult | None = None

        self.zoom = tk.DoubleVar(value=1.0)
        self.background_opacity = tk.DoubleVar(value=0.35)
        self.use_original = tk.BooleanVar(value=False)
        self.show_background = tk.BooleanVar(value=True)
        self.normalize_split_size = tk.BooleanVar(value=True)
        self.original_path_text = tk.StringVar(value="未选择原图")
        self.pieces_folder_text = tk.StringVar(value="未选择碎片文件夹")
        self.scrambled_path_text = tk.StringVar(value="未选择乱拼整图")
        self.grid_rows = tk.StringVar(value="0")
        self.grid_cols = tk.StringVar(value="0")
        self.status = tk.StringVar(value="请选择碎片文件夹；如有原图，在左侧开启并选择原图")
        self.progress_value = tk.DoubleVar(value=0.0)
        self.progress_text = tk.StringVar(value="就绪")
        self.mode_text = tk.StringVar(value="未加载")
        self.piece_count_text = tk.StringVar(value="碎片 0")
        self.grid_text = tk.StringVar(value="网格 -")
        self.selection_text = tk.StringVar(value="未选中")
        self.lsb_enabled = tk.BooleanVar(value=False)
        self.lsb_channel = tk.StringVar(value="R")
        self.lsb_bit = tk.IntVar(value=0)
        self.lsb_target_text = tk.StringVar(value="LSB 关闭")

        self.photo_refs: dict[int | str, ImageTk.PhotoImage] = {}
        self.item_to_piece: dict[int, int] = {}
        self.piece_visual_items: dict[int, tuple[int, int, int]] = {}
        self.selected_piece_id: int | None = None
        self.selected_piece_ids: set[int] = set()
        self.drag_start_world: tuple[float, float] | None = None
        self.drag_piece_start: tuple[float, float] | None = None
        self.group_drag_starts: dict[int, tuple[float, float]] = {}
        self.marquee_start_canvas: tuple[float, float] | None = None
        self.marquee_item: int | None = None
        self.drag_mode: str = "none"
        self.swap_animation_active = False
        self.selection_item: int | None = None
        self.background_item: int | None = None
        self.context_piece_id: int | None = None
        self.context_image_kind: str | None = None
        self.lsb_target_kind = "none"
        self.lsb_target_piece_id: int | None = None
        self.worker_generation = 0
        self.worker_active_generation: int | None = None
        self.worker_messages: queue.Queue[tuple[int, str, object]] = queue.Queue()
        self.render_generation = 0
        self.render_batch_after_id: str | None = None
        self.preview_generation = 0
        self.preview_messages: queue.Queue[tuple[int, str, object]] = queue.Queue()
        self.preview_image: Image.Image | None = None
        self.preview_title = ""
        self.preview_path: Path | None = None
        self.preview_pending = False

        self._build_ui()
        self._bind_events()

    def _configure_theme(self) -> None:
        self.root.configure(bg=self.colors["app"])
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=self.colors["app"], foreground=self.colors["text"], font=("Microsoft YaHei UI", 10))
        style.configure("App.TFrame", background=self.colors["app"])
        style.configure("Panel.TFrame", background=self.colors["panel"])
        style.configure("Inset.TFrame", background=self.colors["panel_2"])
        style.configure("Header.TFrame", background=self.colors["app"])
        style.configure("Status.TFrame", background=self.colors["panel"])
        style.configure("Title.TLabel", background=self.colors["app"], foreground=self.colors["text"], font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=self.colors["app"], foreground=self.colors["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("PanelTitle.TLabel", background=self.colors["panel"], foreground=self.colors["text"], font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("PanelText.TLabel", background=self.colors["panel"], foreground=self.colors["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("Tiny.TLabel", background=self.colors["panel"], foreground=self.colors["subtle"], font=("Microsoft YaHei UI", 8))
        style.configure("WorkspaceTitle.TLabel", background=self.colors["panel"], foreground=self.colors["text"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Brand.TLabel", background=self.colors["panel"], foreground=self.colors["accent_hover"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Metric.TLabel", background=self.colors["panel_2"], foreground=self.colors["text"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("MetricMuted.TLabel", background=self.colors["panel_2"], foreground=self.colors["muted"], font=("Microsoft YaHei UI", 8))
        style.configure("Status.TLabel", background=self.colors["panel"], foreground=self.colors["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("TCheckbutton", background=self.colors["panel"], foreground=self.colors["text"])
        style.map("TCheckbutton", background=[("active", self.colors["panel"])], foreground=[("active", self.colors["text"])])
        style.configure("TRadiobutton", background=self.colors["panel"], foreground=self.colors["text"])
        style.map("TRadiobutton", background=[("active", self.colors["panel"])], foreground=[("active", self.colors["text"])])
        style.configure(
            "TCombobox",
            fieldbackground=self.colors["panel_2"],
            background=self.colors["panel_2"],
            foreground=self.colors["text"],
            arrowcolor=self.colors["muted"],
            bordercolor=self.colors["border"],
        )
        style.map("TCombobox", fieldbackground=[("readonly", self.colors["panel_2"])], foreground=[("readonly", self.colors["text"])])

        style.configure(
            "Primary.TButton",
            background=self.colors["accent"],
            foreground="#ffffff",
            bordercolor=self.colors["accent"],
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
            padding=(14, 9),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.colors["accent_hover"]), ("disabled", self.colors["panel_2"])],
            foreground=[("disabled", self.colors["muted"])],
        )
        style.configure(
            "Tool.TButton",
            background=self.colors["panel_2"],
            foreground=self.colors["text"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["panel_2"],
            darkcolor=self.colors["panel_2"],
            padding=(12, 8),
        )
        style.map(
            "Tool.TButton",
            background=[("active", self.colors["panel_3"]), ("disabled", "#151a21")],
            foreground=[("disabled", self.colors["muted"])],
        )
        style.configure(
            "Danger.TButton",
            background="#30171b",
            foreground="#fecaca",
            bordercolor="#7f1d1d",
            lightcolor="#30171b",
            darkcolor="#30171b",
            padding=(12, 8),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#4a1d24"), ("disabled", "#151a21")],
            foreground=[("disabled", self.colors["muted"])],
        )
        style.configure(
            "Dark.Horizontal.TScale",
            background=self.colors["panel"],
            troughcolor=self.colors["panel_2"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
        )
        style.configure(
            "Dark.Horizontal.TProgressbar",
            background=self.colors["accent"],
            troughcolor=self.colors["panel_2"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
        )
        style.configure(
            "Vertical.TScrollbar",
            background=self.colors["panel_2"],
            troughcolor=self.colors["canvas"],
            bordercolor=self.colors["border"],
            arrowcolor=self.colors["muted"],
        )
        style.configure(
            "Horizontal.TScrollbar",
            background=self.colors["panel_2"],
            troughcolor=self.colors["canvas"],
            bordercolor=self.colors["border"],
            arrowcolor=self.colors["muted"],
        )

    def _build_ui(self) -> None:
        self._configure_theme()
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(18, 16, 18, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=f"Ez_Jigsaw {self.version}", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="矩形碎片拼图、本地自动匹配、边缘评分、手动拖拽微调", style="Subtitle.TLabel").grid(
            row=1, column=0, sticky="w", pady=(2, 0)
        )
        header_badge = tk.Frame(header, bg=self.colors["panel"], highlightthickness=1, highlightbackground=self.colors["border"])
        header_badge.grid(row=0, column=1, rowspan=2, sticky="e", padx=(16, 0))
        tk.Label(
            header_badge,
            text="By Shr1mp",
            bg=self.colors["panel"],
            fg=self.colors["accent_hover"],
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(anchor="e", padx=14, pady=(8, 0))
        tk.Label(
            header_badge,
            text="local CTF puzzle workbench",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="e", padx=14, pady=(0, 8))
        ttk.Button(header, text="问题反馈", style="Tool.TButton", command=self.show_feedback).grid(
            row=0, column=2, rowspan=2, sticky="e", padx=(10, 0)
        )

        main = ttk.Frame(self.root, style="App.TFrame", padding=(18, 0, 18, 12))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        sidebar_shell = ttk.Frame(main, style="Panel.TFrame", padding=0)
        sidebar_shell.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        sidebar_shell.columnconfigure(0, weight=1)
        sidebar_shell.rowconfigure(0, weight=1)

        sidebar_canvas = tk.Canvas(
            sidebar_shell,
            width=310,
            bg=self.colors["panel"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        sidebar_canvas.grid(row=0, column=0, sticky="ns")
        sidebar_scroll = ttk.Scrollbar(sidebar_shell, orient="vertical", command=sidebar_canvas.yview)
        sidebar_scroll.grid(row=0, column=1, sticky="ns")
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)

        sidebar = ttk.Frame(sidebar_canvas, style="Panel.TFrame", padding=16)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind("<Configure>", lambda _event: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")))
        sidebar_canvas.bind("<Configure>", lambda event: sidebar_canvas.itemconfigure(sidebar_window, width=event.width))
        sidebar_canvas.bind("<MouseWheel>", lambda event: sidebar_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units"))
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="控制台", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(sidebar, text="导入文件、拆图选项和图片操作已分组。", style="PanelText.TLabel", wraplength=250).grid(
            row=1, column=0, sticky="w", pady=(4, 14)
        )

        ttk.Label(sidebar, text="导入文件", style="PanelTitle.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 8))
        ttk.Checkbutton(sidebar, text="使用完整原图", variable=self.use_original, command=self.on_use_original_changed).grid(
            row=3, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Button(sidebar, text="选择原图", style="Tool.TButton", command=self.choose_original).grid(row=4, column=0, sticky="ew", pady=4)
        ttk.Label(sidebar, textvariable=self.original_path_text, style="PanelText.TLabel", wraplength=260).grid(
            row=5, column=0, sticky="ew", pady=(0, 8)
        )
        ttk.Button(sidebar, text="选择碎片文件夹", style="Tool.TButton", command=self.choose_pieces_folder).grid(
            row=6, column=0, sticky="ew", pady=4
        )
        ttk.Label(sidebar, textvariable=self.pieces_folder_text, style="PanelText.TLabel", wraplength=260).grid(
            row=7, column=0, sticky="ew", pady=(0, 10)
        )

        ttk.Button(sidebar, text="选择乱拼整图", style="Tool.TButton", command=self.choose_scrambled_image).grid(
            row=8, column=0, sticky="ew", pady=4
        )
        ttk.Label(sidebar, textvariable=self.scrambled_path_text, style="PanelText.TLabel", wraplength=260).grid(
            row=9, column=0, sticky="ew", pady=(0, 8)
        )
        ttk.Button(sidebar, text="加载项目", style="Tool.TButton", command=self.load_from_ui).grid(row=10, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(sidebar, text="拆图选项", style="PanelTitle.TLabel").grid(row=11, column=0, sticky="w", pady=(6, 6))
        grid_frame = ttk.Frame(sidebar, style="Panel.TFrame")
        grid_frame.grid(row=12, column=0, sticky="ew", pady=(0, 10))
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)
        ttk.Label(grid_frame, text="行数", style="PanelText.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Label(grid_frame, text="列数", style="PanelText.TLabel").grid(row=0, column=1, sticky="w", padx=(5, 0))
        self.rows_entry = tk.Entry(
            grid_frame,
            textvariable=self.grid_rows,
            bg=self.colors["panel_2"],
            fg=self.colors["text"],
            disabledbackground=self.colors["panel_2"],
            disabledforeground=self.colors["muted"],
            insertbackground=self.colors["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
            width=8,
        )
        self.rows_entry.grid(row=1, column=0, sticky="ew", padx=(0, 5), ipady=6)
        self.cols_entry = tk.Entry(
            grid_frame,
            textvariable=self.grid_cols,
            bg=self.colors["panel_2"],
            fg=self.colors["text"],
            disabledbackground=self.colors["panel_2"],
            disabledforeground=self.colors["muted"],
            insertbackground=self.colors["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
            width=8,
        )
        self.cols_entry.grid(row=1, column=1, sticky="ew", padx=(5, 0), ipady=6)
        ttk.Label(sidebar, text="行/列填 0 表示自动推断，矩形碎片建议填写真实网格。", style="PanelText.TLabel", wraplength=260).grid(
            row=13, column=0, sticky="ew", pady=(0, 10)
        )
        ttk.Checkbutton(sidebar, text="拆图统一碎片比例", variable=self.normalize_split_size).grid(
            row=14, column=0, sticky="w", pady=(0, 12)
        )

        ttk.Label(sidebar, text="图片操作", style="PanelTitle.TLabel").grid(row=15, column=0, sticky="w", pady=(6, 8))
        ttk.Button(sidebar, text="拆分整图为碎片", style="Tool.TButton", command=self.split_scrambled_to_pieces).grid(
            row=16, column=0, sticky="ew", pady=4
        )
        self.auto_button = ttk.Button(sidebar, text="开始自动拼接", style="Primary.TButton", command=self.auto_solve)
        self.auto_button.grid(row=17, column=0, sticky="ew", pady=(8, 8))
        ttk.Button(sidebar, text="散放碎片", style="Tool.TButton", command=self.reset_scatter).grid(row=18, column=0, sticky="ew", pady=4)
        ttk.Button(sidebar, text="边框对齐校准", style="Tool.TButton", command=self.calibrate_piece_borders).grid(
            row=19, column=0, sticky="ew", pady=4
        )
        ttk.Button(sidebar, text="清除所有图片", style="Danger.TButton", command=self.clear_all_images).grid(
            row=20, column=0, sticky="ew", pady=4
        )
        ttk.Button(sidebar, text="导出所有碎片到文件夹", style="Tool.TButton", command=self.export_pieces_folder).grid(
            row=21, column=0, sticky="ew", pady=4
        )
        ttk.Button(sidebar, text="导出 PNG", style="Tool.TButton", command=self.export_png).grid(row=22, column=0, sticky="ew", pady=4)

        rotate_frame = ttk.Frame(sidebar, style="Panel.TFrame")
        rotate_frame.grid(row=23, column=0, sticky="ew", pady=(10, 4))
        rotate_frame.columnconfigure(0, weight=1)
        rotate_frame.columnconfigure(1, weight=1)
        ttk.Button(rotate_frame, text="左转", style="Tool.TButton", command=lambda: self.rotate_selected(90)).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(rotate_frame, text="右转", style="Tool.TButton", command=lambda: self.rotate_selected(-90)).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        ttk.Frame(sidebar, style="Panel.TFrame", height=10).grid(row=24, column=0)
        ttk.Label(sidebar, text="视图", style="PanelTitle.TLabel").grid(row=25, column=0, sticky="w", pady=(14, 6))
        ttk.Checkbutton(sidebar, text="显示原图底层", variable=self.show_background, command=self.render).grid(
            row=26, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Label(sidebar, text="底图透明度", style="PanelText.TLabel").grid(row=27, column=0, sticky="w")
        ttk.Scale(
            sidebar,
            from_=0.05,
            to=1.0,
            variable=self.background_opacity,
            orient="horizontal",
            length=210,
            style="Dark.Horizontal.TScale",
            command=lambda _value: self.render(),
        ).grid(row=28, column=0, sticky="ew", pady=(2, 12))

        ttk.Label(sidebar, text="画布缩放", style="PanelText.TLabel").grid(row=29, column=0, sticky="w")
        ttk.Scale(
            sidebar,
            from_=self.MIN_CANVAS_ZOOM,
            to=2.0,
            variable=self.zoom,
            orient="horizontal",
            length=210,
            style="Dark.Horizontal.TScale",
            command=lambda _value: self.render(),
        ).grid(row=30, column=0, sticky="ew", pady=(2, 14))

        ttk.Label(sidebar, text="LSB 隐写", style="PanelTitle.TLabel").grid(row=31, column=0, sticky="w", pady=(8, 6))
        ttk.Label(sidebar, textvariable=self.lsb_target_text, style="PanelText.TLabel", wraplength=260).grid(
            row=32, column=0, sticky="ew", pady=(0, 6)
        )
        lsb_frame = ttk.Frame(sidebar, style="Panel.TFrame")
        lsb_frame.grid(row=33, column=0, sticky="ew", pady=(0, 8))
        lsb_frame.columnconfigure(0, weight=1)
        lsb_frame.columnconfigure(1, weight=1)
        ttk.Label(lsb_frame, text="色道", style="PanelText.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Label(lsb_frame, text="bit", style="PanelText.TLabel").grid(row=0, column=1, sticky="w", padx=(5, 0))
        self.lsb_channel_box = ttk.Combobox(
            lsb_frame,
            textvariable=self.lsb_channel,
            values=("R", "G", "B", "A", "RGB"),
            state="readonly",
            width=8,
        )
        self.lsb_channel_box.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        self.lsb_channel_box.bind("<<ComboboxSelected>>", lambda _event: self.on_lsb_control_changed())
        bit_box = ttk.Frame(lsb_frame, style="Panel.TFrame")
        bit_box.grid(row=1, column=1, sticky="ew", padx=(5, 0))
        for bit in range(8):
            ttk.Radiobutton(
                bit_box,
                text=str(bit),
                value=bit,
                variable=self.lsb_bit,
                command=self.on_lsb_control_changed,
            ).grid(row=bit // 4, column=bit % 4, sticky="w")

        lsb_buttons = ttk.Frame(sidebar, style="Panel.TFrame")
        lsb_buttons.grid(row=34, column=0, sticky="ew", pady=(0, 12))
        lsb_buttons.columnconfigure(0, weight=1)
        lsb_buttons.columnconfigure(1, weight=1)
        ttk.Button(lsb_buttons, text="应用选中", style="Tool.TButton", command=self.apply_lsb_to_selected).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(lsb_buttons, text="应用全部", style="Tool.TButton", command=self.apply_lsb_to_all).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        ttk.Button(sidebar, text="关闭 LSB 显示", style="Tool.TButton", command=self.disable_lsb_display).grid(
            row=35, column=0, sticky="ew", pady=(0, 10)
        )

        ttk.Label(sidebar, text="状态", style="PanelTitle.TLabel").grid(row=36, column=0, sticky="w", pady=(8, 6))
        metrics = ttk.Frame(sidebar, style="Panel.TFrame")
        metrics.grid(row=37, column=0, sticky="ew")
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        self._metric(metrics, "模式", self.mode_text, 0, 0)
        self._metric(metrics, "数量", self.piece_count_text, 0, 1)
        self._metric(metrics, "网格", self.grid_text, 1, 0)
        self._metric(metrics, "选中", self.selection_text, 1, 1)

        ttk.Label(sidebar, textvariable=self.status, style="PanelText.TLabel", wraplength=260).grid(
            row=38, column=0, sticky="ew", pady=(14, 0)
        )
        ttk.Progressbar(
            sidebar,
            variable=self.progress_value,
            maximum=100,
            mode="determinate",
            style="Dark.Horizontal.TProgressbar",
        ).grid(row=39, column=0, sticky="ew", pady=(10, 4))
        ttk.Label(sidebar, textvariable=self.progress_text, style="PanelText.TLabel", wraplength=260).grid(
            row=40, column=0, sticky="ew", pady=(0, 0)
        )

        workspace = ttk.Frame(main, style="Panel.TFrame", padding=1)
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(1, weight=1)

        workspace_header = ttk.Frame(workspace, style="Panel.TFrame", padding=(12, 8))
        workspace_header.grid(row=0, column=0, sticky="ew")
        workspace_header.columnconfigure(0, weight=1)
        ttk.Label(workspace_header, text="Workspace", style="WorkspaceTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(workspace_header, textvariable=self.progress_text, style="PanelText.TLabel").grid(row=0, column=1, sticky="e", padx=(12, 0))

        canvas_frame = ttk.Frame(workspace, style="Panel.TFrame")
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg=self.colors["canvas"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
            insertbackground=self.colors["text"],
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        horizontal = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)

        footer = ttk.Frame(self.root, style="Status.TFrame", padding=(18, 8))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(
            footer,
            text="提示：按住 Ctrl 滚轮缩放；右键图片查看 LSB 通道；选中碎片后可按 R / L 旋转。",
            style="Status.TLabel",
        ).grid(row=0, column=0, sticky="w")

    def _metric(self, parent: ttk.Frame, label: str, value: tk.StringVar, row: int, col: int) -> None:
        box = ttk.Frame(parent, style="Panel.TFrame")
        box.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 6, 6 if col == 0 else 0), pady=4)
        inner = tk.Frame(box, bg=self.colors["panel_2"], highlightthickness=1, highlightbackground=self.colors["border"])
        inner.pack(fill="x")
        tk.Label(inner, text=label, bg=self.colors["panel_2"], fg=self.colors["muted"], font=("Microsoft YaHei UI", 8)).pack(
            anchor="w", padx=10, pady=(8, 0)
        )
        tk.Label(inner, textvariable=value, bg=self.colors["panel_2"], fg=self.colors["text"], font=("Microsoft YaHei UI", 10, "bold")).pack(
            anchor="w", padx=10, pady=(0, 8)
        )

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<Button-3>", self.on_context_menu)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.root.bind("<Key-r>", lambda _event: self.rotate_selected(-90))
        self.root.bind("<Key-l>", lambda _event: self.rotate_selected(90))
        self.root.bind("<Control-plus>", lambda _event: self.bump_zoom(0.1))
        self.root.bind("<Control-minus>", lambda _event: self.bump_zoom(-0.1))

    def ask_and_load(self) -> None:
        self.load_from_ui()

    def background_busy(self, action: str) -> bool:
        if self.worker_active_generation is None:
            return False
        self.status.set(f"后台任务进行中，请稍后再{action}")
        return True

    def set_progress(self, value: float, message: str | None = None) -> None:
        value = max(0.0, min(100.0, value))
        self.progress_value.set(value)
        if message is not None:
            self.progress_text.set(f"{message} {value:.0f}%")

    def finish_progress(self, message: str = "完成") -> None:
        self.set_progress(100.0, message)

    def show_feedback(self) -> None:
        messagebox.showinfo("问题反馈", "欢迎发送到 q13881305006@163.com")

    def log_background_error(self, label: str, exc: Exception) -> None:
        try:
            log_path = Path(__file__).resolve().parents[1] / "ez_jigsaw_error.log"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{strftime('%Y-%m-%d %H:%M:%S')}] {label}\n")
                handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        except Exception:
            pass

    def choose_original(self) -> None:
        path = filedialog.askopenfilename(
            title="选择完整原图",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.original_path = Path(path)
        self.original_path_text.set(self.compact_path(self.original_path))
        self.use_original.set(True)
        self.status.set("已选择原图，请继续选择碎片文件夹或加载项目")
        self.start_image_preview(self.original_path, "原图预览")

    def choose_pieces_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择拼图碎片文件夹")
        if not folder:
            return
        self.pieces_folder = Path(folder)
        self.pieces_folder_text.set(self.compact_path(self.pieces_folder))
        self.status.set("已选择碎片文件夹，点击加载项目")

    def choose_scrambled_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择乱拼整图",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.scrambled_image_path = Path(path)
        self.scrambled_path_text.set(self.compact_path(self.scrambled_image_path))
        self.status.set("已选择乱拼整图；可填写行列后点击拆分")
        self.start_image_preview(self.scrambled_image_path, "乱拼整图预览")

    def split_scrambled_to_pieces(self) -> None:
        if self.background_busy("拆分"):
            return
        if self.scrambled_image_path is None:
            self.choose_scrambled_image()
        if self.scrambled_image_path is None:
            self.status.set("请先选择乱拼整图")
            return

        try:
            grid = self.selected_grid()
            rows, cols = grid if grid is not None else (0, 0)
        except ValueError as exc:
            self.status.set(str(exc))
            return

        output_root = Path(__file__).resolve().parents[1] / "split_outputs"
        output_folder = output_root / f"{self.scrambled_image_path.stem}_{strftime('%Y%m%d_%H%M%S')}"
        image_path = self.scrambled_image_path
        normalize_piece_size = bool(self.normalize_split_size.get())
        self.worker_generation += 1
        generation = self.worker_generation
        self.worker_active_generation = generation
        self.status.set("正在检测边缘并拆分碎片...")
        self.set_progress(5, "正在拆分整图")

        def worker() -> None:
            try:
                self.worker_messages.put((generation, "progress", (15, "正在检测切分网格")))
                result = split_scrambled_image(
                    image_path,
                    output_folder,
                    rows=rows,
                    cols=cols,
                    normalize_piece_size=normalize_piece_size,
                )
                self.worker_messages.put((generation, "progress", (95, "碎片写入完成")))
                self.worker_messages.put((generation, "split_done", result))
            except Exception as exc:
                self.log_background_error("split_scrambled_to_pieces", exc)
                self.worker_messages.put((generation, "split_error", exc))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(80, self._poll_worker)

    def on_use_original_changed(self) -> None:
        if not self.use_original.get():
            self.original = None
            self.mode_text.set("边缘评分" if self.pieces else "未加载")
            self.status.set("已切换到无原图模式")
            self.render()

    def load_from_ui(self) -> None:
        if self.pieces_folder is None:
            self.choose_pieces_folder()
        if self.pieces_folder is None:
            self.status.set("请先选择碎片文件夹")
            return

        original_path = self.original_path if self.use_original.get() else None
        if self.use_original.get() and original_path is None:
            self.status.set("已开启使用原图，请先选择完整原图")
            return

        self.start_load_project(self.pieces_folder, original_path, auto_solve=True)

    def load_project(self, pieces_folder: Path, original_path: Path | None = None) -> None:
        original = open_image(original_path) if original_path else None
        pieces = load_pieces(pieces_folder)
        self.apply_loaded_project(pieces_folder, original_path, original, pieces)

    def start_image_preview(self, image_path: Path, title: str) -> None:
        image_path = Path(image_path)
        self.preview_generation += 1
        generation = self.preview_generation
        self.preview_image = None
        self.preview_title = title
        self.preview_path = image_path
        self.preview_pending = True
        self.render()
        self.status.set(f"正在打开预览：{image_path.name}")

        def worker() -> None:
            try:
                image = open_image(image_path)
                resampling = getattr(Image, "Resampling", Image)
                image.thumbnail((1800, 1200), resampling.LANCZOS)
                self.preview_messages.put((generation, "preview_done", (image_path, title, image.copy())))
            except Exception as exc:
                self.log_background_error("preview", exc)
                self.preview_messages.put((generation, "preview_error", exc))
            finally:
                try:
                    image.close()
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self._poll_preview)

    def _poll_preview(self) -> None:
        handled = False
        while True:
            try:
                generation, kind, payload = self.preview_messages.get_nowait()
            except queue.Empty:
                break

            if generation != self.preview_generation:
                continue
            handled = True
            if kind == "preview_done":
                image_path, title, image = payload
                self.preview_pending = False
                self.preview_path = image_path
                self.preview_title = title
                self.preview_image = image
                self.status.set(f"已预览：{image_path.name}")
                self.render()
            elif kind == "preview_error":
                self.preview_pending = False
                self.preview_image = None
                self.status.set(f"预览失败：{payload}")
                self.render()

        if self.preview_pending or self.preview_messages.qsize():
            self.root.after(80, self._poll_preview)

    def start_load_project(self, pieces_folder: Path, original_path: Path | None = None, auto_solve: bool = False) -> None:
        if self.background_busy("加载项目"):
            return
        pieces_folder = Path(pieces_folder)
        original_path = Path(original_path) if original_path else None
        self.pieces_folder = pieces_folder
        self.original_path = original_path
        self.pieces_folder_text.set(self.compact_path(pieces_folder))
        if original_path is not None:
            self.original_path_text.set(self.compact_path(original_path))
            self.use_original.set(True)
        elif not self.use_original.get():
            self.original_path_text.set("未选择原图")

        self.worker_generation += 1
        generation = self.worker_generation
        self.worker_active_generation = generation
        self.preview_generation += 1
        self.preview_image = None
        self.preview_title = "正在读取碎片"
        self.preview_path = pieces_folder
        self.preview_pending = False
        self.render()
        self.auto_button.configure(state="disabled")
        self.result = None
        self.selected_piece_id = None
        self.selected_piece_ids.clear()
        self.mode_text.set("读取中")
        self.piece_count_text.set("碎片 ...")
        self.grid_text.set("等待读取")
        self.status.set("正在后台读取图片碎片...")
        self.set_progress(5, "正在读取图片")

        def worker() -> None:
            try:
                self.worker_messages.put((generation, "progress", (20, "正在读取原图")))
                original = open_image(original_path) if original_path else None
                self.worker_messages.put((generation, "progress", (45, "正在读取碎片")))
                pieces = load_pieces(pieces_folder)
                self.worker_messages.put((generation, "progress", (90, "碎片读取完成")))
                payload = (pieces_folder, original_path, original, pieces, auto_solve)
                self.worker_messages.put((generation, "load_done", payload))
            except Exception as exc:
                self.log_background_error("load_project", exc)
                self.worker_messages.put((generation, "load_error", exc))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(80, self._poll_worker)

    def apply_loaded_project(
        self,
        pieces_folder: Path,
        original_path: Path | None,
        original: Image.Image | None,
        pieces: list[Piece],
    ) -> bool:
        self.pieces_folder = pieces_folder
        self.original_path = original_path
        self.original = original
        self.pieces = pieces
        self.preview_image = None
        self.preview_title = ""
        self.preview_path = None
        self.preview_pending = False
        self.result = None
        self.selected_piece_id = None
        self.selected_piece_ids.clear()

        if not self.pieces:
            messagebox.showerror("没有碎片", "所选文件夹中没有可识别的图片碎片。")
            self.status.set("没有读取到图片碎片")
            self.mode_text.set("未加载")
            self.piece_count_text.set("碎片 0")
            self.grid_text.set("网格 -")
            self.selection_text.set("未选中")
            self.auto_button.configure(state="normal")
            self.render()
            return False

        start_x = self.original.width + 32 if self.original is not None else 0
        scatter_pieces(self.pieces, start_x=start_x, start_y=0)
        self.fit_zoom_to_project()
        self.pieces_folder_text.set(self.compact_path(pieces_folder))
        if original_path is not None:
            self.original_path_text.set(self.compact_path(original_path))
            self.use_original.set(True)
        elif not self.use_original.get():
            self.original_path_text.set("未选择原图")
        self.status.set(f"已读取 {len(self.pieces)} 个碎片")
        self.mode_text.set("原图匹配" if self.original is not None else "边缘评分")
        self.piece_count_text.set(f"碎片 {len(self.pieces)}")
        self.grid_text.set("等待拼接")
        self.selection_text.set("未选中")
        self.auto_button.configure(state="normal")
        self.render()
        return True

    def fit_zoom_to_project(self) -> None:
        if not self.pieces and self.original is None:
            return
        max_width = self.original.width if self.original is not None else 1
        max_height = self.original.height if self.original is not None else 1
        for piece in self.pieces:
            max_width = max(max_width, int(piece.x + piece.width))
            max_height = max(max_height, int(piece.y + piece.height))
        canvas_width = max(640, self.canvas.winfo_width() or 960)
        canvas_height = max(420, self.canvas.winfo_height() or 640)
        target_zoom = min(1.0, canvas_width / max(1, max_width + 120), canvas_height / max(1, max_height + 120))
        self.zoom.set(max(self.MIN_CANVAS_ZOOM, min(1.0, target_zoom)))

    def auto_solve(self) -> None:
        if self.background_busy("拼接"):
            return
        if not self.pieces:
            self.status.set("请先选择碎片文件夹")
            return
        try:
            grid = self.selected_grid()
        except ValueError as exc:
            self.status.set(str(exc))
            return

        if self.original is None and len(self.pieces) > self.EDGE_SOLVE_WARNING_LIMIT:
            self.status.set(f"碎片 {len(self.pieces)} 个较多，边缘拼接容易卡死；建议使用原图或分批处理")
            messagebox.showwarning(
                "碎片过多",
                "无原图边缘拼接会对碎片两两打分，碎片数量过多时容易卡死或崩溃。\n"
                "建议使用完整原图、减少碎片数量，或分批处理后再拼接。",
            )
            return

        pieces = self.clone_pieces_for_worker(self.pieces)
        original = self.original
        self.worker_generation += 1
        generation = self.worker_generation
        self.worker_active_generation = generation
        self.auto_button.configure(state="disabled")
        if self.original is not None:
            self.status.set("正在按原图匹配拼接...")
        else:
            self.status.set("正在按边缘像素差异拼接...")
        self.set_progress(2, "正在启动拼接")

        def progress(value: float, message: str) -> None:
            self.worker_messages.put((generation, "progress", (value, message)))

        def worker() -> None:
            try:
                if original is not None:
                    result = solve_with_reference(pieces, original, grid=grid, progress_callback=progress)
                else:
                    result = solve_by_edges(pieces, grid=grid, progress_callback=progress)
                self.worker_messages.put((generation, "done", (result, pieces)))
            except Exception as exc:
                self.log_background_error("auto_solve", exc)
                self.worker_messages.put((generation, "error", exc))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(80, self._poll_worker)

    @staticmethod
    def clone_pieces_for_worker(pieces: list[Piece]) -> list[Piece]:
        return [
            Piece(piece.id, piece.name, piece.path, piece.image.copy(), piece.x, piece.y, piece.score, piece.matched_cell)
            for piece in pieces
        ]

    def _poll_worker(self) -> None:
        while True:
            try:
                generation, kind, payload = self.worker_messages.get_nowait()
            except queue.Empty:
                if self.worker_active_generation is None:
                    return
                self.root.after(80, self._poll_worker)
                return
            if generation != self.worker_generation:
                continue
            if kind == "progress":
                value, message = payload  # type: ignore[misc]
                self.set_progress(float(value), str(message))
                continue
            if generation == self.worker_active_generation:
                break
        self.worker_active_generation = None

        if kind == "split_error":
            messagebox.showerror("拆分失败", str(payload))
            self.status.set("乱拼整图拆分失败")
            self.set_progress(0, "拆分失败")
            return

        if kind == "split_done":
            result = payload
            self.finish_progress("拆分完成")
            self.pieces_folder = result.folder  # type: ignore[attr-defined]
            self.pieces_folder_text.set(self.compact_path(self.pieces_folder))
            self.grid_rows.set(str(result.rows))  # type: ignore[attr-defined]
            self.grid_cols.set(str(result.cols))  # type: ignore[attr-defined]
            self.piece_count_text.set(f"碎片 {result.count}")  # type: ignore[attr-defined]
            self.grid_text.set(f"{result.rows} x {result.cols}")  # type: ignore[attr-defined]
            self.status.set(f"已拆分 {result.count} 个碎片，请点击加载项目或导出碎片")  # type: ignore[attr-defined]
            return

        self.auto_button.configure(state="normal")
        if kind == "load_error":
            messagebox.showerror("导入失败", str(payload))
            self.status.set("图片导入失败")
            self.set_progress(0, "导入失败")
            return

        if kind == "load_done":
            self.finish_progress("导入完成")
            pieces_folder, original_path, original, pieces, should_auto_solve = payload  # type: ignore[misc]
            loaded = self.apply_loaded_project(pieces_folder, original_path, original, pieces)
            if loaded and should_auto_solve:
                self.root.after(150, self.auto_solve)
            return

        if kind == "export_error":
            messagebox.showerror("导出失败", str(payload))
            self.status.set("PNG 导出失败")
            self.set_progress(0, "导出失败")
            return

        if kind == "export_done":
            self.status.set(f"已导出：{payload}")
            self.finish_progress("导出完成")
            return

        if kind == "pieces_export_error":
            messagebox.showerror("导出碎片失败", str(payload))
            self.status.set("碎片导出失败")
            self.set_progress(0, "导出失败")
            return

        if kind == "pieces_export_done":
            self.status.set(f"碎片已导出到：{payload}")
            self.finish_progress("碎片导出完成")
            return

        if kind == "error":
            messagebox.showerror("拼接失败", str(payload))
            self.status.set("自动拼接失败")
            self.set_progress(0, "拼接失败")
            return

        self.result, self.pieces = payload  # type: ignore[misc]
        score_text = ""
        if self.result and self.result.average_score is not None:
            if self.result.mode == "edges":
                quality = self.edge_quality_score(self.result.average_score)
                score_text = f"，质量分 {quality:.1f}/100"
            else:
                score_text = f"，匹配误差 {self.result.average_score:.1f}"
        if self.result:
            self.status.set(f"拼接完成：{self.result.rows} x {self.result.cols}{score_text}")
            self.grid_text.set(f"{self.result.rows} x {self.result.cols}")
            self.mode_text.set("原图匹配" if self.result.mode == "reference" else "边缘评分")
        self.finish_progress("拼接完成")
        self.render()

    @staticmethod
    def edge_quality_score(edge_cost: float) -> float:
        return max(0.0, min(100.0, 100.0 / (1.0 + edge_cost / 2500.0)))

    def reset_scatter(self) -> None:
        if not self.pieces:
            return
        start_x = self.original.width + 32 if self.original is not None else 0
        scatter_pieces(self.pieces, start_x=start_x, start_y=0)
        self.result = None
        self.status.set("碎片已重新散放")
        self.grid_text.set("手动整理")
        self.render()

    def clear_all_images(self) -> None:
        self.cancel_pending_render()
        self.worker_generation += 1
        self.preview_generation += 1
        self.worker_active_generation = None
        self.pieces = []
        self.original = None
        self.original_path = None
        self.pieces_folder = None
        self.scrambled_image_path = None
        self.result = None
        self.preview_image = None
        self.preview_title = ""
        self.preview_path = None
        self.preview_pending = False
        self.selected_piece_id = None
        self.selected_piece_ids.clear()
        self.drag_start_world = None
        self.drag_piece_start = None
        self.group_drag_starts.clear()
        self.marquee_start_canvas = None
        self.drag_mode = "none"
        if self.marquee_item is not None:
            self.canvas.delete(self.marquee_item)
            self.marquee_item = None
        self.swap_animation_active = False
        self.context_piece_id = None
        self.context_image_kind = None
        self.disable_lsb_display(render=False)

        self.use_original.set(False)
        self.show_background.set(True)
        self.normalize_split_size.set(True)
        self.background_opacity.set(0.35)
        self.original_path_text.set("未选择原图")
        self.pieces_folder_text.set("未选择碎片文件夹")
        self.scrambled_path_text.set("未选择乱拼整图")
        self.mode_text.set("未加载")
        self.piece_count_text.set("碎片 0")
        self.grid_text.set("网格 -")
        self.selection_text.set("未选中")
        self.status.set("已清除所有图片")
        self.set_progress(0, "就绪")
        self.auto_button.configure(state="normal")
        gc.collect()
        self.render()

    def calibrate_piece_borders(self) -> None:
        if not self.pieces:
            self.status.set("没有可校准的碎片")
            return

        matched = [piece for piece in self.pieces if piece.matched_cell is not None]
        if matched:
            row_count = max(piece.matched_cell[0] for piece in matched if piece.matched_cell is not None) + 1
            col_count = max(piece.matched_cell[1] for piece in matched if piece.matched_cell is not None) + 1
            column_widths = [self.median_size_for_axis(matched, "width", col) for col in range(col_count)]
            row_heights = [self.median_size_for_axis(matched, "height", row) for row in range(row_count)]
            column_offsets = self.cumulative_offsets(column_widths)
            row_offsets = self.cumulative_offsets(row_heights)
            for piece in matched:
                row, col = piece.matched_cell or (0, 0)
                piece.x = column_offsets[col]
                piece.y = row_offsets[row]
            self.status.set("已按自动拼接网格校准矩形边框")
            self.grid_text.set(f"{row_count} x {col_count}")
            self.render()
            return

        cell_width = max(1.0, float(median(piece.width for piece in self.pieces)))
        cell_height = max(1.0, float(median(piece.height for piece in self.pieces)))
        origin_x = min(piece.x for piece in self.pieces)
        origin_y = min(piece.y for piece in self.pieces)
        for piece in self.pieces:
            piece.x = origin_x + round((piece.x - origin_x) / cell_width) * cell_width
            piece.y = origin_y + round((piece.y - origin_y) / cell_height) * cell_height
        self.status.set("已按当前布局吸附校准碎片边框")
        self.grid_text.set("手动校准")
        self.render()

    def median_size_for_axis(self, pieces: list[Piece], axis: str, index: int) -> float:
        if axis == "width":
            values = [piece.width for piece in pieces if piece.matched_cell is not None and piece.matched_cell[1] == index]
            fallback = [piece.width for piece in pieces]
        elif axis == "height":
            values = [piece.height for piece in pieces if piece.matched_cell is not None and piece.matched_cell[0] == index]
            fallback = [piece.height for piece in pieces]
        else:
            raise ValueError(f"未知校准轴：{axis}")
        return max(1.0, float(median(values or fallback)))

    @staticmethod
    def cumulative_offsets(sizes: list[float]) -> list[float]:
        offsets: list[float] = []
        current = 0.0
        for size in sizes:
            offsets.append(current)
            current += size
        return offsets

    def export_png(self) -> None:
        if self.background_busy("导出"):
            return
        if not self.pieces:
            self.status.set("没有可导出的碎片")
            return

        path = filedialog.asksaveasfilename(
            title="导出拼接结果",
            defaultextension=".png",
            initialfile="puzzle_result.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return

        pieces = self.clone_pieces_for_worker(self.pieces)
        original = self.original
        include_background = bool(original is not None and self.show_background.get())
        background_opacity = float(self.background_opacity.get())
        self.worker_generation += 1
        generation = self.worker_generation
        self.worker_active_generation = generation
        self.status.set("正在后台导出 PNG...")
        self.set_progress(10, "正在导出 PNG")

        def worker() -> None:
            try:
                self.worker_messages.put((generation, "progress", (35, "正在合成导出图片")))
                image = compose_image(
                    pieces,
                    original=original,
                    include_background=include_background,
                    background_opacity=background_opacity,
                )
                self.worker_messages.put((generation, "progress", (80, "正在保存 PNG")))
                image.save(path)
                self.worker_messages.put((generation, "export_done", path))
            except Exception as exc:
                self.log_background_error("export_png", exc)
                self.worker_messages.put((generation, "export_error", exc))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(80, self._poll_worker)

    def export_pieces_folder(self) -> None:
        if self.background_busy("导出碎片"):
            return

        output_folder = filedialog.askdirectory(title="选择导出碎片图片的文件夹")
        if not output_folder:
            return

        destination = Path(output_folder)
        if self.pieces:
            pieces = self.clone_pieces_for_worker(self.pieces)
            self.worker_generation += 1
            generation = self.worker_generation
            self.worker_active_generation = generation
            self.status.set("正在导出当前工作区碎片...")
            self.set_progress(10, "正在导出碎片")

            def worker() -> None:
                try:
                    total = max(1, len(pieces))
                    for index, piece in enumerate(pieces):
                        name = piece.name if piece.name.lower().endswith(".png") else f"{Path(piece.name).stem}.png"
                        piece.image.save(destination / name)
                        if index % 8 == 0:
                            self.worker_messages.put((generation, "progress", (10 + index / total * 80, "正在保存碎片")))
                    self.worker_messages.put((generation, "pieces_export_done", destination))
                except Exception as exc:
                    self.log_background_error("export_pieces_folder", exc)
                    self.worker_messages.put((generation, "pieces_export_error", exc))

            threading.Thread(target=worker, daemon=True).start()
            self.root.after(80, self._poll_worker)
            return

        if self.pieces_folder is None or not self.pieces_folder.exists():
            self.status.set("没有可导出的碎片，请先拆分整图或加载项目")
            return

        self.worker_generation += 1
        generation = self.worker_generation
        self.worker_active_generation = generation
        source = self.pieces_folder
        self.status.set("正在复制拆分输出碎片...")
        self.set_progress(10, "正在导出碎片")

        def worker() -> None:
            try:
                files = [
                    path
                    for path in sorted(source.iterdir(), key=lambda item: item.name.lower())
                    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
                ]
                total = max(1, len(files))
                for index, path in enumerate(files):
                    shutil.copy2(path, destination / path.name)
                    if index % 8 == 0:
                        self.worker_messages.put((generation, "progress", (10 + index / total * 80, "正在复制碎片")))
                self.worker_messages.put((generation, "pieces_export_done", destination))
            except Exception as exc:
                self.log_background_error("export_pieces_folder", exc)
                self.worker_messages.put((generation, "pieces_export_error", exc))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(80, self._poll_worker)

    def rotate_selected(self, degrees: int) -> None:
        piece = self.selected_piece()
        if piece is None:
            self.status.set("请先选中一个碎片")
            return

        piece.image = piece.image.rotate(degrees, expand=True)
        piece.score = None
        self.status.set(f"已旋转：{piece.name}")
        self.selection_text.set(self.compact_name(piece.name))
        self.render()

    def selected_piece(self) -> Piece | None:
        if self.selected_piece_id is None:
            return None
        return next((piece for piece in self.pieces if piece.id == self.selected_piece_id), None)

    def display_image_for_canvas(self, image: Image.Image, kind: str, piece_id: int | None = None) -> Image.Image:
        if not self.lsb_enabled.get() or not self.lsb_matches(kind, piece_id):
            return image
        try:
            return lsb_bitplane(image, self.lsb_channel.get(), int(self.lsb_bit.get()))
        except Exception as exc:
            self.disable_lsb_display(render=False)
            self.status.set(str(exc))
            return image

    def lsb_matches(self, kind: str, piece_id: int | None = None) -> bool:
        if self.lsb_target_kind == "all":
            return True
        if self.lsb_target_kind == "original":
            return kind == "original"
        if self.lsb_target_kind == "piece":
            return kind == "piece" and piece_id == self.lsb_target_piece_id
        return False

    def render(self) -> None:
        try:
            self.render_canvas()
        except (MemoryError, tk.TclError, ValueError) as exc:
            self.log_background_error("render", exc)
            self.cancel_pending_render()
            self.photo_refs.clear()
            self.item_to_piece.clear()
            self.piece_visual_items.clear()
            self.selection_item = None
            self.background_item = None
            gc.collect()
            safe_zoom = max(self.MIN_CANVAS_ZOOM, float(self.zoom.get()) * 0.5)
            if safe_zoom < float(self.zoom.get()):
                self.zoom.set(safe_zoom)
                self.status.set(f"画布过大，已自动降到 {safe_zoom:.2f}x 缩放")
                try:
                    self.render_canvas()
                    return
                except (MemoryError, tk.TclError, ValueError) as retry_exc:
                    exc = retry_exc
            self.canvas.delete("all")
            self.draw_canvas_grid(max(self.MIN_CANVAS_ZOOM, float(self.zoom.get())))
            self.status.set(f"渲染失败，请降低缩放或减少大图：{exc}")

    def render_canvas(self) -> None:
        self.cancel_pending_render()
        self.canvas.delete("all")
        self.photo_refs.clear()
        self.item_to_piece.clear()
        self.piece_visual_items.clear()
        self.selection_item = None
        self.background_item = None
        self.render_generation += 1

        zoom = self.safe_canvas_zoom()
        max_x = 1
        max_y = 1
        self.draw_canvas_grid(zoom)
        if not self.pieces and self.original is None:
            if self.preview_title:
                self.draw_preview_state()
            else:
                self.draw_empty_state()

        if self.original is not None and self.show_background.get():
            background_source = self.display_image_for_canvas(self.original, "original")
            background = with_opacity(background_source, float(self.background_opacity.get()))
            background = background.resize(
                (max(1, int(background.width * zoom)), max(1, int(background.height * zoom))),
                resample=0,
            )
            photo = ImageTk.PhotoImage(background)
            self.photo_refs["background"] = photo
            self.background_item = self.canvas.create_image(0, 0, image=photo, anchor="nw", tags=("background",))
            max_x = max(max_x, background.width)
            max_y = max(max_y, background.height)

        for piece in self.pieces:
            max_x = max(max_x, int((piece.x + piece.width) * zoom))
            max_y = max(max_y, int((piece.y + piece.height) * zoom))

        self.canvas.configure(scrollregion=(0, 0, max_x + 80, max_y + 80))
        if self.pieces:
            self.status.set(f"正在渲染 {len(self.pieces)} 个碎片...")
            self.render_piece_batch(self.render_generation, zoom, 0)
        else:
            self.update_selection_box()

    def cancel_pending_render(self) -> None:
        if self.render_batch_after_id is None:
            return
        try:
            self.root.after_cancel(self.render_batch_after_id)
        except tk.TclError:
            pass
        self.render_batch_after_id = None

    def render_piece_batch(self, generation: int, zoom: float, start_index: int) -> None:
        try:
            if generation != self.render_generation:
                return
            end_index = min(len(self.pieces), start_index + self.RENDER_BATCH_SIZE)
            for piece in self.pieces[start_index:end_index]:
                self.create_piece_canvas_items(piece, zoom)

            if end_index < len(self.pieces):
                self.progress_text.set(f"渲染碎片 {end_index}/{len(self.pieces)}")
                self.render_batch_after_id = self.root.after(1, lambda: self.render_piece_batch(generation, zoom, end_index))
                return

            self.render_batch_after_id = None
            self.update_selection_box()
            self.status.set(f"已渲染 {len(self.pieces)} 个碎片")
        except (MemoryError, tk.TclError, ValueError) as exc:
            self.render_batch_after_id = None
            self.log_background_error("render_piece_batch", exc)
            self.photo_refs.clear()
            self.item_to_piece.clear()
            self.piece_visual_items.clear()
            gc.collect()
            self.status.set(f"渲染碎片失败，已释放画布缓存：{exc}")

    def create_piece_canvas_items(self, piece: Piece, zoom: float) -> None:
        display_source = self.display_image_for_canvas(piece.image, "piece", piece.id)
        display = display_source.resize(
            (max(1, int(piece.width * zoom)), max(1, int(piece.height * zoom))),
            resample=0,
        )
        photo = ImageTk.PhotoImage(display)
        self.photo_refs[piece.id] = photo
        x = piece.x * zoom
        y = piece.y * zoom
        width = max(1, int(piece.width * zoom))
        height = max(1, int(piece.height * zoom))
        shadow = self.canvas.create_rectangle(
            x + 3,
            y + 3,
            x + width + 3,
            y + height + 3,
            fill="#000000",
            outline="",
            stipple="gray50",
            tags=("piece-shadow", f"piece-visual-{piece.id}"),
        )
        item = self.canvas.create_image(
            x,
            y,
            image=photo,
            anchor="nw",
            tags=("piece", f"piece-{piece.id}", f"piece-visual-{piece.id}"),
        )
        frame = self.canvas.create_rectangle(
            x,
            y,
            x + width,
            y + height,
            outline=self.colors["border"],
            width=1,
            tags=("piece-frame", f"piece-visual-{piece.id}"),
            state=tk.DISABLED,
        )
        self.item_to_piece[item] = piece.id
        self.piece_visual_items[piece.id] = (shadow, item, frame)
        self.canvas.tag_lower(shadow, item)

    def safe_canvas_zoom(self) -> float:
        zoom = max(self.MIN_CANVAS_ZOOM, min(2.0, float(self.zoom.get())))
        max_pixels = self.CANVAS_PIXEL_BUDGET
        source_width = 1
        source_height = 1

        if self.original is not None and self.show_background.get():
            source_width = max(source_width, self.original.width)
            source_height = max(source_height, self.original.height)

        for piece in self.pieces:
            source_width = max(source_width, int(piece.x + piece.width))
            source_height = max(source_height, int(piece.y + piece.height))

        pixels = source_width * source_height * zoom * zoom
        if pixels > max_pixels:
            zoom = max(self.MIN_CANVAS_ZOOM, (max_pixels / max(1, source_width * source_height)) ** 0.5)
            if zoom < float(self.zoom.get()):
                self.zoom.set(zoom)
                self.status.set(f"画布较大，已自动降到 {zoom:.2f}x 缩放")
        return zoom

    def draw_canvas_grid(self, zoom: float) -> None:
        width = max(self.canvas.winfo_width(), 1200)
        height = max(self.canvas.winfo_height(), 800)
        small = max(16, int(32 * zoom))
        large = small * 4
        self.canvas.create_rectangle(0, 0, width, height, fill=self.colors["canvas"], outline="", tags=("grid-bg",))
        for x in range(0, width + small, small):
            color = self.colors["canvas_grid"] if x % large else self.colors["canvas_major"]
            self.canvas.create_line(x, 0, x, height, fill=color, width=1, tags=("grid",))
        for y in range(0, height + small, small):
            color = self.colors["canvas_grid"] if y % large else self.colors["canvas_major"]
            self.canvas.create_line(0, y, width, y, fill=color, width=1, tags=("grid",))

    def draw_empty_state(self) -> None:
        width = max(self.canvas.winfo_width(), 760)
        height = max(self.canvas.winfo_height(), 480)
        cx = width // 2
        cy = height // 2
        self.canvas.create_rectangle(
            cx - 210,
            cy - 82,
            cx + 210,
            cy + 82,
            fill=self.colors["panel"],
            outline=self.colors["border"],
            width=1,
            tags=("empty",),
        )
        self.canvas.create_text(
            cx,
            cy - 22,
            text="等待导入碎片",
            fill=self.colors["text"],
            font=("Microsoft YaHei UI", 15, "bold"),
            tags=("empty",),
        )
        self.canvas.create_text(
            cx,
            cy + 20,
            text="左侧选择碎片文件夹，或导入乱拼整图后拆分",
            fill=self.colors["muted"],
            font=("Microsoft YaHei UI", 10),
            tags=("empty",),
        )

    def draw_preview_state(self) -> None:
        width = max(self.canvas.winfo_width(), 760)
        height = max(self.canvas.winfo_height(), 480)
        cx = width // 2
        cy = height // 2
        title = self.preview_title or "图片预览"
        if self.preview_image is None:
            self.canvas.create_rectangle(
                cx - 220,
                cy - 82,
                cx + 220,
                cy + 82,
                fill=self.colors["panel"],
                outline=self.colors["accent"],
                width=1,
                tags=("preview",),
            )
            self.canvas.create_text(
                cx,
                cy - 20,
                text=title,
                fill=self.colors["text"],
                font=("Microsoft YaHei UI", 15, "bold"),
                tags=("preview",),
            )
            self.canvas.create_text(
                cx,
                cy + 24,
                text="正在准备预览，主界面可继续操作",
                fill=self.colors["muted"],
                font=("Microsoft YaHei UI", 10),
                tags=("preview",),
            )
            return

        max_width = max(240, int(width * 0.88))
        max_height = max(180, int(height * 0.78))
        scale = min(max_width / self.preview_image.width, max_height / self.preview_image.height, 1.0)
        preview_width = max(1, int(self.preview_image.width * scale))
        preview_height = max(1, int(self.preview_image.height * scale))
        resampling = getattr(Image, "Resampling", Image)
        display = self.preview_image.resize((preview_width, preview_height), resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display)
        self.photo_refs["preview"] = photo
        x = max(24, (width - preview_width) // 2)
        y = max(56, (height - preview_height) // 2)
        self.canvas.create_rectangle(
            x - 12,
            y - 44,
            x + preview_width + 12,
            y + preview_height + 12,
            fill=self.colors["panel"],
            outline=self.colors["border"],
            width=1,
            tags=("preview",),
        )
        self.canvas.create_text(
            x,
            y - 23,
            text=title,
            fill=self.colors["text"],
            font=("Microsoft YaHei UI", 12, "bold"),
            anchor="w",
            tags=("preview",),
        )
        self.canvas.create_image(x, y, image=photo, anchor="nw", tags=("preview",))
        if self.preview_path is not None:
            self.canvas.create_text(
                x + preview_width,
                y - 23,
                text=self.preview_path.name,
                fill=self.colors["muted"],
                font=("Microsoft YaHei UI", 9),
                anchor="e",
                tags=("preview",),
            )

    def update_selection_box(self) -> None:
        if self.selection_item is not None:
            self.canvas.delete("selection")
            self.selection_item = None

        selected_pieces = self.selected_pieces()
        if not selected_pieces:
            return

        zoom = max(0.05, float(self.zoom.get()))
        for index, piece in enumerate(selected_pieces):
            x0 = piece.x * zoom
            y0 = piece.y * zoom
            x1 = (piece.x + piece.width) * zoom
            y1 = (piece.y + piece.height) * zoom
            item = self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline=self.colors["accent_hover"],
                width=3,
                tags=("selection",),
                state=tk.DISABLED,
            )
            if index == 0:
                self.selection_item = item
            self.canvas.create_rectangle(
                x0 - 4,
                y0 - 4,
                x1 + 4,
                y1 + 4,
                outline=self.colors["warning"],
                width=1,
                dash=(5, 4),
                tags=("selection",),
                state=tk.DISABLED,
            )

    def on_mouse_down(self, event: tk.Event) -> None:
        if self.swap_animation_active:
            return
        current = self.canvas.find_withtag("current")
        if not current:
            self.start_marquee_selection(event)
            return

        item = current[0]
        piece_id = self.find_piece_id_from_item(item)
        if piece_id is None:
            self.start_marquee_selection(event)
            return

        piece = next(piece for piece in self.pieces if piece.id == piece_id)
        self.selected_piece_id = piece_id
        if piece_id not in self.selected_piece_ids:
            self.selected_piece_ids = {piece_id}
        self.drag_start_world = self.canvas_to_world(event)
        self.drag_piece_start = (piece.x, piece.y)
        if len(self.selected_piece_ids) > 1:
            self.drag_mode = "group"
            self.group_drag_starts = {
                selected.id: (selected.x, selected.y)
                for selected in self.pieces
                if selected.id in self.selected_piece_ids
            }
        else:
            self.drag_mode = "piece"
            self.group_drag_starts.clear()
        self.raise_piece_visual(piece.id)
        self.status.set(f"选中：{piece.name}")
        self.update_selection_text()
        self.update_selection_box()

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.swap_animation_active:
            return
        if self.drag_mode == "marquee":
            self.update_marquee_selection(event)
            return
        if self.drag_start_world is None:
            return

        current_world = self.canvas_to_world(event)
        dx = current_world[0] - self.drag_start_world[0]
        dy = current_world[1] - self.drag_start_world[1]

        if self.drag_mode == "group" and self.group_drag_starts:
            for piece in self.pieces:
                start = self.group_drag_starts.get(piece.id)
                if start is None:
                    continue
                piece.x = start[0] + dx
                piece.y = start[1] + dy
                self.move_piece_item(piece)
        else:
            piece = self.selected_piece()
            if piece is None or self.drag_piece_start is None:
                return
            piece.x = self.drag_piece_start[0] + dx
            piece.y = self.drag_piece_start[1] + dy
            self.move_piece_item(piece)
        self.update_selection_box()

    def on_mouse_up(self, event: tk.Event) -> None:
        if self.drag_mode == "marquee":
            self.finish_marquee_selection(event)
            return
        piece = self.selected_piece()
        if piece is not None and not self.swap_animation_active and self.drag_mode != "group":
            target = self.find_swap_target(piece, event)
            if target is not None:
                self.start_swap_animation(piece, target)
                return
        self.drag_start_world = None
        self.drag_piece_start = None
        self.group_drag_starts.clear()
        self.drag_mode = "none"

    def find_swap_target(self, dragged_piece: Piece, event: tk.Event) -> Piece | None:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        overlapping = self.canvas.find_overlapping(x, y, x, y)
        for item in reversed(overlapping):
            piece_id = self.find_piece_id_from_item(item)
            if piece_id is None or piece_id == dragged_piece.id:
                continue
            return next((piece for piece in self.pieces if piece.id == piece_id), None)

        dragged_center = (dragged_piece.x + dragged_piece.width / 2, dragged_piece.y + dragged_piece.height / 2)
        best_piece: Piece | None = None
        best_distance = float("inf")
        for piece in self.pieces:
            if piece.id == dragged_piece.id:
                continue
            center = (piece.x + piece.width / 2, piece.y + piece.height / 2)
            dx = dragged_center[0] - center[0]
            dy = dragged_center[1] - center[1]
            threshold = max(piece.width, piece.height, dragged_piece.width, dragged_piece.height) * 0.55
            distance = (dx * dx + dy * dy) ** 0.5
            if distance < threshold and distance < best_distance:
                best_piece = piece
                best_distance = distance
        return best_piece

    def start_marquee_selection(self, event: tk.Event) -> None:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.drag_mode = "marquee"
        self.marquee_start_canvas = (x, y)
        self.selected_piece_id = None
        self.selected_piece_ids.clear()
        self.selection_text.set("框选中")
        self.update_selection_box()
        if self.marquee_item is not None:
            self.canvas.delete(self.marquee_item)
        self.marquee_item = self.canvas.create_rectangle(
            x,
            y,
            x,
            y,
            outline=self.colors["accent_hover"],
            width=2,
            dash=(6, 4),
            fill=self.colors["accent"],
            stipple="gray25",
            tags=("marquee",),
        )

    def update_marquee_selection(self, event: tk.Event) -> None:
        if self.marquee_start_canvas is None or self.marquee_item is None:
            return
        start_x, start_y = self.marquee_start_canvas
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.marquee_item, start_x, start_y, current_x, current_y)

    def finish_marquee_selection(self, event: tk.Event) -> None:
        if self.marquee_start_canvas is None:
            self.drag_mode = "none"
            return
        start_x, start_y = self.marquee_start_canvas
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        x0, x1 = sorted((start_x, end_x))
        y0, y1 = sorted((start_y, end_y))
        zoom = max(0.05, float(self.zoom.get()))
        self.selected_piece_ids = {
            piece.id
            for piece in self.pieces
            if self.piece_intersects_canvas_rect(piece, x0, y0, x1, y1, zoom)
        }
        self.selected_piece_id = next(iter(self.selected_piece_ids), None)
        if self.marquee_item is not None:
            self.canvas.delete(self.marquee_item)
            self.marquee_item = None
        self.marquee_start_canvas = None
        self.drag_start_world = None
        self.drag_piece_start = None
        self.group_drag_starts.clear()
        self.drag_mode = "none"
        self.update_selection_text()
        self.update_selection_box()
        if self.selected_piece_ids:
            self.status.set(f"已框选 {len(self.selected_piece_ids)} 个碎片，可拖动任意选中碎片整体移动")
        else:
            self.status.set("未框选到碎片")

    @staticmethod
    def piece_intersects_canvas_rect(piece: Piece, x0: float, y0: float, x1: float, y1: float, zoom: float) -> bool:
        piece_x0 = piece.x * zoom
        piece_y0 = piece.y * zoom
        piece_x1 = (piece.x + piece.width) * zoom
        piece_y1 = (piece.y + piece.height) * zoom
        return piece_x1 >= x0 and piece_x0 <= x1 and piece_y1 >= y0 and piece_y0 <= y1

    def update_selection_text(self) -> None:
        if len(self.selected_piece_ids) > 1:
            self.selection_text.set(f"已选 {len(self.selected_piece_ids)}")
            return
        piece = self.selected_piece()
        self.selection_text.set(self.compact_name(piece.name) if piece is not None else "未选中")

    def selected_pieces(self) -> list[Piece]:
        if self.selected_piece_ids:
            return [piece for piece in self.pieces if piece.id in self.selected_piece_ids]
        piece = self.selected_piece()
        return [piece] if piece is not None else []

    def raise_piece_visual(self, piece_id: int) -> None:
        items = self.piece_visual_items.get(piece_id)
        if items is None:
            return
        for item in items:
            self.canvas.tag_raise(item)

    def start_swap_animation(self, first: Piece, second: Piece) -> None:
        first_start = (first.x, first.y)
        second_start = (second.x, second.y)
        first_target = second_start
        second_target = self.drag_piece_start or first_start
        first.matched_cell, second.matched_cell = second.matched_cell, first.matched_cell
        first.score = None
        second.score = None
        self.drag_start_world = None
        self.drag_piece_start = None
        self.swap_animation_active = True
        self.animate_piece_swap(first, second, first_start, first_target, second_start, second_target, frame=0, frames=12)

    def animate_piece_swap(
        self,
        first: Piece,
        second: Piece,
        first_start: tuple[float, float],
        first_target: tuple[float, float],
        second_start: tuple[float, float],
        second_target: tuple[float, float],
        frame: int,
        frames: int,
    ) -> None:
        if not self.swap_animation_active:
            return
        progress = min(1.0, frame / max(1, frames))
        eased = 1 - (1 - progress) * (1 - progress)
        first.x = first_start[0] + (first_target[0] - first_start[0]) * eased
        first.y = first_start[1] + (first_target[1] - first_start[1]) * eased
        second.x = second_start[0] + (second_target[0] - second_start[0]) * eased
        second.y = second_start[1] + (second_target[1] - second_start[1]) * eased
        self.move_piece_item(first)
        self.move_piece_item(second)
        self.update_selection_box()

        if frame >= frames:
            first.x, first.y = first_target
            second.x, second.y = second_target
            self.move_piece_item(first)
            self.move_piece_item(second)
            self.swap_animation_active = False
            self.result = None
            self.grid_text.set("手动交换")
            self.status.set(f"已交换：{first.name} <-> {second.name}")
            self.update_selection_box()
            return

        self.root.after(
            16,
            lambda: self.animate_piece_swap(
                first,
                second,
                first_start,
                first_target,
                second_start,
                second_target,
                frame + 1,
                frames,
            ),
        )

    def move_piece_item(self, piece: Piece) -> None:
        zoom = max(0.05, float(self.zoom.get()))
        items = self.piece_visual_items.get(piece.id)
        if items is None:
            return
        shadow, item, frame = items
        x = piece.x * zoom
        y = piece.y * zoom
        width = max(1, int(piece.width * zoom))
        height = max(1, int(piece.height * zoom))
        self.canvas.coords(shadow, x + 3, y + 3, x + width + 3, y + height + 3)
        self.canvas.coords(item, x, y)
        self.canvas.coords(frame, x, y, x + width, y + height)

    def on_context_menu(self, event: tk.Event) -> None:
        item = self.find_context_item(event)
        if item is None:
            return

        piece_id = self.item_to_piece.get(item)
        self.context_piece_id = piece_id
        self.context_image_kind = "piece" if piece_id is not None else "original"
        label = "碎片" if piece_id is not None else "原图"

        menu = tk.Menu(self.root, tearoff=0, bg=self.colors["panel_2"], fg=self.colors["text"], activebackground=self.colors["accent"])
        menu.add_command(label=f"{label} 应用当前 LSB 设置", command=self.open_lsb_dialog)
        menu.add_separator()
        for channel in ("R", "G", "B", "A", "RGB"):
            channel_menu = tk.Menu(
                menu,
                tearoff=0,
                bg=self.colors["panel_2"],
                fg=self.colors["text"],
                activebackground=self.colors["accent"],
            )
            for bit in range(8):
                channel_menu.add_command(
                    label=f"bit{bit}",
                    command=lambda item_channel=channel, item_bit=bit: self.show_lsb_preview(item_channel, item_bit),
                )
            menu.add_cascade(label=f"LSB {channel}", menu=channel_menu)
        menu.add_separator()
        menu.add_command(label="关闭 LSB 显示", command=self.disable_lsb_display)

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def find_context_item(self, event: tk.Event) -> int | None:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        for item in reversed(self.canvas.find_overlapping(x, y, x, y)):
            if item in self.item_to_piece:
                return item
            if self.background_item is not None and item == self.background_item:
                return item
        return None

    def context_image(self):
        if self.context_image_kind == "piece" and self.context_piece_id is not None:
            piece = next((candidate for candidate in self.pieces if candidate.id == self.context_piece_id), None)
            return piece.image if piece is not None else None
        if self.context_image_kind == "original":
            return self.original
        return None

    def open_lsb_dialog(self) -> None:
        image = self.context_image()
        if image is None:
            self.status.set("没有可查看的图片")
            return
        self.apply_lsb_to_context(self.lsb_channel.get(), int(self.lsb_bit.get()))

    def show_lsb_preview(self, channel: str, bit: int) -> None:
        image = self.context_image()
        if image is None:
            self.status.set("没有可查看的图片")
            return
        self.apply_lsb_to_context(channel, bit)

    def apply_lsb_to_context(self, channel: str, bit: int) -> None:
        self.lsb_channel.set(channel)
        self.lsb_bit.set(bit)
        if self.context_image_kind == "piece" and self.context_piece_id is not None:
            self.set_lsb_target("piece", self.context_piece_id)
            return
        if self.context_image_kind == "original":
            self.set_lsb_target("original")
            return
        self.status.set("没有可应用 LSB 的图片")

    def apply_lsb_to_selected(self) -> None:
        piece = self.selected_piece()
        if piece is None:
            self.status.set("请先选中一个碎片")
            return
        self.set_lsb_target("piece", piece.id)

    def apply_lsb_to_all(self) -> None:
        if self.original is None and not self.pieces:
            self.status.set("没有可应用 LSB 的图片")
            return
        if len(self.pieces) > 120:
            self.status.set("正在分批渲染全部 LSB，图片较多时会稍慢")
        self.set_lsb_target("all")

    def on_lsb_control_changed(self) -> None:
        if self.lsb_enabled.get():
            self.update_lsb_target_text()
            self.render()

    def set_lsb_target(self, kind: str, piece_id: int | None = None) -> None:
        self.lsb_enabled.set(True)
        self.lsb_target_kind = kind
        self.lsb_target_piece_id = piece_id
        self.update_lsb_target_text()
        self.status.set(f"已在工作区显示 LSB {self.lsb_channel.get()} bit{int(self.lsb_bit.get())}")
        self.render()

    def update_lsb_target_text(self) -> None:
        if not self.lsb_enabled.get():
            self.lsb_target_text.set("LSB 关闭")
            return
        channel = self.lsb_channel.get()
        bit = int(self.lsb_bit.get())
        if self.lsb_target_kind == "all":
            target = "全部图片"
        elif self.lsb_target_kind == "original":
            target = "原图"
        elif self.lsb_target_kind == "piece":
            piece = next((candidate for candidate in self.pieces if candidate.id == self.lsb_target_piece_id), None)
            target = f"碎片 {self.compact_name(piece.name)}" if piece is not None else "碎片"
        else:
            target = "未选择"
        self.lsb_target_text.set(f"{target}: {channel} bit{bit}")

    def disable_lsb_display(self, render: bool = True) -> None:
        self.lsb_enabled.set(False)
        self.lsb_target_kind = "none"
        self.lsb_target_piece_id = None
        self.lsb_target_text.set("LSB 关闭")
        if render:
            self.status.set("已关闭 LSB 显示")
            self.render()

    def on_mouse_wheel(self, event: tk.Event) -> None:
        if event.state & 0x0004:
            self.bump_zoom(0.08 if event.delta > 0 else -0.08)
            return
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def bump_zoom(self, delta: float) -> None:
        value = max(self.MIN_CANVAS_ZOOM, min(2.0, float(self.zoom.get()) + delta))
        self.zoom.set(value)
        self.render()

    def canvas_to_world(self, event: tk.Event) -> tuple[float, float]:
        zoom = max(0.05, float(self.zoom.get()))
        return (self.canvas.canvasx(event.x) / zoom, self.canvas.canvasy(event.y) / zoom)

    def find_item_for_piece(self, piece_id: int) -> int | None:
        for item, mapped_piece_id in self.item_to_piece.items():
            if mapped_piece_id == piece_id:
                return item
        return None

    def find_piece_id_from_item(self, item: int) -> int | None:
        piece_id = self.item_to_piece.get(item)
        if piece_id is not None:
            return piece_id
        for tag in self.canvas.gettags(item):
            if not tag.startswith("piece-visual-"):
                continue
            try:
                return int(tag.removeprefix("piece-visual-"))
            except ValueError:
                return None
        return None

    def selected_grid(self) -> tuple[int, int] | None:
        try:
            rows = int(self.grid_rows.get().strip() or "0")
            cols = int(self.grid_cols.get().strip() or "0")
        except ValueError as exc:
            raise ValueError("行数和列数必须是整数；填 0 表示自动推断") from exc
        if rows < 0 or cols < 0:
            raise ValueError("行数和列数不能为负数；填 0 表示自动推断")
        if rows == 0 and cols == 0:
            return None
        return (rows, cols)

    @staticmethod
    def compact_name(name: str, limit: int = 14) -> str:
        if len(name) <= limit:
            return name
        keep = max(4, limit - 3)
        return f"{name[:keep]}..."

    @staticmethod
    def compact_path(path: Path, limit: int = 32) -> str:
        text = str(path)
        if len(text) <= limit:
            return text
        return f"...{text[-(limit - 3):]}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ez_Jigsaw")
    parser.add_argument("--pieces", type=Path, help="拼图碎片文件夹")
    parser.add_argument("--original", type=Path, help="完整原图，可选")
    parser.add_argument("--rows", type=int, default=0, help="矩形拼图行数，0 表示自动推断")
    parser.add_argument("--cols", type=int, default=0, help="矩形拼图列数，0 表示自动推断")
    parser.add_argument("--no-auto", action="store_true", help="启动后不自动拼接")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = tk.Tk()
    app = PuzzleApp(root)
    app.grid_rows.set(str(args.rows))
    app.grid_cols.set(str(args.cols))

    if args.pieces:
        app.start_load_project(args.pieces, args.original, auto_solve=not args.no_auto)

    root.mainloop()
    return 0
