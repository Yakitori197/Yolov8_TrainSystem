#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影像物件偵測與自動分檔系統 v1.0
- 攝影機狀態
- 多相機預覽穩定
- 標註：支持自定義類別管理、JSON↔YOLO 自動轉換（LabelMe/objects/bboxes）、自動建立空白標註
- 訓練：進度條百分比（callbacks）、日誌即時輸出
- 訓練結果：可選任一模型，一鍵啟用「全攝影機即時辨識」/ 停止
- 自動截圖：可設定間隔時間自動擷取所有相機畫面
"""

import os
import json
import time
import yaml
import math
import queue
import shutil
import random
import threading
from datetime import datetime
from pathlib import Path

# ====== GUI / 第三方套件檢查 ======
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, simpledialog
    TKINTER_AVAILABLE = True
except Exception:
    TKINTER_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    from PIL import Image, ImageTk, ImageDraw
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


class USBLearningApp:
    def __init__(self, use_gui: bool = True):
        self.use_gui = use_gui and TKINTER_AVAILABLE

        # 路徑
        self.base_dir = Path(os.getcwd())
        self.images_dir = self.base_dir / "images"
        self.ann_dir = self.base_dir / "annotations"
        self.models_dir = self.base_dir / "models"
        self.sorted_output_dir = self.base_dir / "sorted_output"
        self.dataset_root = self.base_dir / "dataset"

        # 預設設定（config.yaml 存在時會覆蓋這些值，見 _load_config）
        self.config = {
            "camera_settings": {
                "max_cameras": 12,
                "camera_names": [],
                "capture_format": "png",
                "resolution": {"width": 1920, "height": 1080},
                "fps": 30
            },
            "training_settings": {
                "default_epochs": 100,
                "default_img_size": 640,
                "default_batch_size": 16,
                "data_split_ratio": 0.8
            },
            "detection_settings": {
                "confidence_threshold": 0.5,
                "auto_sort": True
            },
            "object_classes": ["object", "defect", "good", "bad"]
        }

        # 實際載入 config.yaml 覆蓋預設值。
        # 此檔原本從未被讀取（設定全寫死在上面），改 config.yaml 完全沒作用且不報錯。
        self._load_config()

        # 攝影機狀態
        self.available_cameras = []
        self.active_cameras = {}
        self.camera_threads = {}
        self.camera_queues = {}
        self.camera_frames = {}
        self.rt_detect_enabled = False
        self.rt_det_threads = {}
        self.rt_det_flags = {}
        self.rt_frames = {}

        # 自動截圖
        self.auto_capture_enabled = False
        self.auto_capture_interval = 5
        self.auto_capture_thread = None

        # 預覽循環
        self.preview_updater_running = False
        self.preview_after_id = None
        self.preview_tile_w = 320
        self.preview_tile_h = 240
        self.preview_fps_ms = 40
        self.preview_cols_fixed = 4
        self.show_overlay_names = None
        self.status_visible = None

        # 標註項
        self.image_list = []
        self.current_index = 0
        self.current_image = None
        self.current_image_path = None
        self.current_ann = []
        self.current_boxes = []
        self.scale = 1.0
        self.image_x_offset = 0
        self.image_y_offset = 0
        self.drawing = False
        self.start_x = 0
        self.start_y = 0

        # 訓練/推論
        self.current_model_path = None
        self.yolo_model = None
        self.train_thread = None
        self.train_stop_flag = threading.Event()

        if self.use_gui:
            self.root = tk.Tk()
            self.root.title("影像物件偵測與自動分檔系統 v1.0")
            self.root.geometry("1280x820")
            try:
                self.root.state('zoomed')
            except Exception:
                pass

            self.setup_directories()
            self.setup_ui()
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_directories(self):
        for d in [self.images_dir, self.ann_dir, self.models_dir, self.sorted_output_dir, self.dataset_root]:
            d.mkdir(parents=True, exist_ok=True)

    def setup_ui(self):
        self.show_overlay_names = tk.BooleanVar(value=False)

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True)

        self.tab_camera = ttk.Frame(nb); nb.add(self.tab_camera, text="即時預覽")
        self.tab_annot  = ttk.Frame(nb); nb.add(self.tab_annot,  text="標註作業")
        self.tab_train  = ttk.Frame(nb); nb.add(self.tab_train,  text="訓練模型")
        self.tab_result = ttk.Frame(nb); nb.add(self.tab_result, text="訓練結果")

        self.build_camera_tab(self.tab_camera)
        self.build_annot_tab(self.tab_annot)
        self.build_train_tab(self.tab_train)
        self.build_result_tab(self.tab_result)

        nb.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    # === 攝影機頁（完全保持不變，省略以節省空間）===
    def build_camera_tab(self, parent):
        top = ttk.Frame(parent); top.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(top, text="檢測所有攝影機", command=self.detect_cameras).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="開啟所有攝影機", command=self.open_all_cameras).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="關閉所有攝影機", command=self.close_all_cameras).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="統一同步拍照", command=self.capture_all_cameras).pack(side=tk.LEFT, padx=5)
        ttk.Separator(top, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(top, text="開始自動截圖", command=self.start_auto_capture).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="停止自動截圖", command=self.stop_auto_capture).pack(side=tk.LEFT, padx=5)
        ttk.Label(top, text="間隔(秒):").pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.StringVar(value="5")
        interval_entry = ttk.Entry(top, textvariable=self.interval_var, width=5)
        interval_entry.pack(side=tk.LEFT, padx=5)
        self.camera_count_label = ttk.Label(top, text="未檢測", foreground="blue")
        self.camera_count_label.pack(side=tk.LEFT, padx=20)
        self.auto_capture_label = ttk.Label(top, text="", foreground="red")
        self.auto_capture_label.pack(side=tk.LEFT, padx=10)
        self.status_container = ttk.LabelFrame(parent, text="攝影機控制")
        self.status_container.pack(fill=tk.X, padx=10, pady=5)
        self.status_canvas = tk.Canvas(self.status_container, height=100, bg="#fafafa", highlightthickness=0)
        self.status_canvas.pack(fill=tk.X, expand=False, side=tk.TOP)
        self.status_scroll_x = ttk.Scrollbar(self.status_container, orient="horizontal", command=self.status_canvas.xview)
        self.status_scroll_x.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_canvas.configure(xscrollcommand=self.status_scroll_x.set)
        self.camera_status_frame = ttk.Frame(self.status_canvas)
        self.status_canvas_window = self.status_canvas.create_window((0, 0), window=self.camera_status_frame, anchor="nw")
        self.camera_status_frame.bind("<Configure>", lambda e: self.status_canvas.configure(scrollregion=self.status_canvas.bbox("all")))
        self.status_canvas.bind("<Configure>", lambda e: self.status_canvas.itemconfig(self.status_canvas_window, height=100))
        preview = ttk.LabelFrame(parent, text="即時預覽"); preview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.preview_canvas = tk.Canvas(preview, bg="black"); self.preview_canvas.pack(fill=tk.BOTH, expand=True)

    def toggle_status_visibility(self):
        if self.status_visible.get():
            self.status_container.pack(fill=tk.X, padx=10, pady=5)
        else:
            self.status_container.pack_forget()

    # ---------------- 設定載入 ----------------

    def _load_config(self, path=None) -> None:
        """
        載入 config.yaml 覆蓋預設設定。

        檔案不存在或格式錯誤時沿用預設值並提示，不讓程式崩潰。
        """
        cfg_path = Path(path) if path else (self.base_dir / "config.yaml")
        if not cfg_path.exists():
            return
        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"[config] config.yaml 格式錯誤，沿用預設設定：{e}")
            return
        if not isinstance(data, dict):
            print("[config] config.yaml 內容不是對應表，沿用預設設定")
            return

        data = dict(data)
        # object_classes 相容處理：程式內部一律當「清單」使用。
        # 早期範本把它寫成 {英文: 中文} 字典，若直接載入，[0] 與 .append() 會出錯。
        if "object_classes" in data:
            data["object_classes"] = _normalize_classes(data["object_classes"])
            if not data["object_classes"]:
                del data["object_classes"]  # 空的就不要覆蓋預設

        self.config = _deep_merge(self.config, data)

    def _camera_params(self):
        """回傳 (寬, 高, fps)，皆取自設定而非寫死。"""
        cam = self.config.get("camera_settings", {})
        res = cam.get("resolution", {}) or {}
        return (
            int(res.get("width", 1920)),
            int(res.get("height", 1080)),
            int(cam.get("fps", 30)),
        )

    def _ui_message(self, kind: str, title: str, msg: str) -> None:
        """
        從背景執行緒安全地顯示對話框。

        Tkinter 只能在主執行緒操作；直接在 worker thread 呼叫 messagebox
        可能導致 Tk 崩潰，故一律經由 root.after 排回主執行緒。
        """
        if not self.use_gui:
            print(f"[{kind}] {title}: {msg}")
            return
        fn = {
            "error": messagebox.showerror,
            "info": messagebox.showinfo,
            "warning": messagebox.showwarning,
        }.get(kind, messagebox.showinfo)
        try:
            self.root.after(0, lambda: fn(title, msg))
        except Exception:
            pass

    def detect_cameras(self):
        if not OPENCV_AVAILABLE:
            messagebox.showerror("錯誤", "未安裝 OpenCV"); return
        self.available_cameras.clear()
        for i in range(self.config["camera_settings"]["max_cameras"]):
            cap = None
            try:
                cap = cv2.VideoCapture(i)
                if not cap.isOpened(): continue
                cam_w, cam_h, cam_fps = self._camera_params()  # 取自設定，不再寫死
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
                cap.set(cv2.CAP_PROP_FPS, cam_fps)
                actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                if actual_fps <= 0:
                    actual_fps = 30
                ret, frame = cap.read()
                if not (ret and frame is not None): continue
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
                names = self.config["camera_settings"]["camera_names"]
                display_name = names[i] if i < len(names) else f"攝影機{i+1}"
                self.available_cameras.append({"index": i, "name": display_name, "resolution": f"{width}x{height}", "fps": fps})
            finally:
                if cap is not None: cap.release()
        if self.available_cameras:
            self.camera_count_label.config(text=f"發現 {len(self.available_cameras)} 台攝影機", foreground="green")
            self.update_camera_status_display()
            messagebox.showinfo("完成", f"共發現 {len(self.available_cameras)} 台攝影機")
        else:
            self.camera_count_label.config(text="未發現攝影機", foreground="red")
            messagebox.showwarning("提示", "未發現可用攝影機")

    def _status_card(self, parent, cam):
        fr = ttk.Frame(parent, relief="groove", borderwidth=1)
        fr.pack(side=tk.LEFT, padx=4, pady=4)
        row1 = ttk.Frame(fr)
        row1.pack(fill=tk.X, padx=6, pady=(4, 2))
        ttk.Label(row1, text=cam["name"], width=15, anchor="center").pack(side=tk.LEFT)
        opened = cam["index"] in self.active_cameras
        ttk.Label(row1, text=("開啟" if opened else "關閉"), 
              foreground=("green" if opened else "red"), 
              width=6, anchor="center").pack(side=tk.LEFT, padx=4)
        row2 = ttk.Frame(fr)
        row2.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row2, text=cam["resolution"], width=15, anchor="center").pack(side=tk.LEFT)
        ttk.Label(row2, text=f'{cam["fps"]} fps', width=10, anchor="center").pack(side=tk.LEFT)
        row3 = ttk.Frame(fr)
        row3.pack(fill=tk.X, padx=6, pady=(2, 4))
        ttk.Button(row3, text="開啟", 
               command=lambda idx=cam["index"]: self.open_single_camera(idx), 
               width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(row3, text="關閉", 
               command=lambda idx=cam["index"]: self.close_single_camera(idx), 
               width=8).pack(side=tk.LEFT, padx=2)
        return fr

    def update_camera_status_display(self):
        for w in self.camera_status_frame.winfo_children(): w.destroy()
        for cam in self.available_cameras:
            self._status_card(self.camera_status_frame, cam)
        self.status_canvas.update_idletasks()
        self.status_canvas.configure(scrollregion=self.status_canvas.bbox("all"))

    def open_single_camera(self, index: int):
        if index in self.active_cameras: return
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            messagebox.showwarning("警告", f"攝影機 {index+1} 無法開啟"); return
        cam_w, cam_h, cam_fps = self._camera_params()  # 取自設定，不再寫死
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        cap.set(cv2.CAP_PROP_FPS, cam_fps)
        self.active_cameras[index] = cap
        self.camera_queues[index] = queue.Queue(maxsize=2)
        self.camera_frames[index] = None
        t = threading.Thread(target=self.camera_capture_thread, args=(index,), daemon=True)
        self.camera_threads[index] = {"thread": t, "active": True}
        t.start()
        self.update_camera_status_display()
        self.start_preview_loop_if_needed()
        if self.rt_detect_enabled: self._ensure_detector_thread(index)

    def close_single_camera(self, index: int):
        if index not in self.active_cameras: return
        self._stop_detector_thread(index)
        info = self.camera_threads.get(index)
        if info:
            info["active"] = False
            t = info.get("thread")
            if t and t.is_alive(): t.join(timeout=1.0)
            self.camera_threads.pop(index, None)
        cap = self.active_cameras.pop(index, None)
        if cap is not None:
            try: cap.release()
            except Exception: pass
        self.camera_queues.pop(index, None)
        self.camera_frames.pop(index, None)
        self.rt_frames.pop(index, None)
        self.update_camera_status_display()
        if not self.active_cameras: self.stop_preview_loop()

    def open_all_cameras(self):
        for cam in self.available_cameras: self.open_single_camera(cam["index"])

    def close_all_cameras(self):
        for idx in list(self.active_cameras.keys()): self.close_single_camera(idx)

    def camera_capture_thread(self, index: int):
        while self.camera_threads.get(index, {}).get("active", False):
            cap = self.active_cameras.get(index)
            if cap is None: break
            ret, frame = cap.read()
            if ret and frame is not None:
                q = self.camera_queues.get(index)
                if q is not None:
                    if q.full():
                        try: q.get_nowait()
                        except Exception: pass
                    try: q.put_nowait(frame)
                    except Exception: pass
            else:
                time.sleep(0.005)

    def start_auto_capture(self):
        if self.auto_capture_thread and self.auto_capture_thread.is_alive():
            messagebox.showinfo("提示", "自動截圖已在執行中")
            return
        try:
            self.auto_capture_interval = float(self.interval_var.get())
            if self.auto_capture_interval <= 0:
                messagebox.showwarning("警告", "間隔時間必須大於0")
                return
        except ValueError:
            messagebox.showwarning("警告", "請輸入有效的數字")
            return
        if not self.active_cameras:
            messagebox.showwarning("警告", "請先開啟攝影機")
            return
        self.auto_capture_enabled = True
        self.auto_capture_thread = threading.Thread(target=self._auto_capture_worker, daemon=True)
        self.auto_capture_thread.start()
        self.auto_capture_label.config(text=f"自動截圖中({self.auto_capture_interval}秒)", foreground="green")
        self.append_log(f"已開始自動截圖，間隔：{self.auto_capture_interval}秒")

    def stop_auto_capture(self):
        self.auto_capture_enabled = False
        if self.auto_capture_thread and self.auto_capture_thread.is_alive():
            self.auto_capture_thread.join(timeout=1.0)
        self.auto_capture_label.config(text="", foreground="red")
        self.append_log("已停止自動截圖")

    def _auto_capture_worker(self):
        while self.auto_capture_enabled:
            if self.active_cameras:
                self.capture_all_cameras()
            time.sleep(self.auto_capture_interval)

    def start_preview_loop_if_needed(self):
        if not self.preview_updater_running:
            self.preview_updater_running = True; self.schedule_next_preview()

    def stop_preview_loop(self):
        self.preview_updater_running = False
        if self.preview_after_id is not None:
            try: self.root.after_cancel(self.preview_after_id)
            except Exception: pass
            self.preview_after_id = None
        if hasattr(self, "preview_canvas"): self.preview_canvas.delete("all")

    def schedule_next_preview(self):
        if self.preview_updater_running:
            self.preview_after_id = self.root.after(self.preview_fps_ms, self.update_camera_previews)

    def _make_placeholder_tile(self, text="無訊號"):
        if not PIL_AVAILABLE:
            return None
        im = Image.new("RGB", (self.preview_tile_w, self.preview_tile_h), (0, 0, 0))
        draw = ImageDraw.Draw(im); draw.text((12, self.preview_tile_h // 2), text, fill=(180, 180, 180))
        return im

    def update_camera_previews(self):
        if not self.preview_updater_running or not PIL_AVAILABLE: return
        indices = sorted(list(self.active_cameras.keys()))
        n = len(indices)
        tiles = []
        if n > 0:
            for idx in indices:
                f = None
                if self.rt_detect_enabled and idx in self.rt_frames:
                    f = self.rt_frames.get(idx)
                else:
                    q = self.camera_queues.get(idx)
                    if q is not None:
                        try:
                            while q.qsize() > 1: q.get_nowait()
                            f = q.get_nowait()
                        except Exception: pass
                    if f is None: f = self.camera_frames.get(idx)
                if f is not None and OPENCV_AVAILABLE:
                    if not self.rt_detect_enabled:
                        self.camera_frames[idx] = f
                    f2 = cv2.resize(f, (self.preview_tile_w, self.preview_tile_h))
                    if self.show_overlay_names.get() and not self.rt_detect_enabled:
                        name = self._camera_name_by_index(idx)
                        cv2.putText(f2, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    tiles.append(Image.fromarray(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)))
                else:
                    name = self._camera_name_by_index(idx); tiles.append(self._make_placeholder_tile(f"{name} 等待影像…"))
            cols = self.preview_cols_fixed; rows = math.ceil(n / cols)
            big_w = cols * self.preview_tile_w; big_h = rows * self.preview_tile_h
            big = Image.new("RGB", (big_w, big_h), (0, 0, 0))
            k = 0
            for r in range(rows):
                for c in range(cols):
                    if k < len(tiles): big.paste(tiles[k], (c*self.preview_tile_w, r*self.preview_tile_h)); k += 1
            self.preview_photo = ImageTk.PhotoImage(image=big)
            self.preview_canvas.delete("all"); self.preview_canvas.create_image(0, 0, anchor="nw", image=self.preview_photo)
        self.schedule_next_preview()

    def _camera_name_by_index(self, i: int) -> str:
        names = self.config["camera_settings"]["camera_names"]
        return names[i] if i < len(names) else f"攝影機{i+1}"

    def capture_all_cameras(self):
        if not self.active_cameras:
            if not self.auto_capture_enabled:
                messagebox.showwarning("警告", "尚未開啟任何攝影機")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fmt = self.config["camera_settings"]["capture_format"].lower()
        saved = 0
        for idx, cap in self.active_cameras.items():
            ret, frame = cap.read()
            if not (ret and frame is not None):
                print(f"⚠    攝影機 {idx+1} 讀幀失敗"); continue
            fn = f"product_{ts}_cam{idx+1}.{fmt}"
            outp = self.images_dir / fn
            if OPENCV_AVAILABLE: cv2.imwrite(str(outp), frame)
            saved += 1
        if not self.auto_capture_enabled:
            messagebox.showinfo("完成", f"已截圖 {saved} 張影像至 images/")
        else:
            print(f"自動截圖：已儲存 {saved} 張影像")
# === 標註頁 ===
    def build_annot_tab(self, parent):
        top = ttk.Frame(parent); top.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(top, text="載入影像資料夾", command=self.load_images_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="上一張", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="下一張", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="儲存標註", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="清除標註", command=self.clear_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Separator(top, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(top, text="標註類別：").pack(side=tk.LEFT, padx=8)
        self.class_var = tk.StringVar(value=self.config["object_classes"][0])
        self.class_box = ttk.Combobox(top, textvariable=self.class_var, values=self.config["object_classes"], state="readonly", width=12)
        self.class_box.pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="新增類別", command=self.add_new_class).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="刪除類別", command=self.delete_current_class).pack(side=tk.LEFT, padx=5)
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        info_frame = ttk.Frame(left_frame)
        info_frame.pack(fill=tk.X, pady=5)
        self.image_info_label = ttk.Label(info_frame, text="未載入影像", foreground="blue")
        self.image_info_label.pack(side=tk.LEFT)
        self.annot_canvas = tk.Canvas(left_frame, bg="white")
        self.annot_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        self.annot_canvas.bind("<Button-1>", self.start_draw)
        self.annot_canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.annot_canvas.bind("<ButtonRelease-1>", self.finish_draw)
        self.annot_canvas.bind("<Button-3>", self.delete_last_annotation)
        right_frame = ttk.LabelFrame(main_frame, text="標註統計", width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)
        stats_frame = ttk.Frame(right_frame)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(stats_frame, text="總體統計", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.total_images_label = ttk.Label(stats_frame, text="總影像數量: 0")
        self.total_images_label.pack(anchor="w", pady=2)
        self.annotated_images_label = ttk.Label(stats_frame, text="已標註影像: 0")
        self.annotated_images_label.pack(anchor="w", pady=2)
        self.unannotated_images_label = ttk.Label(stats_frame, text="未標註影像: 0")
        self.unannotated_images_label.pack(anchor="w", pady=2)
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        current_stats_frame = ttk.Frame(stats_frame)
        current_stats_frame.pack(fill=tk.X)
        ttk.Label(current_stats_frame, text="當前影像", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.current_annotations_label = ttk.Label(current_stats_frame, text="標註數量: 0")
        self.current_annotations_label.pack(anchor="w", pady=2)
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(stats_frame, text="類別統計", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.class_stats_frame = ttk.Frame(stats_frame)
        self.class_stats_frame.pack(fill=tk.X)
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(stats_frame, text="標註進度", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.annotation_progress = ttk.Progressbar(stats_frame, mode="determinate", length=200)
        self.annotation_progress.pack(fill=tk.X, pady=5)
        self.progress_label = ttk.Label(stats_frame, text="0%", font=("Arial", 9))
        self.progress_label.pack(anchor="w", pady=2)
        self.annot_canvas.bind("<Key-s>", lambda e: self.save_annotations())
        self.annot_canvas.bind("<Key-Left>", lambda e: self.prev_image())
        self.annot_canvas.bind("<Key-Right>", lambda e: self.next_image())
        self.annot_canvas.focus_set()

    def add_new_class(self):
        new_class = simpledialog.askstring("新增類別", "請輸入新類別名稱:")
        if new_class:
            new_class = new_class.strip()
            if not new_class:
                messagebox.showwarning("警告", "類別名稱不能為空")
                return
            if new_class in self.config["object_classes"]:
                messagebox.showwarning("警告", "類別名稱已存在")
                return
            self.config["object_classes"].append(new_class)
            self.class_box['values'] = self.config["object_classes"]
            self.class_var.set(new_class)
            self.update_annotation_statistics()
            messagebox.showinfo("成功", f"已新增類別：{new_class}")

    def delete_current_class(self):
        current_class = self.class_var.get()
        if len(self.config["object_classes"]) <= 1:
            messagebox.showwarning("警告", "至少需要保留一個類別")
            return
        is_in_use = self.check_class_in_use(current_class)
        if is_in_use:
            result = messagebox.askyesno("類別使用中", 
                                       f"類別 '{current_class}' 正在一些標註中使用。\n\n"
                                       f"刪除此類別將會影響相關的標註檔案。\n\n"
                                       f"確定要刪除嗎？")
            if not result:
                return
        else:
            result = messagebox.askyesno("確認刪除", f"確定要刪除類別 '{current_class}' 嗎？")
            if not result:
                return
        class_idx = self.config["object_classes"].index(current_class)
        self.config["object_classes"].remove(current_class)
        self.clean_current_annotations_for_deleted_class(class_idx)
        self.class_box['values'] = self.config["object_classes"]
        if self.config["object_classes"]:
            self.class_var.set(self.config["object_classes"][0])
        self.redraw_annotations()
        self.update_annotation_statistics()
        messagebox.showinfo("成功", f"已刪除類別：{current_class}")

    def check_class_in_use(self, class_name):
        if not self.image_list:
            return False
        class_idx = self.config["object_classes"].index(class_name)
        for img_path in self.image_list:
            img_path_obj = Path(img_path)
            txt_path = (self.ann_dir / img_path_obj.stem).with_suffix(".txt")
            if txt_path.exists():
                try:
                    content = txt_path.read_text(encoding="utf-8").strip()
                    if content:
                        for line in content.splitlines():
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    cls_idx = int(float(parts[0]))
                                    if cls_idx == class_idx:
                                        return True
                                except:
                                    pass
                except:
                    pass
            json_path = (self.ann_dir / img_path_obj.stem).with_suffix(".json")
            if json_path.exists():
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "shapes" in data:
                        for shape in data["shapes"]:
                            if shape.get("label", "") == class_name:
                                return True
                except:
                    pass
        return False

    def clean_current_annotations_for_deleted_class(self, deleted_class_idx):
        new_annotations = []
        for (cls_idx, x1, y1, x2, y2) in self.current_ann:
            if cls_idx == deleted_class_idx:
                continue
            elif cls_idx > deleted_class_idx:
                new_annotations.append((cls_idx - 1, x1, y1, x2, y2))
            else:
                new_annotations.append((cls_idx, x1, y1, x2, y2))
        self.current_ann = new_annotations

    def load_images_folder(self):
        folder = filedialog.askdirectory(title="選擇影像資料夾", initialdir=str(self.images_dir))
        if not folder: return
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = []
        for root, _, fnames in os.walk(folder):
            for f in fnames:
                if f.lower().endswith(exts): files.append(str(Path(root) / f))
        files.sort()
        if not files:
            messagebox.showwarning("提示", "此資料夾沒有影像"); return
        self.image_list = files; self.current_index = 0; self.load_current_image()
        self.update_annotation_statistics()
        messagebox.showinfo("完成", f"已載入 {len(files)} 張影像")

    def load_current_image(self):
        if not self.image_list or self.current_index < 0 or self.current_index >= len(self.image_list): return
        path = Path(self.image_list[self.current_index]); self.current_image_path = path
        if OPENCV_AVAILABLE: self.current_image = cv2.imread(str(path))
        else: self.current_image = None
        if self.current_image is None: return
        filename = path.name
        progress = f"{self.current_index + 1}/{len(self.image_list)}"
        self.image_info_label.config(text=f"當前影像：{filename} ({progress})")
        self.load_existing_annotations(); self.display_annotation_image()
        self.update_annotation_statistics()

    def display_annotation_image(self):
        if self.current_image is None or not OPENCV_AVAILABLE or not PIL_AVAILABLE: return
        img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        ch, cw = img.shape[:2]
        W = self.annot_canvas.winfo_width() or 800; H = self.annot_canvas.winfo_height() or 600
        sx = W / cw; sy = H / ch; self.scale = min(sx, sy)
        disp = cv2.resize(img, (int(cw*self.scale), int(ch*self.scale)))
        im = Image.fromarray(disp); self.annot_photo = ImageTk.PhotoImage(image=im)
        self.annot_canvas.delete("all")
        self.image_x_offset = (W - disp.shape[1])//2; self.image_y_offset = (H - disp.shape[0])//2
        self.annot_canvas.create_image(self.image_x_offset, self.image_y_offset, anchor="nw", image=self.annot_photo)
        self.redraw_annotations()

    def redraw_annotations(self):
        for rid in getattr(self, "current_boxes", []): self.annot_canvas.delete(rid)
        self.current_boxes = []
        if self.current_image is None: return
        colors = ["lime", "red", "blue", "yellow", "magenta", "cyan", "orange", "purple"]
        for (cls_idx, x1, y1, x2, y2) in self.current_ann:
            sx1 = int(x1 * self.scale) + self.image_x_offset
            sy1 = int(y1 * self.scale) + self.image_y_offset
            sx2 = int(x2 * self.scale) + self.image_x_offset
            sy2 = int(y2 * self.scale) + self.image_y_offset
            if cls_idx < len(self.config["object_classes"]):
                cls_name = self.config["object_classes"][cls_idx]
                color = colors[cls_idx % len(colors)]
            else:
                cls_name = f"類別{cls_idx}"
                color = "gray"
            rid = self.annot_canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=2)
            rid2 = self.annot_canvas.create_text(sx1+5, sy1+10, anchor="w", text=cls_name, fill="yellow", font=("Arial", 12, "bold"))
            self.current_boxes.extend([rid, rid2])
        self.update_annotation_statistics()

    def start_draw(self, event):
        self.drawing = True; self.start_x, self.start_y = event.x, event.y
        self.temp_rect = self.annot_canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="lime", width=2)

    def draw_rectangle(self, event):
        if not self.drawing: return
        self.annot_canvas.coords(self.temp_rect, self.start_x, self.start_y, event.x, event.y)

    def finish_draw(self, event):
        if not self.drawing: return
        self.drawing = False
        x1 = min(self.start_x, event.x) - self.image_x_offset; y1 = min(self.start_y, event.y) - self.image_y_offset
        x2 = max(self.start_x, event.x) - self.image_x_offset; y2 = max(self.start_y, event.y) - self.image_y_offset
        if self.scale <= 0: self.annot_canvas.delete(self.temp_rect); return
        x1 = int(x1 / self.scale); y1 = int(y1 / self.scale); x2 = int(x2 / self.scale); y2 = int(y2 / self.scale)
        if self.current_image is None: self.annot_canvas.delete(self.temp_rect); return
        h, w = self.current_image.shape[:2]
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1)); y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
        if x2-x1 < 2 or y2-y1 < 2: self.annot_canvas.delete(self.temp_rect); return
        cls_name = self.class_var.get()
        if cls_name not in self.config["object_classes"]: self.config["object_classes"].append(cls_name)
        cls_idx = self.config["object_classes"].index(cls_name)
        self.current_ann.append((cls_idx, x1, y1, x2, y2))
        self.annot_canvas.delete(self.temp_rect); self.redraw_annotations()

    def delete_last_annotation(self, event=None):
        if self.current_ann: self.current_ann.pop(); self.redraw_annotations()

    def prev_image(self):
        if not self.image_list: return
        self.current_index = max(0, self.current_index - 1); self.load_current_image()

    def next_image(self):
        if not self.image_list: return
        self.current_index = min(len(self.image_list) - 1, self.current_index + 1); self.load_current_image()

    def update_annotation_statistics(self):
        if not hasattr(self, 'total_images_label'):
            return
        total_images = len(self.image_list)
        annotated_count = 0
        unannotated_count = 0
        class_counts = {}
        for cls in self.config["object_classes"]:
            class_counts[cls] = 0
        if self.image_list:
            for img_path in self.image_list:
                img_path_obj = Path(img_path)
                txt_path = (self.ann_dir / img_path_obj.stem).with_suffix(".txt")
                json_path = (self.ann_dir / img_path_obj.stem).with_suffix(".json")
                has_annotation = False
                if txt_path.exists():
                    try:
                        content = txt_path.read_text(encoding="utf-8").strip()
                        if content:
                            has_annotation = True
                            for line in content.splitlines():
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    try:
                                        cls_idx = int(float(parts[0]))
                                        if cls_idx < len(self.config["object_classes"]):
                                            cls_name = self.config["object_classes"][cls_idx]
                                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                                    except:
                                        pass
                    except:
                        pass
                elif json_path.exists():
                    try:
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        if isinstance(data, dict) and "shapes" in data and data["shapes"]:
                            has_annotation = True
                            for shape in data["shapes"]:
                                label = shape.get("label", "")
                                if label in class_counts:
                                    class_counts[label] += 1
                    except:
                        pass
                if has_annotation:
                    annotated_count += 1
                else:
                    unannotated_count += 1
        self.total_images_label.config(text=f"總影像數量: {total_images}")
        self.annotated_images_label.config(text=f"已標註影像: {annotated_count}")
        self.unannotated_images_label.config(text=f"未標註影像: {unannotated_count}")
        current_ann_count = len(self.current_ann) if hasattr(self, 'current_ann') else 0
        self.current_annotations_label.config(text=f"標註數量: {current_ann_count}")
        for widget in self.class_stats_frame.winfo_children():
            widget.destroy()
        colors = ["green", "red", "blue", "orange", "purple", "brown", "pink", "gray"]
        for i, (cls_name, count) in enumerate(class_counts.items()):
            if count > 0:
                color = colors[i % len(colors)]
                label = ttk.Label(self.class_stats_frame, text=f"{cls_name}: {count}", foreground=color)
                label.pack(anchor="w", pady=1)
        if total_images > 0:
            progress_percent = (annotated_count / total_images) * 100
            self.annotation_progress["maximum"] = 100
            self.annotation_progress["value"] = progress_percent
            self.progress_label.config(text=f"{progress_percent:.1f}% ({annotated_count}/{total_images})")
        else:
            self.annotation_progress["value"] = 0
            self.progress_label.config(text="0%")

    def save_annotations(self):
        if self.current_image_path is None:
            messagebox.showwarning("提示", "尚未載入影像"); return
        img_path = Path(self.current_image_path)
        txt_path = (self.ann_dir / img_path.stem).with_suffix(".txt")
        h, w = self.current_image.shape[:2]; lines = []
        for (cls_idx, x1, y1, x2, y2) in self.current_ann:
            cx = (x1 + x2) / 2.0 / w; cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w; bh = (y2 - y1) / h
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        json_path = (self.ann_dir / img_path.stem).with_suffix(".json")
        json_data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": str(img_path.name),
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }
        for (cls_idx, x1, y1, x2, y2) in self.current_ann:
            cls_name = self.config["object_classes"][cls_idx] if cls_idx < len(self.config["object_classes"]) else str(cls_idx)
            shape = {
                "label": cls_name,
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            json_data["shapes"].append(shape)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"已儲存標註：{txt_path.name} 及 {json_path.name}")
        self.update_annotation_statistics()
        self.show_save_success_dialog()

    def show_save_success_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("標註儲存完成")
        dialog.geometry("600x500")
        dialog.grab_set()
        dialog.transient(self.root)
        x = (dialog.winfo_screenwidth() // 2) - 200
        y = (dialog.winfo_screenheight() // 2) - 150
        dialog.geometry(f"400x300+{x}+{y}")
        header_frame = ttk.Frame(dialog)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        title_label = ttk.Label(header_frame, text="✓ 標註儲存成功！", 
                               font=("Arial", 14, "bold"), foreground="green")
        title_label.pack()
        current_frame = ttk.LabelFrame(dialog, text="當前影像")
        current_frame.pack(fill=tk.X, padx=20, pady=10)
        filename = self.current_image_path.name if self.current_image_path else "未知"
        current_ann_count = len(self.current_ann)
        progress = f"{self.current_index + 1}/{len(self.image_list)}" if self.image_list else "0/0"
        ttk.Label(current_frame, text=f"檔案名稱: {filename}").pack(anchor="w", padx=10, pady=2)
        ttk.Label(current_frame, text=f"標註數量: {current_ann_count}").pack(anchor="w", padx=10, pady=2)
        ttk.Label(current_frame, text=f"進度: {progress}").pack(anchor="w", padx=10, pady=2)
        total_frame = ttk.LabelFrame(dialog, text="整體統計")
        total_frame.pack(fill=tk.X, padx=20, pady=10)
        total_images = len(self.image_list)
        annotated_count = 0
        if self.image_list:
            for img_path in self.image_list:
                img_path_obj = Path(img_path)
                txt_path = (self.ann_dir / img_path_obj.stem).with_suffix(".txt")
                json_path = (self.ann_dir / img_path_obj.stem).with_suffix(".json")
                if txt_path.exists():
                    try:
                        content = txt_path.read_text(encoding="utf-8").strip()
                        if content:
                            annotated_count += 1
                    except:
                        pass
                elif json_path.exists():
                    try:
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        if isinstance(data, dict) and "shapes" in data and data["shapes"]:
                            annotated_count += 1
                    except:
                        pass
        completion_rate = (annotated_count / total_images * 100) if total_images > 0 else 0
        ttk.Label(total_frame, text=f"總影像數量: {total_images}").pack(anchor="w", padx=10, pady=2)
        ttk.Label(total_frame, text=f"已完成標註: {annotated_count}").pack(anchor="w", padx=10, pady=2)
        ttk.Label(total_frame, text=f"剩餘待標註: {total_images - annotated_count}").pack(anchor="w", padx=10, pady=2)
        ttk.Label(total_frame, text=f"完成率: {completion_rate:.1f}%", 
                 foreground="green" if completion_rate > 80 else "orange" if completion_rate > 50 else "red").pack(anchor="w", padx=10, pady=2)
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        ttk.Button(button_frame, text="繼續標註", command=lambda: [dialog.destroy(), self.next_image()]).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="確定", command=dialog.destroy).pack(side=tk.RIGHT, padx=10)
        dialog.after(1000, dialog.destroy)

    def clear_annotations(self):
        self.current_ann = []
        self.redraw_annotations()
        self.update_annotation_statistics()

    def load_existing_annotations(self):
        self.current_ann = []
        if self.current_image_path is None: return
        img_stem = Path(self.current_image_path).stem
        txt_path = (self.ann_dir / img_stem).with_suffix(".txt")
        json_path = (self.ann_dir / img_stem).with_suffix(".json")
        loaded_format = None
        if txt_path.exists():
            try: 
                txt_content = txt_path.read_text(encoding="utf-8").strip()
                if txt_content:
                    h, w = self.current_image.shape[:2]
                    for line_num, line in enumerate(txt_content.splitlines(), 1):
                        parts = line.strip().split()
                        if len(parts) != 5: 
                            print(f"警告：{txt_path.name} 第{line_num}行格式錯誤，跳過")
                            continue
                        try:
                            cls_idx = int(float(parts[0]))
                            cx, cy, bw, bh = map(float, parts[1:])
                            if cls_idx >= len(self.config["object_classes"]):
                                print(f"警告：{txt_path.name} 第{line_num}行類別索引 {cls_idx} 超出範圍，跳過")
                                continue
                            x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
                            x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
                            self.current_ann.append((cls_idx, x1, y1, x2, y2))
                        except (ValueError, IndexError) as e:
                            print(f"警告：{txt_path.name} 第{line_num}行數據錯誤，跳過：{e}")
                            continue
                    loaded_format = "TXT (YOLO)"
                    print(f"已載入 TXT 標註：{len(self.current_ann)} 個標註框")
            except Exception as e:
                print(f"讀取 TXT 檔案失敗：{e}")
        if not self.current_ann and json_path.exists():
            try:
                json_content = json_path.read_text(encoding="utf-8")
                data = json.loads(json_content)
                if isinstance(data, dict) and "shapes" in data:
                    for shape in data["shapes"]:
                        if shape.get("shape_type") == "rectangle" and len(shape.get("points", [])) >= 2:
                            label = shape.get("label", "unknown")
                            if label not in self.config["object_classes"]:
                                self.config["object_classes"].append(label)
                                if hasattr(self, 'class_box'):
                                    self.class_box['values'] = self.config["object_classes"]
                                print(f"自動添加新類別：{label}")
                            cls_idx = self.config["object_classes"].index(label)
                            pts = shape["points"]
                            x1, y1 = int(float(pts[0][0])), int(float(pts[0][1]))
                            x2, y2 = int(float(pts[1][0])), int(float(pts[1][1]))
                            self.current_ann.append((cls_idx, min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)))
                    loaded_format = "JSON (LabelMe)"
                elif isinstance(data, dict) and "objects" in data:
                    for obj in data["objects"]:
                        bbox = obj.get("bbox", {})
                        if all(k in bbox for k in ("x", "y", "w", "h")):
                            label = obj.get("class", obj.get("label", "unknown"))
                            if label not in self.config["object_classes"]:
                                self.config["object_classes"].append(label)
                                if hasattr(self, 'class_box'):
                                    self.class_box['values'] = self.config["object_classes"]
                                print(f"自動添加新類別：{label}")
                            cls_idx = self.config["object_classes"].index(label)
                            x, y, w, h = float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
                            x1, y1 = int(x), int(y)
                            x2, y2 = int(x + w), int(y + h)
                            self.current_ann.append((cls_idx, x1, y1, x2, y2))
                    loaded_format = "JSON (Objects)"
                elif isinstance(data, dict) and "bboxes" in data:
                    for bbox in data["bboxes"]:
                        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                            label = bbox.get("label", bbox.get("class", "unknown"))
                            if label not in self.config["object_classes"]:
                                self.config["object_classes"].append(label)
                                if hasattr(self, 'class_box'):
                                    self.class_box['values'] = self.config["object_classes"]
                                print(f"自動添加新類別：{label}")
                            cls_idx = self.config["object_classes"].index(label)
                            x1 = int(float(bbox["x1"]))
                            y1 = int(float(bbox["y1"]))
                            x2 = int(float(bbox["x2"]))
                            y2 = int(float(bbox["y2"]))
                            self.current_ann.append((cls_idx, x1, y1, x2, y2))
                    loaded_format = "JSON (Bboxes)"
                if loaded_format:
                    print(f"已載入 {loaded_format} 標註：{len(self.current_ann)} 個標註框")
                else:
                    print(f"警告：無法識別 JSON 格式：{json_path.name}")
            except json.JSONDecodeError as e:
                print(f"JSON 格式錯誤：{json_path.name} - {e}")
            except Exception as e:
                print(f"讀取 JSON 檔案失敗：{e}")
        if txt_path.exists() and json_path.exists():
            txt_has_content = False
            json_has_content = False
            try:
                txt_has_content = bool(txt_path.read_text(encoding="utf-8").strip())
            except:
                pass
            try:
                json_content = json_path.read_text(encoding="utf-8")
                json_data = json.loads(json_content)
                json_has_content = bool(json_data)
            except:
                pass
            if txt_has_content and json_has_content and loaded_format:
                print(f"註：同時存在 TXT 和 JSON 標註檔，已載入 {loaded_format} 格式")
        if not self.current_ann and not loaded_format:
            print(f"未找到有效的標註檔案：{img_stem}.txt 或 {img_stem}.json")
# === 訓練頁 ===
    def build_train_tab(self, parent):
        top = ttk.Frame(parent); top.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(top, text="掃描標註", command=self.scan_annotation_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="準備資料集", command=self.prepare_training_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="診斷資料集", command=self.diagnose_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="開始訓練", command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="停止訓練", command=self.stop_training).pack(side=tk.LEFT, padx=5)
        ttk.Label(top, text="模型：").pack(side=tk.LEFT, padx=10)
        ttk.Button(top, text="載入模型", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="使用最新訓練模型", command=self.load_latest_model).pack(side=tk.LEFT, padx=5)
        param_frame = ttk.LabelFrame(parent, text="訓練參數設定")
        param_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(param_frame, text="訓練輪數:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.epochs_var = tk.IntVar(value=self.config["training_settings"]["default_epochs"])
        ttk.Spinbox(param_frame, from_=10, to=300, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(param_frame, text="影像大小:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.imgsize_var = tk.IntVar(value=self.config["training_settings"]["default_img_size"])
        ttk.Combobox(param_frame, values=[320, 480, 640, 800], textvariable=self.imgsize_var, width=10, state="readonly").grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(param_frame, text="批次大小:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.batch_var = tk.IntVar(value=self.config["training_settings"]["default_batch_size"])
        ttk.Spinbox(param_frame, from_=4, to=32, textvariable=self.batch_var, width=10).grid(row=0, column=5, padx=5, pady=5)
        detect_frame = ttk.Frame(parent); detect_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(detect_frame, text="信心閾值:").pack(side=tk.LEFT, padx=5)
        self.confidence_var = tk.DoubleVar(value=self.config["detection_settings"]["confidence_threshold"])
        self.confidence_label = ttk.Label(detect_frame, text=f"{float(self.confidence_var.get()):.2f}")
        scale = ttk.Scale(detect_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL, variable=self.confidence_var, length=200,
                          command=lambda v: self.update_confidence_label())
        scale.pack(side=tk.LEFT, padx=5); self.confidence_label.pack(side=tk.LEFT, padx=5)
        progress_frame = ttk.LabelFrame(parent, text="訓練進度")
        progress_frame.pack(fill=tk.X, padx=10, pady=10)
        pb_container = ttk.Frame(progress_frame)
        pb_container.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(pb_container, text="整體進度:").pack(anchor="w", pady=2)
        self.progress = ttk.Progressbar(pb_container, mode="determinate", maximum=100, value=0, length=500)
        self.progress.pack(fill=tk.X, pady=5)
        progress_info = ttk.Frame(pb_container)
        progress_info.pack(fill=tk.X, pady=5)
        self.epoch_label = ttk.Label(progress_info, text="等待訓練開始...", font=("Arial", 10))
        self.epoch_label.pack(side=tk.LEFT)
        self.progress_percent_label = ttk.Label(progress_info, text="0%", font=("Arial", 10, "bold"), foreground="blue")
        self.progress_percent_label.pack(side=tk.RIGHT)
        log_frame = ttk.LabelFrame(parent, text="訓練日誌")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def update_confidence_label(self):
        try: self.confidence_label.config(text=f"{float(self.confidence_var.get()):.2f}")
        except Exception: pass

    def scan_annotation_data(self):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        imgs = []
        for p in self.images_dir.glob("**/*"):
            if p.suffix.lower() in exts: imgs.append(p)
        txt_cnt = len(list(self.ann_dir.glob("*.txt")))
        json_cnt = len(list(self.ann_dir.glob("*.json")))
        self.append_log(f"影像數量：{len(imgs)}，TXT 標註檔數量：{txt_cnt}，JSON 標註檔數量：{json_cnt}")

    def diagnose_dataset(self):
        """診斷資料集狀態"""
        self.append_log("=" * 60)
        self.append_log("開始診斷資料集...")
        img_files = list(self.images_dir.glob("*.[jp][pn][g]*"))
        txt_files = list(self.ann_dir.glob("*.txt"))
        json_files = list(self.ann_dir.glob("*.json"))
        self.append_log(f"\n【原始資料】")
        self.append_log(f"  images/ : {len(img_files)} 張影像")
        self.append_log(f"  annotations/ : {len(txt_files)} 個 .txt, {len(json_files)} 個 .json")
        self.append_log(f"\n【YOLO 資料集】")
        if not self.dataset_root.exists():
            self.append_log(f"  ✗ dataset/ 資料夾不存在 - 請執行「準備資料集」")
            self.append_log("=" * 60)
            return
        train_imgs = list((self.dataset_root / "train/images").glob("*")) if (self.dataset_root / "train/images").exists() else []
        train_labels = list((self.dataset_root / "train/labels").glob("*.txt")) if (self.dataset_root / "train/labels").exists() else []
        val_imgs = list((self.dataset_root / "val/images").glob("*")) if (self.dataset_root / "val/images").exists() else []
        val_labels = list((self.dataset_root / "val/labels").glob("*.txt")) if (self.dataset_root / "val/labels").exists() else []
        self.append_log(f"  train/images : {len(train_imgs)} 張")
        self.append_log(f"  train/labels : {len(train_labels)} 個")
        self.append_log(f"  val/images : {len(val_imgs)} 張")
        self.append_log(f"  val/labels : {len(val_labels)} 個")
        yaml_path = self.dataset_root / "dataset.yaml"
        if yaml_path.exists():
            try:
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                self.append_log(f"\n【dataset.yaml】")
                self.append_log(f"  ✓ 檔案存在")
                self.append_log(f"  nc: {data.get('nc')}")
                self.append_log(f"  names: {data.get('names')}")
            except Exception as e:
                self.append_log(f"\n【dataset.yaml】")
                self.append_log(f"  ✗ YAML 格式錯誤: {e}")
        else:
            self.append_log(f"\n【dataset.yaml】")
            self.append_log(f"  ✗ 檔案不存在")
        self.append_log(f"\n【診斷結果】")
        if len(train_imgs) > 0 and len(train_labels) > 0:
            self.append_log(f"  ✓ 資料集準備完成，可以開始訓練")
        else:
            self.append_log(f"  ✗ 資料集未準備好，請執行以下步驟:")
            if len(img_files) == 0:
                self.append_log(f"     1. 拍攝/複製影像到 images/ 資料夾")
            if len(txt_files) == 0 and len(json_files) == 0:
                self.append_log(f"     2. 在「標註作業」頁籤標註影像")
            self.append_log(f"     3. 點擊「準備資料集」")
        self.append_log("=" * 60)

    def _parse_json_to_yolo_lines(self, json_path: Path, img_w: int, img_h: int):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return [], "讀檔失敗"
        names = self.config["object_classes"]
        def cls_index(label):
            if label in names: return names.index(label)
            try:
                i = int(label)
                if 0 <= i < len(names): return i
            except Exception:
                pass
            return None
        lines = []
        used = False
        if isinstance(data, dict) and "shapes" in data:
            for s in data.get("shapes", []):
                if s.get("shape_type") == "rectangle" and "points" in s:
                    pts = s["points"]
                    if len(pts) >= 2:
                        (x1, y1), (x2, y2) = pts[0], pts[1]
                        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                        label = s.get("label", "0")
                        ci = cls_index(label)
                        if ci is None: continue
                        cx = ((x1 + x2) / 2.0) / img_w
                        cy = ((y1 + y2) / 2.0) / img_h
                        bw = abs(x2 - x1) / img_w
                        bh = abs(y2 - y1) / img_h
                        lines.append(f"{ci} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        used = True
        if not used and isinstance(data, dict) and "objects" in data:
            for obj in data.get("objects", []):
                bbox = obj.get("bbox", {})
                if all(k in bbox for k in ("x", "y", "w", "h")):
                    x, y, w, h = float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
                    label = obj.get("class", "0")
                    ci = cls_index(label)
                    if ci is None: continue
                    cx = (x + w/2.0) / img_w
                    cy = (y + h/2.0) / img_h
                    bw = w / img_w
                    bh = h / img_h
                    lines.append(f"{ci} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    used = True
        if not used and isinstance(data, dict) and "bboxes" in data:
            for b in data.get("bboxes", []):
                if all(k in b for k in ("x1", "y1", "x2", "y2")):
                    x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
                    label = b.get("label", "0")
                    ci = cls_index(label)
                    if ci is None: continue
                    cx = ((x1 + x2) / 2.0) / img_w
                    cy = ((y1 + y2) / 2.0) / img_h
                    bw = abs(x2 - x1) / img_w
                    bh = abs(y2 - y1) / img_h
                    lines.append(f"{ci} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    used = True
        return lines, (None if used else "無法識別 JSON 結構")

    def prepare_training_dataset(self):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        all_imgs = []
        for p in self.images_dir.glob("**/*"):
            if p.suffix.lower() in exts: all_imgs.append(p)
        if not all_imgs:
            messagebox.showerror("錯誤", "images/ 內沒有影像")
            return
        created_empty = 0
        converted_from_json = 0
        json_unparsed = 0
        invalid_lines = 0
        for img in all_imgs:
            txt = (self.ann_dir / img.stem).with_suffix(".txt")
            if not txt.exists():
                j = (self.ann_dir / img.stem).with_suffix(".json")
                if j.exists():
                    if OPENCV_AVAILABLE:
                        im = cv2.imread(str(img))
                        if im is None:
                            txt.parent.mkdir(parents=True, exist_ok=True)
                            txt.write_text("", encoding="utf-8"); created_empty += 1; continue
                        h, w = im.shape[:2]
                    else:
                        h = w = None
                    if h is not None and w is not None:
                        lines, err = self._parse_json_to_yolo_lines(j, w, h)
                        txt.parent.mkdir(parents=True, exist_ok=True)
                        if lines:
                            txt.write_text("\n".join(lines), encoding="utf-8"); converted_from_json += 1
                        else:
                            txt.write_text("", encoding="utf-8"); json_unparsed += 1
                    else:
                        txt.parent.mkdir(parents=True, exist_ok=True)
                        txt.write_text("", encoding="utf-8"); created_empty += 1
                else:
                    txt.parent.mkdir(parents=True, exist_ok=True)
                    txt.write_text("", encoding="utf-8"); created_empty += 1
        cls_cnt = len(self.config["object_classes"])
        for txt in sorted(self.ann_dir.glob("*.txt")):
            try:
                raw = txt.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            new_lines = []
            for line in raw:
                parts = line.strip().split()
                if len(parts) != 5: invalid_lines += 1; continue
                try:
                    c = int(float(parts[0]))
                    if c < 0 or c >= cls_cnt: invalid_lines += 1; continue
                    new_lines.append(f"{c} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                except Exception:
                    invalid_lines += 1
            txt.write_text("\n".join(new_lines), encoding="utf-8")
        if self.dataset_root.exists(): shutil.rmtree(self.dataset_root)
        (self.dataset_root / "train/images").mkdir(parents=True, exist_ok=True)
        (self.dataset_root / "train/labels").mkdir(parents=True, exist_ok=True)
        (self.dataset_root / "val/images").mkdir(parents=True, exist_ok=True)
        (self.dataset_root / "val/labels").mkdir(parents=True, exist_ok=True)
        txts = sorted(self.ann_dir.glob("*.txt"))
        ratio = self.config["training_settings"]["data_split_ratio"]
        random.shuffle(txts); split_idx = int(len(txts) * ratio)
        train_txts = txts[:split_idx]; val_txts = txts[split_idx:]
        def copy_pair(txt_path, subset):
            img_candidates = []
            for ext in exts:
                c = self.images_dir / (txt_path.stem + ext)
                if c.exists(): img_candidates.append(c)
            if not img_candidates: return
            img = img_candidates[0]
            dst_img = self.dataset_root / subset / "images" / img.name
            dst_lbl = self.dataset_root / subset / "labels" / txt_path.name
            shutil.copy2(img, dst_img); shutil.copy2(txt_path, dst_lbl)
        for t in train_txts: copy_pair(t, "train")
        for t in val_txts:   copy_pair(t, "val")
        data = {
            "path": str(self.dataset_root.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(self.config["object_classes"]),
            "names": list(self.config["object_classes"])
        }
        (self.dataset_root / "dataset.yaml").write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        n_train_imgs = len(list((self.dataset_root / "train/images").glob("*")))
        n_val_imgs = len(list((self.dataset_root / "val/images").glob("*")))
        n_train_labels = len(list((self.dataset_root / "train/labels").glob("*.txt")))
        n_val_labels = len(list((self.dataset_root / "val/labels").glob("*.txt")))
        if n_train_imgs == 0:
            messagebox.showerror("錯誤", "訓練集影像為空！請確保 images/ 資料夾有檔案")
            return
        if n_train_labels == 0:
            messagebox.showerror("錯誤", "訓練集標註為空！請先在「標註作業」標註影像")
            return
        yaml_path = self.dataset_root / "dataset.yaml"
        if yaml_path.exists():
            try:
                yaml_content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                self.append_log(f"✓ dataset.yaml 檢查通過")
                self.append_log(f"  類別數量: {yaml_content.get('nc', 0)}")
                self.append_log(f"  類別名稱: {yaml_content.get('names', [])}")
            except Exception as e:
                messagebox.showerror("錯誤", f"dataset.yaml 格式錯誤: {e}")
                return
        self.append_log(f"資料集已準備完成（dataset/）")
        self.append_log(f"✓ Train 影像: {n_train_imgs}, 標註: {n_train_labels}")
        self.append_log(f"✓ Val 影像: {n_val_imgs}, 標註: {n_val_labels}")
        self.append_log(f"JSON→TXT 轉換：{converted_from_json}，無法解析 JSON：{json_unparsed}，新建空白標註：{created_empty}，跳過異常標註行：{invalid_lines}")
        self.append_log("=" * 50)
        self.append_log("資料集準備完成！可以開始訓練")
        messagebox.showinfo("完成", 
            f"資料集準備完成！\n\n"
            f"Train: {n_train_imgs} 張影像\n"
            f"Val: {n_val_imgs} 張影像\n\n"
            f"請繼續「開始訓練」")

    def _set_progress(self, value: int, epoch_text: str = ""):
        if not self.use_gui: return
        def _apply():
            progress_value = max(0, min(100, int(value)))
            self.progress["value"] = progress_value
            self.progress_percent_label.config(text=f"{progress_value}%")
            if epoch_text:
                self.epoch_label.config(text=epoch_text)
            elif progress_value == 0:
                self.epoch_label.config(text="等待訓練開始...")
            elif progress_value == 100:
                self.epoch_label.config(text="訓練完成！")
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    def _register_train_callbacks(self, total_epochs: int) -> None:
        """
        以 ultralytics callback 回報逐 epoch 進度，並讓停止旗標真正生效。

        原本 train_stop_flag 只有 set()、沒有任何地方檢查，按「停止訓練」不會有作用；
        進度也只在 0/10/100 更新，並非逐 epoch。

        callback 內一律吞例外：回呼失敗不可影響訓練本身（最差退回原本行為）。
        """
        def _on_epoch_end(trainer):
            try:
                epoch = int(getattr(trainer, "epoch", 0)) + 1
                pct = 10 + int(85 * epoch / max(1, total_epochs))
                self._set_progress(pct, f"EPOCH {epoch}/{total_epochs}")
                if self.train_stop_flag.is_set():
                    self.append_log("收到停止要求，將在本 epoch 結束後停止訓練")
                    # ultralytics 的早停旗標（不同版本命名略有差異，兩者都設）
                    trainer.stop = True
                    trainer.stop_training = True
            except Exception:
                pass

        try:
            self.yolo_model.add_callback("on_train_epoch_end", _on_epoch_end)
        except Exception as e:
            self.append_log(f"註冊訓練回呼失敗（進度/停止功能可能不可用）：{e}")

    def _train_worker(self, model_name="yolov8n.pt", epochs=100, imgsz=640, batch=16):
        try:
            self._set_progress(0, "正在初始化...")
            self.append_log("開始訓練...")
            if not YOLO_AVAILABLE:
                self.append_log("未安裝 ultralytics，無法訓練。"); return
            data_yaml = self.dataset_root / "dataset.yaml"
            if not data_yaml.exists():
                self.append_log("找不到 dataset/dataset.yaml，請先準備資料集"); return
            train_img_dir = self.dataset_root / "train" / "images"
            train_label_dir = self.dataset_root / "train" / "labels"
            if not train_img_dir.exists():
                self.append_log("錯誤: train/images 資料夾不存在！")
                self._ui_message("error", "錯誤", "請先執行「準備資料集」")
                return
            train_images = list(train_img_dir.glob("*.[jp][pn][g]*"))
            train_labels = list(train_label_dir.glob("*.txt"))
            if len(train_images) == 0:
                self.append_log("錯誤: 訓練集沒有影像！")
                self.append_log(f"檢查路徑: {train_img_dir}")
                self._ui_message("error", "錯誤",
                    "訓練集為空！\n\n"
                    "請確認:\n"
                    "1. images/ 資料夾有影像檔\n"
                    "2. annotations/ 資料夾有 .txt 標註檔\n"
                    "3. 已執行「準備資料集」")
                return
            if len(train_labels) == 0:
                self.append_log("錯誤: 訓練集沒有標註檔！")
                self._ui_message("error", "錯誤", "請先在「標註作業」標註影像")
                return
            self.append_log(f"✓ 發現 {len(train_images)} 張訓練影像")
            self.append_log(f"✓ 發現 {len(train_labels)} 個標註檔")
            try:
                yaml_data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
                self.append_log(f"✓ dataset.yaml 載入成功")
                self.append_log(f"  nc={yaml_data.get('nc')}, names={yaml_data.get('names')}")
            except Exception as e:
                self.append_log(f"錯誤: dataset.yaml 格式錯誤 - {e}")
                return
            device = 'cpu'
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                    self.append_log(f"檢測到 GPU，將使用 CUDA 加速訓練")
                else:
                    self.append_log(f"未檢測到 GPU，將使用 CPU 進行訓練（速度較慢）")
            except ImportError:
                self.append_log("未安裝 PyTorch，使用 CPU 模式")
            self.append_log(f"載入預訓練模型：{model_name}")
            self.yolo_model = YOLO(model_name)
            # 註冊 epoch 回呼：逐 epoch 回報進度，並讓「停止訓練」真正生效
            self._register_train_callbacks(epochs)
            self.append_log(f"開始訓練 - 資料集：{data_yaml}")
            self.append_log(f"訓練參數 - Epochs: {epochs}, 影像大小: {imgsz}, 批次大小: {batch}")
            self.append_log(f"使用設備：{device.upper()}")
            if device == 'cpu' and batch > 8:
                batch = 8
                self.append_log(f"CPU模式下調整批次大小為：{batch}")
            self._set_progress(10, "訓練進行中...")
            results = self.yolo_model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                verbose=True,
                patience=50,
                save=True,
                plots=True,
                device=device
            )
            self._set_progress(100, f"EPOCH {epochs}/{epochs}")
            best = None
            if hasattr(results, "save_dir"):
                save_dir = Path(results.save_dir)
                self.append_log(f"訓練結果保存於：{save_dir}")
                best_candidates = [
                    save_dir / "weights" / "best.pt",
                    save_dir / "best.pt"
                ]
                for candidate in best_candidates:
                    if candidate.exists():
                        best = candidate
                        break
            if best is None and hasattr(results, "save_dir"):
                last_path = Path(results.save_dir) / "weights" / "last.pt"
                if last_path.exists():
                    best = last_path
                    self.append_log(f"使用 last.pt 作為訓練結果")
            if best:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                out = self.models_dir / f"best_{timestamp}.pt"
                shutil.copy2(best, out)
                self.current_model_path = str(out)
                self.append_log(f"訓練完成！模型已保存至：{out}")
                try:
                    self.yolo_model = YOLO(str(out))
                    self.append_log("已自動載入訓練完成的模型")
                except Exception as e:
                    self.append_log(f"載入新模型時出現警告：{e}")
            else:
                self.append_log("警告：訓練完成但無法找到模型文件")
        except Exception as e:
            self.append_log(f"訓練過程發生錯誤：{str(e)}")
            import traceback
            self.append_log(f"詳細錯誤：{traceback.format_exc()}")
        finally:
            self._set_progress(100, "訓練完成")
            self.train_thread = None

    def start_training(self):
        if self.train_thread is not None:
            messagebox.showinfo("提示", "訓練已在進行中"); return
        if not YOLO_AVAILABLE:
            messagebox.showerror("錯誤", "未安裝 ultralytics，無法訓練"); return
        epochs = self.epochs_var.get()
        imgsz = self.imgsize_var.get()
        batch = self.batch_var.get()
        self._set_progress(0, "正在初始化訓練...")
        self.train_stop_flag.clear()
        self.train_thread = threading.Thread(target=self._train_worker, args=("yolov8n.pt", epochs, imgsz, batch), daemon=True)
        self.train_thread.start()

    def stop_training(self):
        if self.train_thread is None:
            messagebox.showinfo("提示", "目前沒有進行中的訓練"); return
        self.train_stop_flag.set()
        self.append_log("已送出停止要求，將在目前這個 epoch 結束後停止訓練。")

    def load_model(self):
        p = filedialog.askopenfilename(title="選擇 YOLO 權重檔", filetypes=[("YOLO PT", "*.pt"), ("All Files", "*.*")], initialdir=str(self.models_dir))
        if not p: return
        self.current_model_path = p
        try:
            if YOLO_AVAILABLE: self.yolo_model = YOLO(self.current_model_path)
            self.append_log(f"已載入模型：{self.current_model_path}")
        except Exception as e:
            messagebox.showerror("錯誤", f"載入模型失敗：{e}")

    def load_latest_model(self):
        pts = sorted(self.models_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not pts:
            messagebox.showwarning("提示", "models/ 無 .pt 檔案"); return
        self.current_model_path = str(pts[0])
        try:
            if YOLO_AVAILABLE: self.yolo_model = YOLO(self.current_model_path)
            self.append_log(f"已載入最新模型：{self.current_model_path}")
        except Exception as e:
            messagebox.showerror("錯誤", f"載入最新模型失敗：{e}")

    def detect_current_image(self):
        if self.current_image is None or not OPENCV_AVAILABLE:
            messagebox.showwarning("提示", "請先在「標註」頁載入影像"); return
        if not YOLO_AVAILABLE or self.yolo_model is None:
            messagebox.showwarning("提示", "尚未載入 YOLO 模型"); return
        conf = float(self.confidence_var.get()); img = self.current_image.copy()
        res = self.yolo_model.predict(source=img, conf=conf, verbose=False)
        if res and len(res) > 0:
            r0 = res[0]
            if hasattr(r0, "plot"):
                plot = r0.plot(); self.current_image = plot[..., ::-1]; self.display_annotation_image()
                self.append_log(f"偵測結果：共 {len(r0.boxes)} 個")
        else:
            self.append_log("無偵測結果")

    def _ensure_detector_thread(self, index: int):
        if not YOLO_AVAILABLE or self.yolo_model is None: return
        if index in self.rt_det_threads and self.rt_det_threads[index].is_alive(): return
        stop_ev = threading.Event()
        self.rt_det_flags[index] = stop_ev
        t = threading.Thread(target=self._detector_worker, args=(index, stop_ev), daemon=True)
        self.rt_det_threads[index] = t
        t.start()

    def _stop_detector_thread(self, index: int):
        ev = self.rt_det_flags.pop(index, None)
        if ev: ev.set()
        t = self.rt_det_threads.pop(index, None)
        if t and t.is_alive(): t.join(timeout=1.0)
        self.rt_frames.pop(index, None)

    def _detector_worker(self, index: int, stop_event: threading.Event):
        conf = float(self.confidence_var.get())
        while not stop_event.is_set():
            f = None
            q = self.camera_queues.get(index)
            if q is not None:
                try:
                    while q.qsize() > 1: q.get_nowait()
                    f = q.get_nowait()
                except Exception:
                    pass
            if f is None: f = self.camera_frames.get(index)
            if f is None:
                time.sleep(0.01); continue
            try:
                res = self.yolo_model.predict(source=f, conf=conf, verbose=False)
                if res and len(res) > 0 and hasattr(res[0], "plot"):
                    plot = res[0].plot()
                    self.rt_frames[index] = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
                else:
                    self.rt_frames[index] = f
            except Exception:
                self.rt_frames[index] = f
            time.sleep(0.02)

    # === 訓練結果頁（完整保留，省略部分以節省空間）===
    def build_result_tab(self, parent):
        top = ttk.Frame(parent); top.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(top, text="刷新模型清單", command=self.refresh_model_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="以此模型啟用全攝影機即時辨識", command=self.enable_realtime_detection_from_selection).pack(side=tk.LEFT, padx=10)
        ttk.Button(top, text="停止即時辨識", command=self.disable_realtime_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="批次辨識並分類", command=self.batch_detect_and_sort).pack(side=tk.LEFT, padx=10)
        management_frame = ttk.Frame(parent)
        management_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(management_frame, text="模型管理:").pack(side=tk.LEFT, padx=5)
        ttk.Button(management_frame, text="重新命名", command=self.rename_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(management_frame, text="刪除模型", command=self.delete_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(management_frame, text="複製模型", command=self.copy_model).pack(side=tk.LEFT, padx=5)
        info_frame = ttk.LabelFrame(parent, text="模型資訊")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        self.model_info_label = ttk.Label(info_frame, text="請選擇一個模型查看詳細資訊", justify=tk.LEFT)
        self.model_info_label.pack(anchor="w", padx=10, pady=10)
        self.model_listbox = tk.Listbox(parent, height=10)
        self.model_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.model_listbox.bind('<<ListboxSelect>>', self.on_model_select)
        self.refresh_model_list()

    def refresh_model_list(self):
        self.model_listbox.delete(0, tk.END)
        pts = sorted(self.models_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        for p in pts: 
            mod_time = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            self.model_listbox.insert(tk.END, f"{p.name} ({mod_time})")

    def on_model_select(self, event):
        selection = self.model_listbox.curselection()
        if not selection:
            return
        model_text = self.model_listbox.get(selection[0])
        model_name = model_text.split(" (")[0]
        model_path = self.models_dir / model_name
        if model_path.exists():
            stat = model_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            create_time = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
            info_text = f"""檔案名稱: {model_name}
檔案大小: {size_mb:.2f} MB
建立時間: {create_time}
修改時間: {mod_time}
檔案路徑: {model_path}"""
            self.model_info_label.config(text=info_text)
        else:
            self.model_info_label.config(text="模型檔案不存在")

    def rename_model(self):
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "請先選擇要重新命名的模型")
            return
        model_text = self.model_listbox.get(selection[0])
        old_name = model_text.split(" (")[0]
        old_path = self.models_dir / old_name
        if not old_path.exists():
            messagebox.showerror("錯誤", "模型檔案不存在")
            return
        rename_dialog = tk.Toplevel(self.root)
        rename_dialog.title("重新命名模型")
        rename_dialog.geometry("400x150")
        rename_dialog.grab_set()
        ttk.Label(rename_dialog, text=f"原檔名: {old_name}").pack(pady=10)
        ttk.Label(rename_dialog, text="新檔名:").pack(pady=5)
        new_name_var = tk.StringVar(value=old_name.replace('.pt', ''))
        name_entry = ttk.Entry(rename_dialog, textvariable=new_name_var, width=40)
        name_entry.pack(pady=5)
        name_entry.select_range(0, tk.END)
        name_entry.focus()
        def do_rename():
            new_name = new_name_var.get().strip()
            if not new_name:
                messagebox.showwarning("警告", "請輸入新的檔名")
                return
            if not new_name.endswith('.pt'):
                new_name += '.pt'
            new_path = self.models_dir / new_name
            if new_path.exists():
                messagebox.showerror("錯誤", "檔名已存在")
                return
            try:
                old_path.rename(new_path)
                self.refresh_model_list()
                self.append_log(f"模型重新命名成功: {old_name} -> {new_name}")
                rename_dialog.destroy()
                messagebox.showinfo("成功", f"模型已重新命名為: {new_name}")
            except Exception as e:
                messagebox.showerror("錯誤", f"重新命名失敗: {str(e)}")
        button_frame = ttk.Frame(rename_dialog)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="確定", command=do_rename).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="取消", command=rename_dialog.destroy).pack(side=tk.LEFT, padx=10)
        name_entry.bind('<Return>', lambda e: do_rename())

    def delete_model(self):
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "請先選擇要刪除的模型")
            return
        model_text = self.model_listbox.get(selection[0])
        model_name = model_text.split(" (")[0]
        model_path = self.models_dir / model_name
        if not model_path.exists():
            messagebox.showerror("錯誤", "模型檔案不存在")
            return
        result = messagebox.askyesno("確認刪除", 
                                   f"確定要刪除模型檔案嗎?\n\n檔名: {model_name}\n\n此操作無法復原!")
        if result:
            try:
                model_path.unlink()
                self.refresh_model_list()
                self.model_info_label.config(text="請選擇一個模型查看詳細資訊")
                self.append_log(f"已刪除模型: {model_name}")
                messagebox.showinfo("成功", f"模型 {model_name} 已刪除")
            except Exception as e:
                messagebox.showerror("錯誤", f"刪除失敗: {str(e)}")

    def copy_model(self):
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "請先選擇要複製的模型")
            return
        model_text = self.model_listbox.get(selection[0])
        original_name = model_text.split(" (")[0]
        original_path = self.models_dir / original_name
        if not original_path.exists():
            messagebox.showerror("錯誤", "模型檔案不存在")
            return
        base_name = original_name.replace('.pt', '')
        counter = 1
        while True:
            copy_name = f"{base_name}_copy{counter}.pt"
            copy_path = self.models_dir / copy_name
            if not copy_path.exists():
                break
            counter += 1
        try:
            shutil.copy2(original_path, copy_path)
            self.refresh_model_list()
            self.append_log(f"已複製模型: {original_name} -> {copy_name}")
            messagebox.showinfo("成功", f"模型已複製為: {copy_name}")
        except Exception as e:
            messagebox.showerror("錯誤", f"複製失敗: {str(e)}")

    def enable_realtime_detection_from_selection(self):
        sel = self.model_listbox.curselection()
        if not sel:
            messagebox.showwarning("提示", "請先在清單選取模型"); return
        model_text = self.model_listbox.get(sel[0])
        model_name = model_text.split(" (")[0]
        path = str(self.models_dir / model_name)
        if not Path(path).exists():
            messagebox.showerror("錯誤", "模型檔不存在"); return
        try:
            self.yolo_model = YOLO(path)
            self.current_model_path = path
            self.append_log(f"已載入即時辨識模型：{path}")
        except Exception as e:
            messagebox.showerror("錯誤", f"載入模型失敗：{e}"); return
        if not self.active_cameras:
            messagebox.showinfo("提示", "請先在「攝影機」頁開啟要辨識的攝影機"); return
        self.rt_detect_enabled = True
        for idx in list(self.active_cameras.keys()):
            self._ensure_detector_thread(idx)
        messagebox.showinfo("即時辨識", "已啟用全攝影機即時辨識。")

    def disable_realtime_detection(self):
        self.rt_detect_enabled = False
        for idx in list(self.rt_det_threads.keys()):
            self._stop_detector_thread(idx)
        messagebox.showinfo("即時辨識", "已停止全攝影機即時辨識。")

    def batch_detect_and_sort(self):
        if not self.yolo_model:
            messagebox.showwarning("警告", "請先載入模型")
            return
        folder = filedialog.askdirectory(title="選擇要批次處理的資料夾", initialdir=str(self.images_dir))
        if not folder:
            return
        # 以「載入模型自身的類別名稱」為準，而非硬編的 config["object_classes"]。
        # 否則當模型類別與 config 清單不一致時（例如載入 OK/NG 模型），
        # 會用模型的索引去查 config 清單，把影像分到錯誤的資料夾。
        names = self.yolo_model.names  # {類別索引: 類別名稱}
        for cls in names.values():
            (self.sorted_output_dir / str(cls)).mkdir(exist_ok=True)
        (self.sorted_output_dir / "unknown").mkdir(exist_ok=True)
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        images = [f for f in Path(folder).glob("*") if f.suffix.lower() in exts]
        if not images:
            messagebox.showwarning("提示", "所選資料夾沒有影像")
            return
        progress_window = tk.Toplevel(self.root)
        progress_window.title("批次處理進度")
        progress_window.geometry("400x100")
        progress_bar = ttk.Progressbar(progress_window, length=350, mode='determinate', maximum=len(images))
        progress_bar.pack(padx=20, pady=20)
        progress_label = ttk.Label(progress_window, text="")
        progress_label.pack()
        processed_count = {"total": 0}
        class_counts = {str(cls): 0 for cls in names.values()}
        class_counts["unknown"] = 0
        for i, img_path in enumerate(images):
            progress_label.config(text=f"處理中: {img_path.name} ({i+1}/{len(images)})")
            progress_window.update()
            if OPENCV_AVAILABLE:
                img = cv2.imread(str(img_path))
                if img is not None:
                    results = self.yolo_model.predict(source=img, conf=self.confidence_var.get(), verbose=False)
                    if results and len(results) > 0 and len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        max_conf_idx = boxes.conf.argmax()
                        cls_idx = int(boxes.cls[max_conf_idx])
                        cls_name = str(names.get(cls_idx, "unknown"))
                        if cls_name not in class_counts:
                            cls_name = "unknown"
                        dst = self.sorted_output_dir / cls_name / img_path.name
                        shutil.copy2(img_path, dst)
                        class_counts[cls_name] += 1
                        processed_count["total"] += 1
                    else:
                        dst = self.sorted_output_dir / "unknown" / img_path.name
                        shutil.copy2(img_path, dst)
                        class_counts["unknown"] += 1
                        processed_count["total"] += 1
            progress_bar['value'] = i + 1
        progress_window.destroy()
        result_msg = "批次處理完成！\n\n分類結果：\n"
        for cls_name, count in class_counts.items():
            if count > 0:
                result_msg += f"{cls_name}: {count} 張\n"
        result_msg += f"\n總計處理：{processed_count['total']} 張影像"
        result_msg += f"\n結果已儲存至：{self.sorted_output_dir}"
        messagebox.showinfo("完成", result_msg)

    def append_log(self, text: str):
        try:
            self.log_text.insert(tk.END, text + "\n"); self.log_text.see(tk.END)
        except Exception:
            print(text)

    def on_tab_changed(self, event=None):
        if self.root.focus_get() is not None: self.root.after(100, self.display_annotation_image)

    def on_closing(self):
        try:
            self.stop_auto_capture()
            self.disable_realtime_detection()
            self.close_all_cameras()
            self.stop_preview_loop()
        finally:
            self.root.destroy()


def _normalize_classes(value):
    """
    object_classes 正規化為清單。

    接受 list（["object", "defect"]）或 dict（{"person": "人物"}，取 key），
    其他型別視為未設定。程式內部一律以清單操作（索引 / append / index）。
    """
    if isinstance(value, dict):
        return [str(k) for k in value.keys()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def _deep_merge(base: dict, override: dict) -> dict:
    """遞迴合併設定：override 有的覆蓋，沒有的沿用 base。"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def main():
    app = USBLearningApp(use_gui=True)
    if app.use_gui: 
        app.root.mainloop()
    else: 
        print("此腳本需要 GUI（tkinter）")


if __name__ == "__main__":
    main()