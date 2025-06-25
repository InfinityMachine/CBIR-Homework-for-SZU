"""
gui_retrieval.py  ⌁  Tkinter 美化版
-----------------------------------------------------------
依赖:  pillow  (pip install pillow)
"""

import time, threading, textwrap
from pathlib import Path
from tkinter import (
    Tk,
    ttk,
    Frame,
    Label,
    Button,
    filedialog,
    SUNKEN,
    PhotoImage,
    BOTH,
    LEFT,
    RIGHT,
    X,
)
from PIL import Image, ImageTk

# -------- 检索系统 --------
from image_retrieval_system_cpu import RetrievalSystem, cfg

TOP_K = 5  # 显示前 5 张
THUMB = 200  # 缩略图边长
BG_GRAY = "#f4f4f4"

sys = RetrievalSystem()  # 加载索引


# -------- Tkinter GUI --------
class Main(Tk):
    def __init__(self):
        super().__init__()
        self.title("示例图像检索 (CPU)")
        self.configure(bg=BG_GRAY, padx=12, pady=12)

        # 主题美化 (ttk)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", font=("Helvetica", 11), padding=6)
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("Caption.TLabel", foreground="#555")
        style.configure("Title.TLabel", font=("Helvetica", 11, "bold"))

        # ---- 顶部工具栏 ----
        top = Frame(self, bg=BG_GRAY)
        top.pack(fill=X)
        self.open_btn = ttk.Button(top, text="选择查询图像", command=self.choose_file)
        self.open_btn.pack(side=LEFT, padx=4)
        self.status_lbl = ttk.Label(top, text="请选择图片…")
        self.status_lbl.pack(side=LEFT, padx=12)

        # ---- 图像展示区 ----
        board = Frame(self, bg="#e1e1e1", relief=SUNKEN, bd=1)
        board.pack(pady=10, fill=BOTH, expand=True)

        # 查询图面板
        self.query_panel = ImagePanel(board, "Query", highlight=True)
        self.query_panel.frame.pack(side=LEFT, padx=6, pady=6)

        # 结果面板列表
        self.result_panels = []
        for i in range(TOP_K):
            p = ImagePanel(board, f"Top {i+1}")
            p.frame.pack(side=LEFT, padx=6, pady=6)
            self.result_panels.append(p)

    # ---------- 文件选择 ----------
    def choose_file(self):
        path = filedialog.askopenfilename(
            title="选择查询图像",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return
        self.query_panel.set_image(path, textwrap.fill(Path(path).name, 18))
        # 开启后台线程执行检索
        self.open_btn.state(["disabled"])
        self.status_lbl.config(text="检索中…")
        threading.Thread(target=self._search, args=(path,), daemon=True).start()

    # ---------- 检索线程 ----------
    def _search(self, qpath: str):
        t0 = time.perf_counter()
        results = sys.search(qpath, TOP_K)
        elapsed = (time.perf_counter() - t0) * 1000
        # 回到主线程更新 UI
        self.after(0, self._show_results, results, elapsed)

    # ---------- 显示结果 ----------
    def _show_results(self, results, ms: float):
        for panel, (p, score) in zip(self.result_panels, results):
            fname = Path(p).name
            panel.set_image(p, f"{score:.3f}\n{textwrap.fill(fname, 18)}")
        self.status_lbl.config(text=f"完成 · 耗时 {ms:.1f} ms")
        self.open_btn.state(["!disabled"])


# -------- 单个图片面板 --------
class ImagePanel:
    def __init__(self, master, title: str, highlight: bool = False):
        self.frame = Frame(
            master,
            width=THUMB,
            height=THUMB + 55,
            bg="white",
            bd=2,
            relief="ridge" if highlight else "flat",
        )
        self.frame.pack_propagate(True)  # 固定大小

        self.title_lbl = ttk.Label(self.frame, text=title, style="Title.TLabel")
        self.title_lbl.pack(pady=(4, 2))
        self.img_lbl = Label(self.frame, bg="#ddd")
        self.img_lbl.pack()
        self.caption = ttk.Label(
            self.frame, text="", style="Caption.TLabel", justify="center"
        )
        self.caption.pack(pady=2)
        self.photo: PhotoImage | None = None  # 引用保存

    def set_image(self, img_path: str, caption: str):
        im = Image.open(img_path).convert("RGB")
        im.thumbnail((THUMB, THUMB))
        self.photo = ImageTk.PhotoImage(im)
        self.img_lbl.config(image=self.photo)
        self.caption.config(text=caption)


# -------- main --------
if __name__ == "__main__":
    if not Path(cfg.index_path).exists():
        raise SystemExit("索引缺失，请先运行 build-index。")
    Main().mainloop()
