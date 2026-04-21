"""ProbeFlow — graphical interface for Createc-to-Nanonis file conversion."""

import json
import logging
import queue
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog

from PIL import Image, ImageTk

CONFIG_PATH     = Path.home() / ".probeflow_config.json"
REPO_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_CUSHION = REPO_ROOT / "src" / "file_cushions"
LOGO_PATH       = REPO_ROOT / "assets" / "logo.png"
GITHUB_URL      = "https://github.com/SPMQT-Lab/Createc-to-Nanonis-file-conversion"

TOPBAR_BG = "#0e639c"
TOPBAR_FG = "#ffffff"

THEMES = {
    "dark": {
        "bg":        "#1e1e2e",
        "fg":        "#cdd6f4",
        "entry_bg":  "#313244",
        "btn_bg":    "#45475a",
        "btn_fg":    "#cdd6f4",
        "log_bg":    "#181825",
        "log_fg":    "#cdd6f4",
        "ok_fg":     "#a6e3a1",
        "err_fg":    "#f38ba8",
        "warn_fg":   "#fab387",
        "accent_bg": "#89b4fa",
        "accent_fg": "#1e1e2e",
        "sep":       "#45475a",
        "sub_fg":    "#6c7086",
    },
    "light": {
        "bg":        "#f8f9fa",
        "fg":        "#1e1e2e",
        "entry_bg":  "#ffffff",
        "btn_bg":    "#e0e0e0",
        "btn_fg":    "#1e1e2e",
        "log_bg":    "#ffffff",
        "log_fg":    "#1e1e2e",
        "ok_fg":     "#1a7a1a",
        "err_fg":    "#c0392b",
        "warn_fg":   "#b07800",
        "accent_bg": "#0078d4",
        "accent_fg": "#ffffff",
        "sep":       "#dee2e6",
        "sub_fg":    "#6c757d",
    },
}


def load_config() -> dict:
    defaults = {
        "dark_mode": True, "input_dir": "", "output_dir": "",
        "do_png": True, "do_sxm": True, "clip_low": 1.0, "clip_high": 99.0,
    }
    try:
        if CONFIG_PATH.exists():
            defaults.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    return defaults


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


class QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(record)


def show_about(parent: tk.Tk, dark: bool) -> None:
    t = THEMES["dark" if dark else "light"]
    win = tk.Toplevel(parent)
    win.title("About ProbeFlow")
    win.resizable(False, False)
    win.configure(bg=t["bg"])
    win.grab_set()

    try:
        img = Image.open(LOGO_PATH).convert("RGBA")
        img.thumbnail((260, 90), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo, bg=t["bg"])
        lbl.image = photo
        lbl.pack(pady=(18, 4))
    except Exception:
        pass

    def lbl(text, size=10, bold=False, color=None):
        tk.Label(win, text=text,
                 font=("Helvetica", size, "bold" if bold else "normal"),
                 bg=t["bg"], fg=color or t["fg"],
                 wraplength=340, justify="center").pack(pady=2, padx=20)

    lbl("ProbeFlow", 15, bold=True)
    lbl("Createc → Nanonis File Conversion", 10, color=t["sub_fg"])
    tk.Frame(win, height=1, bg=t["sep"]).pack(fill="x", padx=20, pady=10)
    lbl("Developed at SPMQT-Lab", 10, bold=True)
    lbl("Under the supervision of Dr. Peter Jacobson\nThe University of Queensland", 9, color=t["sub_fg"])
    tk.Frame(win, height=1, bg=t["sep"]).pack(fill="x", padx=20, pady=10)
    lbl("Original code by Rohan Platts", 10, bold=True)
    lbl("The core conversion algorithms were built by Rohan Platts.\n"
        "This software is a refactored and extended version of his work,\n"
        "developed within SPMQT-Lab.", 9, color=t["sub_fg"])
    tk.Frame(win, height=1, bg=t["sep"]).pack(fill="x", padx=20, pady=10)
    tk.Button(win, text="View on GitHub", bg=TOPBAR_BG, fg=TOPBAR_FG,
              relief="flat", cursor="hand2", font=("Helvetica", 9),
              command=lambda: webbrowser.open(GITHUB_URL)).pack(pady=(0, 16), ipadx=12, ipady=4)


class ProbeFlowGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ProbeFlow")
        self.root.minsize(640, 600)
        self.root.resizable(True, True)

        self.cfg = load_config()
        self.log_queue: queue.Queue = queue.Queue()
        self._running = False
        self._advanced_visible = False
        self._seps = []
        self._registry = []

        self.dark_mode  = tk.BooleanVar(value=self.cfg["dark_mode"])
        self.input_dir  = tk.StringVar(value=self.cfg["input_dir"])
        self.output_dir = tk.StringVar(value=self.cfg["output_dir"])
        self.do_png     = tk.BooleanVar(value=self.cfg["do_png"])
        self.do_sxm     = tk.BooleanVar(value=self.cfg["do_sxm"])
        self.clip_low   = tk.DoubleVar(value=self.cfg["clip_low"])
        self.clip_high  = tk.DoubleVar(value=self.cfg["clip_high"])

        self._build_ui()
        self._apply_theme()
        self._poll_log()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        # ── Top bar ──────────────────────────────────────────────────────────
        topbar = tk.Frame(self.root, bg=TOPBAR_BG, height=52)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        try:
            img = Image.open(LOGO_PATH).convert("RGBA")
            img.thumbnail((110, 40), Image.LANCZOS)
            data = img.getdata()
            img.putdata([(r, g, b, 0) if r > 220 and g > 220 and b > 220 else (r, g, b, a)
                         for r, g, b, a in data])
            self._logo_photo = ImageTk.PhotoImage(img)
            logo_lbl = tk.Label(topbar, image=self._logo_photo, bg=TOPBAR_BG, cursor="hand2")
            logo_lbl.pack(side="left", padx=(10, 0), pady=6)
            logo_lbl.bind("<Button-1>", lambda e: webbrowser.open(GITHUB_URL))
        except Exception:
            pass

        tk.Label(topbar, text="ProbeFlow", font=("Helvetica", 14, "bold"),
                 bg=TOPBAR_BG, fg=TOPBAR_FG).pack(side="left", padx=(8, 0))
        tk.Label(topbar, text="Createc → Nanonis", font=("Helvetica", 9),
                 bg=TOPBAR_BG, fg="#a8c8e8").pack(side="left", padx=(6, 0))

        tk.Button(topbar, text="About", bg=TOPBAR_BG, fg=TOPBAR_FG,
                  relief="flat", cursor="hand2", font=("Helvetica", 9),
                  activebackground="#1a7ab0", activeforeground=TOPBAR_FG,
                  command=lambda: show_about(self.root, self.dark_mode.get())
                  ).pack(side="right", padx=(0, 10), pady=12)

        self.theme_btn = tk.Button(
            topbar, text="☀" if self.dark_mode.get() else "🌙",
            bg=TOPBAR_BG, fg=TOPBAR_FG, relief="flat", cursor="hand2",
            activebackground="#1a7ab0", activeforeground=TOPBAR_FG,
            font=("Helvetica", 11), command=self._toggle_theme,
        )
        self.theme_btn.pack(side="right", padx=(0, 4), pady=12)

        # ── Content ──────────────────────────────────────────────────────────
        self.content = tk.Frame(self.root)
        self.content.pack(fill="both", expand=True)
        tk.Frame(self.content, height=6).pack()

        self._folder_row("Input folder:",  self.input_dir,  self._browse_input)
        self._folder_row("Output folder:", self.output_dir, self._browse_output)
        self._sep()

        conv = tk.Frame(self.content)
        conv.pack(fill="x", padx=16, pady=6)
        tk.Label(conv, text="Convert to:", width=14, anchor="w").pack(side="left")
        self.png_cb = tk.Checkbutton(conv, text="PNG", variable=self.do_png)
        self.png_cb.pack(side="left", padx=(0, 20))
        self.sxm_cb = tk.Checkbutton(conv, text="SXM", variable=self.do_sxm)
        self.sxm_cb.pack(side="left")
        self._sep()

        adv_hdr = tk.Frame(self.content)
        adv_hdr.pack(fill="x", padx=16, pady=(2, 0))
        self.adv_btn = tk.Button(adv_hdr, text="▶  Advanced options",
                                 relief="flat", bd=0, cursor="hand2", anchor="w",
                                 command=self._toggle_advanced)
        self.adv_btn.pack(side="left")

        self.adv_frame = tk.Frame(self.content)
        self._slider_row(self.adv_frame, "Clip low (%):",  self.clip_low,   0.0,  10.0)
        self._slider_row(self.adv_frame, "Clip high (%):", self.clip_high, 90.0, 100.0)
        self._sep()

        btn_f = tk.Frame(self.content)
        btn_f.pack(fill="x", padx=16, pady=6)
        self.run_btn = tk.Button(btn_f, text="  RUN  ",
                                 font=("Helvetica", 11, "bold"),
                                 relief="flat", cursor="hand2", command=self._run)
        self.run_btn.pack(expand=True, ipadx=24, ipady=7)
        self._sep()

        log_hdr = tk.Frame(self.content)
        log_hdr.pack(fill="x", padx=16, pady=(2, 0))
        tk.Label(log_hdr, text="Log", font=("Helvetica", 10, "bold")).pack(side="left")
        tk.Button(log_hdr, text="Clear", relief="flat", cursor="hand2",
                  command=self._clear_log).pack(side="right")

        self.log_text = tk.Text(self.content, height=10, wrap="word",
                                relief="flat", bd=0, font=("Courier", 9),
                                state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(2, 0))

        self.footer = tk.Frame(self.content)
        self.footer.pack(fill="x", padx=16, pady=(6, 10))
        self.footer_lbl = tk.Label(
            self.footer,
            text="SPMQT-Lab · Dr. Peter Jacobson · The University of Queensland  |  Original code by Rohan Platts",
            font=("Helvetica", 8), anchor="center",
        )
        self.footer_lbl.pack()

    def _sep(self) -> None:
        s = tk.Frame(self.content, height=1)
        s.pack(fill="x", padx=16, pady=5)
        self._seps.append(s)

    def _folder_row(self, label: str, var: tk.StringVar, cmd) -> None:
        f = tk.Frame(self.content)
        f.pack(fill="x", padx=16, pady=5)
        tk.Label(f, text=label, width=14, anchor="w").pack(side="left")
        e = tk.Entry(f, textvariable=var, relief="flat", bd=2)
        e.pack(side="left", fill="x", expand=True, padx=(0, 6))
        b = tk.Button(f, text="Browse…", relief="flat", cursor="hand2", command=cmd)
        b.pack(side="right")
        self._registry.extend([(e, "entry"), (b, "btn")])

    def _slider_row(self, parent, label: str, var: tk.DoubleVar, from_: float, to: float) -> None:
        f = tk.Frame(parent)
        f.pack(fill="x", padx=16, pady=4)
        tk.Label(f, text=label, width=14, anchor="w").pack(side="left")
        s = tk.Scale(f, variable=var, from_=from_, to=to, resolution=0.5,
                     orient="horizontal", length=220, sliderlength=14,
                     relief="flat", bd=0, highlightthickness=0)
        s.pack(side="left")
        self._registry.append((s, "slider"))

    def _apply_theme(self) -> None:
        t = THEMES["dark" if self.dark_mode.get() else "light"]
        self.root.configure(bg=t["bg"])
        self.content.configure(bg=t["bg"])
        self._repaint(self.content, t)
        self.log_text.configure(bg=t["log_bg"], fg=t["log_fg"], insertbackground=t["fg"])
        self.log_text.tag_config("ok",   foreground=t["ok_fg"])
        self.log_text.tag_config("err",  foreground=t["err_fg"])
        self.log_text.tag_config("warn", foreground=t["warn_fg"])
        self.log_text.tag_config("info", foreground=t["log_fg"])
        self.run_btn.configure(bg=t["accent_bg"], fg=t["accent_fg"],
                               activebackground=t["accent_bg"], activeforeground=t["accent_fg"])
        self.footer.configure(bg=t["bg"])
        self.footer_lbl.configure(bg=t["bg"], fg=t["sub_fg"])
        for s in self._seps:
            s.configure(bg=t["sep"])

    def _repaint(self, widget, t: dict) -> None:
        cls = widget.winfo_class()
        try:
            if cls == "Frame":
                widget.configure(bg=t["bg"])
            elif cls == "Label":
                widget.configure(bg=t["bg"], fg=t["fg"])
            elif cls == "Button":
                widget.configure(bg=t["btn_bg"], fg=t["btn_fg"],
                                 activebackground=t["bg"], activeforeground=t["fg"], relief="flat")
            elif cls == "Checkbutton":
                widget.configure(bg=t["bg"], fg=t["fg"], selectcolor=t["entry_bg"],
                                 activebackground=t["bg"], activeforeground=t["fg"])
            elif cls == "Entry":
                widget.configure(bg=t["entry_bg"], fg=t["fg"],
                                 insertbackground=t["fg"], relief="flat")
            elif cls == "Scale":
                widget.configure(bg=t["bg"], fg=t["fg"],
                                 troughcolor=t["entry_bg"], activebackground=t["accent_bg"])
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._repaint(child, t)

    def _toggle_theme(self) -> None:
        self.dark_mode.set(not self.dark_mode.get())
        self.theme_btn.config(text="☀" if self.dark_mode.get() else "🌙")
        self._apply_theme()

    def _toggle_advanced(self) -> None:
        if self._advanced_visible:
            self.adv_frame.pack_forget()
            self.adv_btn.config(text="▶  Advanced options")
        else:
            self.adv_frame.pack(fill="x")
            self.adv_btn.config(text="▼  Advanced options")
            self._repaint(self.adv_frame, THEMES["dark" if self.dark_mode.get() else "light"])
        self._advanced_visible = not self._advanced_visible

    def _browse_input(self) -> None:
        d = filedialog.askdirectory(title="Select input folder containing .dat files")
        if d:
            self.input_dir.set(d)

    def _browse_output(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir.set(d)

    def _log(self, msg: str, tag: str = "info") -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _poll_log(self) -> None:
        try:
            while True:
                record = self.log_queue.get_nowait()
                msg = record.getMessage()
                tag = ("err" if record.levelno >= logging.ERROR else
                       "warn" if record.levelno == logging.WARNING else
                       "ok" if "[OK]" in msg else "info")
                self._log(msg, tag)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log)

    def _run(self) -> None:
        if self._running:
            return
        in_dir  = self.input_dir.get().strip()
        out_dir = self.output_dir.get().strip()
        if not in_dir:
            self._log("ERROR: Please select an input folder.", "err"); return
        if not out_dir:
            self._log("ERROR: Please select an output folder.", "err"); return
        if not self.do_png.get() and not self.do_sxm.get():
            self._log("ERROR: Select at least one output format.", "err"); return
        if not Path(in_dir).is_dir():
            self._log(f"ERROR: Input folder not found: {in_dir}", "err"); return

        self._running = True
        self.run_btn.configure(text="  Running…  ", state="disabled")
        self._clear_log()
        handler = QueueHandler(self.log_queue)
        handler.setLevel(logging.DEBUG)
        threading.Thread(target=self._worker, args=(in_dir, out_dir, handler), daemon=True).start()

    def _worker(self, in_dir: str, out_dir: str, handler: QueueHandler) -> None:
        logger = logging.getLogger("nanonis_tools")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        clip_low  = self.clip_low.get()
        clip_high = self.clip_high.get()
        in_path   = Path(in_dir)
        out_path  = Path(out_dir)
        try:
            if self.do_png.get():
                from nanonis_tools.dats_to_pngs import main as png_main
                logger.info("── PNG conversion ──────────────────────")
                png_main(src=in_path, out_root=out_path / "png",
                         clip_low=clip_low, clip_high=clip_high, verbose=True)

            if self.do_sxm.get():
                from nanonis_tools.dat_sxm_cli import convert_dat_to_sxm
                logger.info("── SXM conversion ──────────────────────")
                files = sorted(in_path.glob("*.dat"))
                if not files:
                    logger.warning("No .dat files found in %s", in_path)
                else:
                    sxm_out = out_path / "sxm"
                    sxm_out.mkdir(parents=True, exist_ok=True)
                    errors = {}
                    logger.info("Found %d .dat file(s)", len(files))
                    for i, dat in enumerate(files, 1):
                        logger.info("[%d/%d] %s ...", i, len(files), dat.name)
                        try:
                            convert_dat_to_sxm(dat, sxm_out, DEFAULT_CUSHION, clip_low, clip_high)
                        except Exception as exc:
                            logger.error("FAILED %s: %s", dat.name, exc)
                            errors[dat.name] = str(exc)
                    if errors:
                        import json as _j
                        (sxm_out / "errors.json").write_text(_j.dumps(errors, indent=2))
                        logger.warning("%d file(s) failed — see errors.json", len(errors))
                    else:
                        logger.info("All SXM files processed successfully.")
                    logger.info("Outputs: %s", sxm_out)
        except Exception as exc:
            logger.error("Unexpected error: %s", exc)
        finally:
            logger.removeHandler(handler)
            self.root.after(0, self._done)

    def _done(self) -> None:
        self._running = False
        self.run_btn.configure(text="  RUN  ", state="normal")

    def _on_close(self) -> None:
        save_config({
            "dark_mode": self.dark_mode.get(), "input_dir": self.input_dir.get(),
            "output_dir": self.output_dir.get(), "do_png": self.do_png.get(),
            "do_sxm": self.do_sxm.get(), "clip_low": self.clip_low.get(),
            "clip_high": self.clip_high.get(),
        })
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ProbeFlowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
