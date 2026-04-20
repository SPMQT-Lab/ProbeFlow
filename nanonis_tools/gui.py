"""Tkinter GUI for Nanonis .dat file conversion."""

import json
import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

CONFIG_PATH = Path.home() / ".nanonis_gui_config.json"
REPO_ROOT   = Path(__file__).resolve().parent.parent
DEFAULT_CUSHION_DIR = REPO_ROOT / "src" / "file_cushions"

# ── Themes ────────────────────────────────────────────────────────────────────

THEMES = {
    "dark": {
        "bg":        "#1e1e1e",
        "fg":        "#d4d4d4",
        "entry_bg":  "#2d2d2d",
        "btn_bg":    "#3c3c3c",
        "btn_fg":    "#d4d4d4",
        "log_bg":    "#1a1a1a",
        "log_fg":    "#d4d4d4",
        "ok_fg":     "#4ec994",
        "err_fg":    "#f44747",
        "warn_fg":   "#cd9731",
        "accent_bg": "#0e639c",
        "accent_fg": "#ffffff",
        "sep":       "#3c3c3c",
    },
    "light": {
        "bg":        "#f5f5f5",
        "fg":        "#1e1e1e",
        "entry_bg":  "#ffffff",
        "btn_bg":    "#e0e0e0",
        "btn_fg":    "#1e1e1e",
        "log_bg":    "#ffffff",
        "log_fg":    "#1e1e1e",
        "ok_fg":     "#1a7a1a",
        "err_fg":    "#c0392b",
        "warn_fg":   "#b07800",
        "accent_bg": "#0078d4",
        "accent_fg": "#ffffff",
        "sep":       "#cccccc",
    },
}

# ── Config persistence ─────────────────────────────────────────────────────────

def load_config() -> dict:
    defaults = {
        "dark_mode":  True,
        "input_dir":  "",
        "output_dir": "",
        "do_png":     True,
        "do_sxm":     True,
        "clip_low":   1.0,
        "clip_high":  99.0,
    }
    try:
        if CONFIG_PATH.exists():
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            defaults.update(saved)
    except Exception:
        pass
    return defaults


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── Queue log handler (worker thread → GUI thread) ────────────────────────────

class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(record)


# ── Main application ──────────────────────────────────────────────────────────

class NanonisGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Nanonis File Conversion")
        self.root.resizable(True, True)
        self.root.minsize(620, 580)

        self.cfg = load_config()
        self.log_queue: queue.Queue = queue.Queue()
        self._running = False
        self._advanced_visible = False

        # Tk variables
        self.dark_mode   = tk.BooleanVar(value=self.cfg["dark_mode"])
        self.input_dir   = tk.StringVar(value=self.cfg["input_dir"])
        self.output_dir  = tk.StringVar(value=self.cfg["output_dir"])
        self.do_png      = tk.BooleanVar(value=self.cfg["do_png"])
        self.do_sxm      = tk.BooleanVar(value=self.cfg["do_sxm"])
        self.clip_low    = tk.DoubleVar(value=self.cfg["clip_low"])
        self.clip_high   = tk.DoubleVar(value=self.cfg["clip_high"])

        self._build_ui()
        self._apply_theme()
        self._poll_log()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        PAD = {"padx": 14, "pady": 6}

        # ── Top bar ──
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=14, pady=(12, 4))

        tk.Label(top, text="Nanonis File Conversion", font=("Helvetica", 14, "bold")).pack(side="left")

        self.theme_btn = tk.Button(
            top, text="☀  Light" if self.dark_mode.get() else "🌙  Dark",
            relief="flat", bd=0, cursor="hand2",
            command=self._toggle_theme,
        )
        self.theme_btn.pack(side="right", padx=4)

        self._sep()

        # ── Input folder ──
        self._folder_row("Input folder:", self.input_dir, self._browse_input)

        # ── Output folder ──
        self._folder_row("Output folder:", self.output_dir, self._browse_output)

        self._sep()

        # ── Convert to ──
        conv_frame = tk.Frame(self.root)
        conv_frame.pack(fill="x", **PAD)
        tk.Label(conv_frame, text="Convert to:", width=14, anchor="w").pack(side="left")
        self.png_cb = tk.Checkbutton(conv_frame, text="PNG", variable=self.do_png)
        self.png_cb.pack(side="left", padx=(0, 16))
        self.sxm_cb = tk.Checkbutton(conv_frame, text="SXM", variable=self.do_sxm)
        self.sxm_cb.pack(side="left")

        self._sep()

        # ── Advanced options (collapsible) ──
        adv_toggle_frame = tk.Frame(self.root)
        adv_toggle_frame.pack(fill="x", padx=14, pady=(2, 0))
        self.adv_btn = tk.Button(
            adv_toggle_frame, text="▶  Advanced options",
            relief="flat", bd=0, cursor="hand2", anchor="w",
            command=self._toggle_advanced,
        )
        self.adv_btn.pack(side="left")

        self.adv_frame = tk.Frame(self.root)

        self._slider_row(self.adv_frame, "Clip low (%):", self.clip_low,   0.0,  10.0)
        self._slider_row(self.adv_frame, "Clip high (%):", self.clip_high, 90.0, 100.0)

        self._sep()

        # ── Run button ──
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=14, pady=6)
        self.run_btn = tk.Button(
            btn_frame, text="  RUN  ",
            font=("Helvetica", 11, "bold"),
            relief="flat", cursor="hand2",
            command=self._run,
        )
        self.run_btn.pack(expand=True, ipadx=20, ipady=6)

        self._sep()

        # ── Log panel ──
        log_label_frame = tk.Frame(self.root)
        log_label_frame.pack(fill="x", padx=14, pady=(4, 0))
        tk.Label(log_label_frame, text="Log", font=("Helvetica", 10, "bold")).pack(side="left")
        tk.Button(
            log_label_frame, text="Clear", relief="flat", cursor="hand2",
            command=self._clear_log,
        ).pack(side="right")

        self.log_text = tk.Text(
            self.root, height=12, wrap="word",
            relief="flat", bd=0,
            font=("Courier", 9),
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, padx=14, pady=(2, 12))

        scrollbar = tk.Scrollbar(self.log_text)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Text tags for colors (applied after theme)
        self.log_text.tag_config("ok",   foreground="#4ec994")
        self.log_text.tag_config("err",  foreground="#f44747")
        self.log_text.tag_config("warn", foreground="#cd9731")
        self.log_text.tag_config("info", foreground="#d4d4d4")

        # Save config on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _sep(self) -> None:
        self._seps = getattr(self, "_seps", [])
        sep = tk.Frame(self.root, height=1)
        sep.pack(fill="x", padx=14, pady=4)
        self._seps.append(sep)

    def _folder_row(self, label: str, var: tk.StringVar, cmd) -> None:
        frame = tk.Frame(self.root)
        frame.pack(fill="x", padx=14, pady=5)
        tk.Label(frame, text=label, width=14, anchor="w").pack(side="left")
        entry = tk.Entry(frame, textvariable=var, relief="flat", bd=2)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        btn = tk.Button(frame, text="Browse…", relief="flat", cursor="hand2", command=cmd)
        btn.pack(side="right")
        self._register_widget(entry, "entry")
        self._register_widget(btn,   "btn")

    def _slider_row(self, parent, label: str, var: tk.DoubleVar, from_: float, to: float) -> None:
        frame = tk.Frame(parent)
        frame.pack(fill="x", padx=14, pady=4)
        tk.Label(frame, text=label, width=14, anchor="w").pack(side="left")
        slider = tk.Scale(
            frame, variable=var, from_=from_, to=to,
            resolution=0.5, orient="horizontal",
            length=220, sliderlength=14,
            relief="flat", bd=0, highlightthickness=0,
        )
        slider.pack(side="left")
        self._register_widget(slider, "slider")

    # Widget registry for theme repainting
    _widget_registry: list = []

    def _register_widget(self, w, kind: str) -> None:
        self._widget_registry.append((w, kind))

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        t = THEMES["dark" if self.dark_mode.get() else "light"]

        self.root.configure(bg=t["bg"])
        self._repaint(self.root, t)

        # Log panel
        self.log_text.configure(bg=t["log_bg"], fg=t["log_fg"],
                                insertbackground=t["fg"])
        self.log_text.tag_config("ok",   foreground=t["ok_fg"])
        self.log_text.tag_config("err",  foreground=t["err_fg"])
        self.log_text.tag_config("warn", foreground=t["warn_fg"])
        self.log_text.tag_config("info", foreground=t["log_fg"])

        # Run button accent
        self.run_btn.configure(bg=t["accent_bg"], fg=t["accent_fg"],
                               activebackground=t["accent_bg"],
                               activeforeground=t["accent_fg"])

        # Separators
        for sep in getattr(self, "_seps", []):
            sep.configure(bg=t["sep"])

    def _repaint(self, widget, t: dict) -> None:
        cls = widget.winfo_class()
        try:
            if cls in ("Frame", "Toplevel"):
                widget.configure(bg=t["bg"])
            elif cls == "Label":
                widget.configure(bg=t["bg"], fg=t["fg"])
            elif cls == "Button":
                widget.configure(bg=t["btn_bg"], fg=t["btn_fg"],
                                 activebackground=t["bg"],
                                 activeforeground=t["fg"],
                                 relief="flat")
            elif cls == "Checkbutton":
                widget.configure(bg=t["bg"], fg=t["fg"],
                                 selectcolor=t["entry_bg"],
                                 activebackground=t["bg"],
                                 activeforeground=t["fg"])
            elif cls == "Entry":
                widget.configure(bg=t["entry_bg"], fg=t["fg"],
                                 insertbackground=t["fg"],
                                 relief="flat")
            elif cls == "Scale":
                widget.configure(bg=t["bg"], fg=t["fg"],
                                 troughcolor=t["entry_bg"],
                                 activebackground=t["accent_bg"])
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._repaint(child, t)

    def _toggle_theme(self) -> None:
        self.dark_mode.set(not self.dark_mode.get())
        self.theme_btn.config(
            text="☀  Light" if self.dark_mode.get() else "🌙  Dark"
        )
        self._apply_theme()

    # ── Advanced panel toggle ─────────────────────────────────────────────────

    def _toggle_advanced(self) -> None:
        if self._advanced_visible:
            self.adv_frame.pack_forget()
            self.adv_btn.config(text="▶  Advanced options")
        else:
            self.adv_frame.pack(fill="x", after=self.adv_btn.master)
            self.adv_btn.config(text="▼  Advanced options")
            self._repaint(self.adv_frame, THEMES["dark" if self.dark_mode.get() else "light"])
        self._advanced_visible = not self._advanced_visible

    # ── Folder pickers ────────────────────────────────────────────────────────

    def _browse_input(self) -> None:
        d = filedialog.askdirectory(title="Select input folder containing .dat files")
        if d:
            self.input_dir.set(d)

    def _browse_output(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir.set(d)

    # ── Log panel ─────────────────────────────────────────────────────────────

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
                if record.levelno >= logging.ERROR:
                    tag = "err"
                elif record.levelno == logging.WARNING:
                    tag = "warn"
                elif "[OK]" in msg:
                    tag = "ok"
                else:
                    tag = "info"
                self._log(msg, tag)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log)

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self) -> None:
        if self._running:
            return

        in_dir  = self.input_dir.get().strip()
        out_dir = self.output_dir.get().strip()

        if not in_dir:
            self._log("ERROR: Please select an input folder.", "err")
            return
        if not out_dir:
            self._log("ERROR: Please select an output folder.", "err")
            return
        if not self.do_png.get() and not self.do_sxm.get():
            self._log("ERROR: Please select at least one output format (PNG or SXM).", "err")
            return
        if not Path(in_dir).is_dir():
            self._log(f"ERROR: Input folder not found: {in_dir}", "err")
            return

        self._running = True
        self.run_btn.configure(text="  Running…  ", state="disabled")
        self._clear_log()

        handler = QueueHandler(self.log_queue)
        handler.setLevel(logging.DEBUG)

        thread = threading.Thread(
            target=self._worker,
            args=(in_dir, out_dir, handler),
            daemon=True,
        )
        thread.start()

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
                logger.info("── Starting PNG conversion ──")
                png_main(
                    src=in_path,
                    out_root=out_path / "png",
                    clip_low=clip_low,
                    clip_high=clip_high,
                    verbose=True,
                )

            if self.do_sxm.get():
                from nanonis_tools.dat_sxm_cli import main as sxm_main
                logger.info("── Starting SXM conversion ──")
                # Call directly to avoid argparse
                from nanonis_tools.dat_sxm_cli import convert_dat_to_sxm
                from nanonis_tools.common import setup_logging
                files = sorted(in_path.glob("*.dat"))
                if not files:
                    logger.warning("No .dat files found in %s", in_path)
                else:
                    logger.info("Found %d .dat file(s) to process", len(files))
                    errors = {}
                    sxm_out = out_path / "sxm"
                    sxm_out.mkdir(parents=True, exist_ok=True)
                    for i, dat in enumerate(files, 1):
                        logger.info("[%d/%d] Processing %s ...", i, len(files), dat.name)
                        try:
                            convert_dat_to_sxm(dat, sxm_out, DEFAULT_CUSHION_DIR, clip_low, clip_high)
                        except Exception as exc:
                            logger.error("FAILED %s: %s", dat.name, exc)
                            errors[dat.name] = str(exc)
                    if errors:
                        import json
                        err_path = sxm_out / "errors.json"
                        err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
                        logger.warning("%d file(s) failed. See errors.json", len(errors))
                    else:
                        logger.info("All SXM files processed successfully.")
                    logger.info("SXM outputs in: %s", sxm_out)

        except Exception as exc:
            logger.error("Unexpected error: %s", exc)
        finally:
            logger.removeHandler(handler)
            self.root.after(0, self._done)

    def _done(self) -> None:
        self._running = False
        self.run_btn.configure(text="  RUN  ", state="normal")

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        save_config({
            "dark_mode":  self.dark_mode.get(),
            "input_dir":  self.input_dir.get(),
            "output_dir": self.output_dir.get(),
            "do_png":     self.do_png.get(),
            "do_sxm":     self.do_sxm.get(),
            "clip_low":   self.clip_low.get(),
            "clip_high":  self.clip_high.get(),
        })
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    app = NanonisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
