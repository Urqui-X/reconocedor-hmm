"""
GUI.py - Interfaz grafica (Tkinter + Canvas, sin dependencias extra).
    Ventana A: MenuView          (seleccion 2D / 3D)
    Ventana B: Recognizer2DView  (lienzo 2D + panel de probabilidades)
    Ventana C: Recognizer3DView  (esfera 3D con rotacion/zoom + panel)
Toda la matematica vive en HMMEngine / FeatureExtractor / Session; aqui solo se dibuja.

Correcciones del profesor:
  * El MODO se reconoce sin mirar los botones: cambia el CURSOR (lapiz = dibujar, mano =
    rotar/manipular, flecha de 4 puntas mientras se arrastra la esfera), el borde del lienzo y
    una etiqueta de modo.
  * Umbral de reconocimiento: si ninguna letra supera el 20 % (o el trazo no se parece a
    ninguna letra) se avisa "no se reconocio ningun gesto" en vez de dar una letra equivocada.
  * Escala y coherencia: lineas guia y proyeccion al tamano que escribe la persona; los iconos
    del panel son muestras reales de cada letra.
"""
import json
import math
import os
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from FeatureExtractor import XH_RAD
from Geometry import GeometryModel, PairRefiner
from HMMEngine import HMMClassifier
from LetterTemplates import ALPHABET, TEMPLATES
from Rejection import PROB_THRESHOLD, Rejector
from Session import Session2D, Session3D

# ---------------------------------------------------------------- estilo
BG, CARD, INK, MUTED = "#f1f5f9", "#ffffff", "#0f172a", "#64748b"
NAVY, ACCENT, LINE, GUIDE = "#0f172a", "#4f46e5", "#e2e8f0", "#cbd5e1"
DRAW_C, ROT_C = "#2563eb", "#d97706"          # color de cada modo (dibujar / rotar)
OK_BG, OK_FG = "#dcfce7", "#166534"
WARN_BG, WARN_FG = "#fef3c7", "#92400e"
IDLE_BG, IDLE_FG = "#e2e8f0", "#475569"
FONT = "Segoe UI"
TEMPERATURE = 2.5        # calibracion de la confianza mostrada (T=1 es el posterior bayesiano puro)
XH_PX = 90               # altura de la 'x' en pixeles (lienzo 2D): ascendente = 2x, descendente = 1x
CANVAS = 600
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DESCENDERS = set("gjpqy")
ICON_ONLY = set(ALPHABET)   # en el panel todas las letras se muestran solo con su dibujo (sin rotulo)
IDLE_TEXT = "Dibuja una letra. Puedes levantar el mouse entre trazos."


def _load_json(name):
    try:
        with open(os.path.join(MODEL_DIR, name), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _icon_strokes():
    """Muestra real representativa de cada letra (models/icons.json); si falta, la plantilla."""
    icons = _load_json("icons.json")
    return {ch: icons.get(ch) or [np.asarray(TEMPLATES[ch]).tolist()] for ch in ALPHABET}


# ================================================================ panel derecho
class ResultsPanel(tk.Frame):
    """Listado ordenado descendentemente: icono, letra, barra (con marca del umbral) y probabilidad."""
    BAR_W = 150

    def __init__(self, master, icons):
        super().__init__(master, bg=CARD, highlightthickness=1, highlightbackground=LINE)
        self.icons = icons
        top = tk.Frame(self, bg=CARD)
        top.pack(fill="x", padx=14, pady=(12, 2))
        tk.Label(top, text="Clasificación", font=(FONT, 12, "bold"), bg=CARD, fg=INK).pack(side="left")
        tk.Label(top, text="P(letra | trazo)", font=(FONT, 9), bg=CARD, fg=MUTED).pack(side="left", padx=8)
        leg = tk.Frame(self, bg=CARD)
        leg.pack(fill="x", padx=14, pady=(0, 6))
        tick = tk.Canvas(leg, width=10, height=12, bg=CARD, highlightthickness=0)
        tick.create_line(5, 0, 5, 12, fill="#ef4444", width=2)
        tick.pack(side="left")
        tk.Label(leg, text=f"umbral de reconocimiento = {PROB_THRESHOLD*100:.0f} %", font=(FONT, 8),
                 bg=CARD, fg=MUTED).pack(side="left", padx=4)
        wrap = tk.Frame(self, bg=CARD)
        wrap.pack(fill="both", expand=True)
        self.cv = tk.Canvas(wrap, bg=CARD, highlightthickness=0, width=310)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.cv.yview)
        self.inner = tk.Frame(self.cv, bg=CARD)
        self.inner.bind("<Configure>", lambda e: self.cv.configure(scrollregion=self.cv.bbox("all")))
        self.cv.create_window((0, 0), window=self.inner, anchor="nw")
        self.cv.configure(yscrollcommand=sb.set)
        self.cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.cv.bind("<Enter>", lambda e: self._bind_wheel(True))
        self.cv.bind("<Leave>", lambda e: self._bind_wheel(False))
        self.rows = {ch: self._make_row(ch) for ch in ALPHABET}
        self.reset()

    def _bind_wheel(self, on):
        f = lambda e: self.cv.yview_scroll(int(-e.delta / 120) if e.delta else 0, "units")
        up = lambda e: self.cv.yview_scroll(-1, "units")
        dn = lambda e: self.cv.yview_scroll(1, "units")
        if on:
            self.cv.bind_all("<MouseWheel>", f)
            self.cv.bind_all("<Button-4>", up)
            self.cv.bind_all("<Button-5>", dn)
        else:
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.cv.unbind_all(ev)

    def _make_row(self, ch):
        row = tk.Frame(self.inner, bg=CARD)
        icon = tk.Canvas(row, width=42, height=50, bg=CARD, highlightthickness=0)
        icon.grid(row=0, column=0, rowspan=2, padx=(10, 4), pady=2)
        self._draw_icon(icon, ch)
        lab = tk.Label(row, text="" if ch in ICON_ONLY else ch, font=(FONT, 18, "bold"), bg=CARD, fg=INK, width=2)
        lab.grid(row=0, column=1, rowspan=2)
        bar = tk.Canvas(row, width=self.BAR_W, height=10, bg=LINE, highlightthickness=0)
        bar.grid(row=0, column=2, padx=8, pady=(12, 0), sticky="w")
        fill = bar.create_rectangle(0, 0, 0, 10, fill=ACCENT, width=0)
        bar.create_line(self.BAR_W * PROB_THRESHOLD, 0, self.BAR_W * PROB_THRESHOLD, 10, fill="#ef4444", width=2)
        pct = tk.Label(row, text="0.0 %", font=(FONT, 9), bg=CARD, fg=MUTED)
        pct.grid(row=1, column=2, sticky="w", padx=8)
        return {"frame": row, "bar": bar, "fill": fill, "pct": pct, "icon": icon}

    def _draw_icon(self, cv, ch):
        strokes = self.icons.get(ch)
        if not strokes:
            return
        allp = np.vstack([np.array(s) for s in strokes])
        u0 = allp[:, 0].min()
        v0 = allp[:, 1].min() + (1.0 if ch in DESCENDERS else 0.0)     # linea base alineada
        sc = 12.0
        for k, s in enumerate(strokes):
            pts = []
            for u, v in np.array(s):
                pts += [6 + (u - u0) * sc, 36 - (v - v0) * sc]
            if len(pts) >= 4:
                cv.create_line(*pts, fill="#334155" if k == 0 else "#2563eb", width=1.6, capstyle="round")
            cv.create_oval(pts[0] - 2, pts[1] - 2, pts[0] + 2, pts[1] + 2, fill="#16a34a", outline="")

    def reset(self):
        self.update_ranking([(c, 1.0 / len(ALPHABET)) for c in ALPHABET], recognized=False)

    def update_ranking(self, ranking, recognized=True):
        for i, (ch, p) in enumerate(ranking):
            r = self.rows[ch]
            r["frame"].grid(row=i, column=0, sticky="ew", pady=1)
            r["pct"].config(text=f"{p*100:5.1f} %")
            r["bar"].coords(r["fill"], 0, 0, self.BAR_W * p, 10)
            r["bar"].itemconfig(r["fill"], fill=ACCENT if p >= PROB_THRESHOLD else "#94a3b8")
            bg = "#eef2ff" if (i == 0 and recognized) else CARD
            r["frame"].config(bg=bg)
            r["icon"].config(bg=bg)
            for w in r["frame"].winfo_children():
                if isinstance(w, tk.Label):
                    w.config(bg=bg)


# ================================================================ base
class RecognizerView(tk.Frame):
    TITLE, OTHER = "", ""
    HELP = ""

    def __init__(self, app):
        super().__init__(app.container, bg=BG)
        self.app = app
        head = tk.Frame(self, bg=NAVY)
        head.pack(fill="x")
        self._flat(head, "←  Menú", lambda: app.show("menu")).pack(side="left", padx=(12, 6), pady=10)
        tk.Label(head, text=self.TITLE, font=(FONT, 15, "bold"), bg=NAVY, fg="white").pack(side="left", padx=10)
        self.chip = tk.Label(head, text="", font=(FONT, 10, "bold"), fg="white", padx=12, pady=4)
        self.chip.pack(side="left", padx=14)
        self._flat(head, f"Cambiar a {self.OTHER}", lambda: app.show(self.OTHER.lower())).pack(side="right", padx=12)
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")
        self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS, bg="white", highlightthickness=3,
                                highlightbackground=LINE)
        self.canvas.pack()
        bar = tk.Frame(left, bg=BG)
        bar.pack(fill="x", pady=(10, 6))
        ttk.Button(bar, text="Limpiar lienzo", command=self.clear).pack(side="left")
        ttk.Button(bar, text="Reiniciar captura", command=self.restart).pack(side="left", padx=6)
        ttk.Button(bar, text="✔  Finalizar letra", style="Primary.TButton", command=self.finish_capture).pack(side="left", padx=6)
        self.build_extra_controls(bar)
        self.status = tk.Label(left, text="", font=(FONT, 11, "bold"), anchor="w", justify="left", padx=12,
                               pady=8, wraplength=CANVAS - 10)
        self.status.pack(fill="x")
        self.hint = tk.Label(left, text=self.HELP, font=(FONT, 9), fg=MUTED, bg=BG, anchor="w", justify="left",
                             wraplength=CANVAS - 10)
        self.hint.pack(fill="x", pady=(4, 0))
        self.panel = ResultsPanel(body, app.icons)
        self.panel.pack(side="left", fill="both", expand=True, padx=(16, 0))
        self.set_status("idle", IDLE_TEXT)
        app.bind("<Return>", lambda e: self.winfo_ismapped() and self.finish_capture(), add="+")

    @staticmethod
    def _flat(master, text, cmd):
        return tk.Button(master, text=text, command=cmd, bg="#1e293b", fg="white", activebackground="#334155",
                         activeforeground="white", relief="flat", bd=0, padx=12, pady=5, font=(FONT, 10),
                         cursor="hand2")

    def build_extra_controls(self, bar):
        pass

    # ---- estado visible
    def set_mode_chip(self, text, color):
        self.chip.config(text=text, bg=color)
        self.canvas.config(highlightbackground=color)

    def set_status(self, kind, text):
        bg, fg = {"ok": (OK_BG, OK_FG), "warn": (WARN_BG, WARN_FG)}.get(kind, (IDLE_BG, IDLE_FG))
        self.status.config(text=text, bg=bg, fg=fg)

    def show_result(self, res, final=False):
        self.panel.update_ranking(res.display_ranking or res.ranking, recognized=res.recognized)
        p = res.ranking[0][1]
        if res.verdict.reason == "vacio":
            self.set_status("idle", res.verdict.message)
        elif not res.recognized:
            wait = ""
            if not final and res.verdict.reason == "ambiguo":
                top = ", ".join(f"«{c}» {p*100:.0f} %" for c, p in res.ranking[:3])
                wait = f"  Candidatas: {top}. Sigue dibujando o pulsa «Finalizar letra» (Enter)."
            self.set_status("warn", "⚠  No se reconoció ningún gesto.  " + res.verdict.message + wait)
        else:
            tail = "   ·   letra finalizada" if final else "   ·   (en curso)"
            self.set_status("ok", f"Predicción: «{res.best}»  {p*100:.1f} %{tail}")
        self.hint.config(text=self._similar_hint(res, final))

    def _similar_hint(self, res, final):
        sim = self.app.similar.get(res.best, [])[:2]
        if not (final and res.recognized and sim):
            return self.HELP
        txt = "  ·  ".join(f"«{s['letter']}»: {s['reason']}" for s in sim)
        return f"Letras parecidas a «{res.best}» →  {txt}"

    def reset_panel(self):
        self.panel.reset()
        self.set_status("idle", IDLE_TEXT)
        self.hint.config(text=self.HELP)

    # ---- botones (los dos exigidos por la guia)
    def clear(self):
        """Limpiar lienzo: borra lo dibujado y la proyeccion; conserva el ultimo resultado en el panel."""
        raise NotImplementedError

    def restart(self):
        """Reiniciar captura: borra el lienzo Y reinicia el reconocedor y el panel (y la vista 3D)."""
        raise NotImplementedError


# ================================================================ 2D
class Recognizer2DView(RecognizerView):
    TITLE, OTHER = "Reconocedor 2D", "3D"
    HELP = ("Escribe en el mismo sentido que el icono (punto verde = inicio): «l» e «i» se trazan de ARRIBA hacia abajo. "
            "Escribe entre las líneas guía: la «x» ocupa entre la línea continua (base) y la primera punteada de arriba; "
            "las letras altas (b, d, f, h, k, l, t) llegan a la segunda punteada; g, j, p, q, y bajan hasta la punteada de abajo. "
            "Un clic sin mover dibuja un punto (i, j, ñ).  Enter = finalizar.")

    def __init__(self, app, clf, K, geo, rej, ref):
        super().__init__(app)
        self.session = Session2D(clf, K, geo, rej, ref)
        self.base = 380
        self.segments, self.pts = [], []
        self.canvas.config(cursor="pencil")                       # modo unico: dibujar
        self.set_mode_chip("✏  MODO DIBUJAR", DRAW_C)
        self.draw_guides()
        c = self.canvas
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_move)
        c.bind("<ButtonRelease-1>", self.on_release)

    def draw_guides(self):
        c, b, xh = self.canvas, self.base, XH_PX
        c.delete("guide")
        c.create_rectangle(0, b - xh, CANVAS, b, fill="#f8fafc", outline="", tags="guide")   # banda de la «x»
        for y, txt, dash in ((b - 2 * xh, "ascendente", (3, 4)), (b - xh, "altura x", (3, 4)),
                             (b, "línea base", None), (b + xh, "descendente", (3, 4))):
            c.create_line(16, y, CANVAS - 16, y, fill=GUIDE, dash=dash, tags="guide")
        c.tag_lower("guide")

    def to_uv(self, pts):
        a = np.array(pts, dtype=float)
        return np.stack([a[:, 0] / XH_PX, (self.base - a[:, 1]) / XH_PX], axis=1)

    def from_uv(self, uv):
        return [(float(u) * XH_PX, self.base - float(v) * XH_PX) for u, v in uv]

    def _all_segments(self):
        result = list(self.segments)
        if len(self.pts) >= 2:
            result.append(self.to_uv(self.pts))
        return result

    def on_press(self, e):
        self.pts = [(e.x, e.y)]                                   # cada clic empieza un trazo nuevo
        self.canvas.create_oval(e.x - 2, e.y - 2, e.x + 2, e.y + 2, fill=INK, outline=INK, tags="stroke")

    def on_move(self, e):
        if not self.pts:
            return
        x0, y0 = self.pts[-1]
        if math.hypot(e.x - x0, e.y - y0) < 2:
            return
        self.canvas.create_line(x0, y0, e.x, e.y, fill=INK, width=3, capstyle="round", tags="stroke")
        self.pts.append((e.x, e.y))
        self.update_recognition()

    def on_release(self, e):
        if not self.pts:
            return
        if len(self.pts) == 1:                                    # clic sin mover = punto
            x, y = self.pts[0]
            self.pts = [(x, y), (x, y + 0.5)]
        self.segments.append(self.to_uv(self.pts))
        self.pts = []
        self.update_recognition()

    def update_recognition(self, final=False):
        segments = self._all_segments()
        if not segments:
            return
        res = self.session.process_segments(segments, final=final)
        self.show_result(res, final)
        self.canvas.delete("proj", "hint")
        if res.recognized and len(res.projection) >= 2:
            xy = self.from_uv(res.projection)
            self.canvas.create_line(*[v for point in xy for v in point], fill=ACCENT, width=3, dash=(5, 5),
                                    capstyle="round", tags="proj")
            ex, ey = xy[-1]
            self.canvas.create_text(ex + 12, ey, text=res.best, fill=ACCENT, font=(FONT, 18, "bold"), tags="hint")

    def finish_capture(self):
        if len(self.pts) > 1:
            self.segments.append(self.to_uv(self.pts))
            self.pts = []
        if self.segments:
            self.update_recognition(final=True)

    def clear(self):
        self.canvas.delete("stroke", "proj", "hint")
        self.segments, self.pts = [], []
        self.session.reset()
        self.draw_guides()
        self.set_status("idle", "Lienzo limpio. El último resultado sigue en el panel; dibuja otra letra.")

    def restart(self):
        self.clear()
        self.reset_panel()


# ================================================================ 3D
def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Recognizer3DView(RecognizerView):
    TITLE, OTHER = "Reconocedor 3D (esfera)", "2D"
    R0 = 240
    HELP = ("✏ Dibujar: arrastra con el clic izquierdo.   ✋ Rotar: clic derecho + arrastrar, mantener Shift, "
            "o tecla R (cambia de modo).   Rueda = zoom.   Enter = finalizar.   El cuadro punteado inferior "
            "es el tamaño sugerido de la «x».")

    def __init__(self, app, clf, K, geo, rej, ref):
        super().__init__(app)
        self.session = Session3D(clf, K, geo, rej, ref)
        self.Rm = np.eye(3)                       # objeto -> vista
        self.zoom = 1.0
        self.view_pts = np.zeros((0, 3))          # trazo actual en el marco de la vista
        self.stroke_segments_obj = []             # trazos terminados, adheridos a la esfera
        self.active_stroke_obj = np.zeros((0, 3))
        self.snapR = np.eye(3)
        self.proj_obj = np.zeros((0, 3))
        self.best = ""
        self.mode = None                          # None | "draw" | "rotate"
        self.last_xy = None
        self.hover_xy = None
        self.rotate_mode = False                  # modo "manipular" fijo (boton / tecla R)
        self.shift_held = False                   # modo "manipular" temporal
        self.rotating = False                     # arrastrando la esfera ahora mismo
        c = self.canvas
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_move)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Motion>", self.on_hover)
        for b in ("<ButtonPress-3>", "<ButtonPress-2>", "<Shift-ButtonPress-1>"):
            c.bind(b, self.on_rot_press)
        for b in ("<B3-Motion>", "<B2-Motion>", "<Shift-B1-Motion>"):
            c.bind(b, self.on_rot_move)
        for b in ("<ButtonRelease-3>", "<ButtonRelease-2>", "<Shift-ButtonRelease-1>"):
            c.bind(b, self.on_rot_release)
        c.bind("<MouseWheel>", lambda e: self.set_zoom(self.zoom * (1.08 if e.delta > 0 else 1 / 1.08)))
        c.bind("<Button-4>", lambda e: self.set_zoom(self.zoom * 1.08))
        c.bind("<Button-5>", lambda e: self.set_zoom(self.zoom / 1.08))
        for k in ("Shift_L", "Shift_R"):
            app.bind(f"<KeyPress-{k}>", self.on_shift_down, add="+")
            app.bind(f"<KeyRelease-{k}>", self.on_shift_up, add="+")
        for k in ("r", "R"):
            app.bind(f"<KeyPress-{k}>", lambda e: self.winfo_ismapped() and self.toggle_rotate_mode(), add="+")
        app.bind("<Escape>", lambda e: self.winfo_ismapped() and self.set_rotate_mode(False), add="+")
        self.refresh_mode()
        self.redraw()

    def build_extra_controls(self, bar):
        ttk.Button(bar, text="Zoom +", command=lambda: self.set_zoom(self.zoom * 1.15)).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Zoom −", command=lambda: self.set_zoom(self.zoom / 1.15)).pack(side="left", padx=4)
        self.rotate_button = ttk.Button(bar, text="✋ Rotar (R)", command=self.toggle_rotate_mode)
        self.rotate_button.pack(side="left", padx=4)

    # ---------------- modo y cursor (indicacion intuitiva del estado)
    def manipulating(self):
        return self.rotate_mode or self.shift_held or self.rotating

    def refresh_mode(self):
        """Cursor + etiqueta + borde del lienzo: el modo se ve sin mirar los botones."""
        if self.rotating:
            cur, text, col = "fleur", "✊  ROTANDO LA ESFERA", ROT_C
        elif self.rotate_mode or self.shift_held:
            cur, text, col = "hand2", "✋  MODO ROTAR / MANIPULAR", ROT_C
        else:
            over = self.hover_xy is None or self.screen_to_view(*self.hover_xy) is not None
            cur, text, col = ("pencil" if over else "arrow"), "✏  MODO DIBUJAR", DRAW_C
        self.canvas.config(cursor=cur)
        self.set_mode_chip(text, col)
        if hasattr(self, "rotate_button"):
            self.rotate_button.config(text="✏ Volver a dibujar (R)" if self.rotate_mode else "✋ Rotar (R)")

    def set_rotate_mode(self, on):
        self.rotate_mode = bool(on)
        self.refresh_mode()
        self.redraw()

    def toggle_rotate_mode(self):
        self.set_rotate_mode(not self.rotate_mode)

    def on_hover(self, e):
        self.hover_xy = (e.x, e.y)
        shift = bool(e.state & 0x0001)
        if shift != self.shift_held:
            self.shift_held = shift
            self.redraw()
        self.refresh_mode()

    def on_shift_down(self, e):
        if self.winfo_ismapped():
            self.shift_held = True
            self.refresh_mode()
            self.redraw()

    def on_shift_up(self, e):
        if self.winfo_ismapped():
            self.shift_held = False
            self.refresh_mode()
            self.redraw()

    # ---------------- geometria
    @property
    def R(self):
        return self.R0 * self.zoom

    @property
    def cx(self):
        return CANVAS / 2

    def screen_to_view(self, X, Y):
        x, y = (X - self.cx) / self.R, (self.cx - Y) / self.R
        r2 = x * x + y * y
        if r2 >= 0.985 ** 2:
            return None
        return np.array([x, y, math.sqrt(1 - r2)])

    def view_to_screen(self, v):
        return self.cx + self.R * v[:, 0], self.cx - self.R * v[:, 1]

    def _runs(self, view, **kw):
        """Dibuja la parte visible (z>0) de una polilinea 3D ya expresada en el marco de la vista."""
        if len(view) < 2:
            return
        X, Y = self.view_to_screen(view)
        vis = view[:, 2] > 0.0
        run = []
        for i in range(len(view)):
            if vis[i]:
                run += [X[i], Y[i]]
            else:
                if len(run) >= 4:
                    self.canvas.create_line(*run, **kw)
                run = []
        if len(run) >= 4:
            self.canvas.create_line(*run, **kw)

    def redraw(self):
        c = self.canvas
        c.delete("all")
        R, cx = self.R, self.cx
        rot = self.manipulating()
        c.create_oval(cx - R, cx - R, cx + R, cx + R, fill="#eef2f7", outline=ROT_C if rot else "#cbd5e1",
                      width=3 if rot else 2)
        c.create_oval(cx - R * .72, cx - R * .85, cx - R * .1, cx - R * .3, fill="#f5f8fb", outline="")
        t = np.radians(np.linspace(0, 360, 90))
        for lat in range(-60, 61, 30):                           # paralelos
            la = math.radians(lat)
            ring = np.stack([np.cos(la) * np.sin(t), np.full_like(t, math.sin(la)), np.cos(la) * np.cos(t)], 1)
            self._runs(ring @ self.Rm.T, fill="#c7d0db" if lat else "#9aa7b8", width=1)
        for lon in range(0, 180, 30):                            # meridianos
            lo = math.radians(lon)
            m = np.stack([np.cos(t) * math.sin(lo), np.sin(t), np.cos(t) * math.cos(lo)], 1)
            self._runs(m @ self.Rm.T, fill="#c7d0db", width=1)
        for seg in self.stroke_segments_obj:
            if len(seg) > 1:
                self._runs(seg @ self.Rm.T, fill=INK, width=3, capstyle="round")
        if len(self.active_stroke_obj) > 1:
            self._runs(self.active_stroke_obj @ self.Rm.T, fill=INK, width=3, capstyle="round")
        if len(self.proj_obj) > 1:
            self._runs(self.proj_obj @ self.Rm.T, fill=ACCENT, width=3, dash=(5, 5), capstyle="round")
            if self.best:
                end = self.proj_obj[-1] @ self.Rm.T
                if end[2] > 0:
                    X, Y = self.view_to_screen(end[None, :])
                    c.create_text(X[0] + 12, Y[0], text=self.best, fill=ACCENT, font=(FONT, 18, "bold"))
        s = XH_RAD * R                                            # referencia de tamaño: una 'x'
        c.create_rectangle(14, CANVAS - 14 - s, 14 + s, CANVAS - 14, outline=GUIDE, dash=(3, 3))
        c.create_text(20 + s, CANVAS - 14 - s / 2, text="tamaño sugerido de la «x»", anchor="w",
                      fill="#94a3b8", font=(FONT, 8))
        label, col = ("✋ ROTAR", ROT_C) if rot else ("✏ DIBUJAR", DRAW_C)   # refuerza el cursor
        c.create_text(16, 14, text=label, anchor="w", fill=col, font=(FONT, 11, "bold"))

    # ---------------- interaccion
    def on_press(self, e):
        if self.rotate_mode:
            self.mode = "rotate"
            self.rotating = True
            self.last_xy = (e.x, e.y)
            self.refresh_mode()
            return
        v = self.screen_to_view(e.x, e.y)
        if v is None:
            self.mode = None
            return
        self.mode = "draw"
        self.snapR = self.Rm.copy()                               # cada pulsacion = trazo independiente
        self.view_pts = v[None, :]
        self.active_stroke_obj = (self.snapR.T @ v)[None, :]
        self.proj_obj = np.zeros((0, 3))
        self.best = ""

    def on_move(self, e):
        if self.mode == "rotate":
            self.on_rot_move(e)
            return
        if self.mode != "draw":
            return
        v = self.screen_to_view(e.x, e.y)
        if v is None or math.acos(np.clip(v @ self.view_pts[-1], -1, 1)) < 0.004:
            return
        self.view_pts = np.vstack([self.view_pts, v])
        self.active_stroke_obj = np.vstack([self.active_stroke_obj, self.snapR.T @ v])
        self.update_recognition()

    def on_release(self, e):
        if self.mode == "rotate":
            self.rotating = False
        elif self.mode == "draw" and len(self.view_pts) >= 1:
            if len(self.view_pts) == 1:                           # clic sin mover = punto
                w = self.view_pts[0] + np.array([0.0, -0.0015, 0.0])
                self.active_stroke_obj = np.vstack([self.active_stroke_obj,
                                                    self.snapR.T @ (w / np.linalg.norm(w))])
            self.stroke_segments_obj.append(self.active_stroke_obj.copy())
            self.active_stroke_obj = np.zeros((0, 3))
            self.view_pts = np.zeros((0, 3))
            self.update_recognition()
        self.mode = None
        self.last_xy = None
        self.rotating = False                                     # tambien si se solto Shift antes
        self.refresh_mode()
        self.redraw()

    def update_recognition(self, final=False):
        segments_view = [seg @ self.Rm.T for seg in self.stroke_segments_obj]
        if len(self.view_pts) > 1:
            segments_view.append(self.view_pts)
        if not segments_view:
            return
        res = self.session.process_segments(segments_view, final=final)
        self.show_result(res, final)
        self.best = res.best if res.recognized else ""
        self.proj_obj = (self.Rm.T @ res.projection.T).T if len(res.projection) else np.zeros((0, 3))
        self.redraw()

    def finish_capture(self):
        if self.stroke_segments_obj:
            self.update_recognition(final=True)

    def on_rot_press(self, e):
        self.last_xy = (e.x, e.y)
        self.rotating = True
        self.refresh_mode()
        self.redraw()

    def on_rot_release(self, e):
        self.last_xy = None
        self.rotating = False
        self.refresh_mode()
        self.redraw()

    def on_rot_move(self, e):
        if self.last_xy is None:
            return
        dx, dy = e.x - self.last_xy[0], e.y - self.last_xy[1]
        self.last_xy = (e.x, e.y)
        k = 1.0 / self.R
        self.Rm = _rot_x(dy * k) @ _rot_y(dx * k) @ self.Rm
        u, _, vt = np.linalg.svd(self.Rm)                        # mantener ortonormalidad
        self.Rm = u @ vt
        self.redraw()

    def set_zoom(self, z):
        self.zoom = float(np.clip(z, 0.55, 1.35))
        self.redraw()

    def clear(self):
        self.stroke_segments_obj = []
        self.active_stroke_obj = np.zeros((0, 3))
        self.proj_obj = np.zeros((0, 3))
        self.view_pts = np.zeros((0, 3))
        self.best = ""
        self.session.reset()
        self.set_status("idle", "Esfera limpia. El último resultado sigue en el panel; dibuja otra letra.")
        self.redraw()

    def restart(self):
        self.Rm, self.zoom = np.eye(3), 1.0
        self.clear()
        self.reset_panel()


# ================================================================ menú
class MenuView(tk.Frame):
    def __init__(self, app):
        super().__init__(app.container, bg=BG)
        band = tk.Frame(self, bg=NAVY)
        band.pack(fill="x")
        tk.Label(band, text="UNIVERSIDAD DE LOS LLANOS", font=(FONT, 10, "bold"), bg=NAVY, fg="#94a3b8").pack(pady=(18, 0))
        tk.Label(band, text="Ingeniería de Sistemas · Procesos Estocásticos (603504)", font=(FONT, 10), bg=NAVY,
                 fg="#94a3b8").pack(pady=(0, 18))
        box = tk.Frame(self, bg=BG)
        box.place(relx=.5, rely=.52, anchor="center")
        tk.Label(box, text="Reconocedor de caracteres con HMM", font=(FONT, 26, "bold"), bg=BG, fg=INK).pack()
        tk.Label(box, text="Clasificación y autocompletado de trazos en tiempo real", font=(FONT, 11), bg=BG,
                 fg=MUTED).pack(pady=(4, 26))
        row = tk.Frame(box, bg=BG)
        row.pack()
        for title, sub, key in (("Modo 2D", "plano cartesiano", "2d"), ("Modo 3D", "superficie esférica", "3d")):
            card = tk.Frame(row, bg=CARD, highlightthickness=1, highlightbackground=LINE, cursor="hand2")
            card.pack(side="left", padx=12)
            tk.Label(card, text=title, font=(FONT, 18, "bold"), bg=CARD, fg=ACCENT, padx=46, pady=18).pack()
            tk.Label(card, text=sub, font=(FONT, 10), bg=CARD, fg=MUTED).pack(pady=(0, 18))
            for w in (card, *card.winfo_children()):
                w.bind("<Button-1>", lambda e, k=key: app.show(k))
        tk.Label(box, text="Samuel Esteban Rojas Amézquita - 160005133", font=(FONT, 10), bg=BG, fg=MUTED).pack(pady=(34, 2))
        tk.Label(box, text="Julian Camilo Urquiza Leon - 160005136", font=(FONT, 10), bg=BG, fg=MUTED).pack()
        tk.Label(box, text="Willy Stiven Saenz Betancourt - 160005028", font=(FONT, 10), bg=BG, fg=MUTED).pack()
        tk.Label(box, text="Darwin Rene Roncancio Silva - 160005027", font=(FONT, 10), bg=BG, fg=MUTED).pack()


# ================================================================ aplicación
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HMM · Reconocedor de caracteres 2D/3D")
        self.geometry("1100x800")
        self.minsize(1040, 780)
        self.configure(bg=BG)
        self._style()
        self.container = tk.Frame(self, bg=BG)
        self.container.pack(fill="both", expand=True)
        self.icons = _icon_strokes()
        self.similar = _load_json("similarity.json")
        try:
            c2, m2 = HMMClassifier.load(os.path.join(MODEL_DIR, "hmm_2d.json"), TEMPERATURE)
            c3, m3 = HMMClassifier.load(os.path.join(MODEL_DIR, "hmm_3d.json"), TEMPERATURE)
        except FileNotFoundError:
            messagebox.showerror("Modelos no encontrados",
                                 "No existen models/hmm_2d.json ni models/hmm_3d.json.\n"
                                 "Ejecuta primero:  python train_models.py")
            self.destroy()
            raise SystemExit(1)
        self.views = {"menu": MenuView(self)}
        for key, clf, m, cls in (("2d", c2, m2, Recognizer2DView), ("3d", c3, m3, Recognizer3DView)):
            self.views[key] = cls(self, clf, m["K"], GeometryModel.from_dict(m["geometry"]),
                                  Rejector.from_dict(m["rejector"]), PairRefiner.from_dict(m.get("refiner", {})))
        self.show("menu")

    def _style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure("TButton", padding=(12, 6), font=(FONT, 10), background="#e2e8f0", foreground=INK, borderwidth=0)
        st.map("TButton", background=[("active", "#cbd5e1")])
        st.configure("Primary.TButton", background=ACCENT, foreground="white", font=(FONT, 10, "bold"))
        st.map("Primary.TButton", background=[("active", "#4338ca")])

    def show(self, name):
        for v in self.views.values():
            v.pack_forget()
        self.views[name].pack(fill="both", expand=True)
