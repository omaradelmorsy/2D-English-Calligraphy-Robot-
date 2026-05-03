

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import math
import random
import os
import pickle
import numpy as np



BG         = "#0d0d0f"
SURFACE    = "#16161a"
SURFACE2   = "#1e1e24"
ACCENT     = "#c8a96e"
ACCENT2    = "#8b6f47"
TEXT_LIGHT = "#f0e6d3"
TEXT_DIM   = "#7a7068"
SUCCESS    = "#5cb85c"
WARNING    = "#e0a030"
ERROR      = "#e05252"

FONT_TITLE = ("Georgia", 24, "bold")
FONT_SUB   = ("Georgia", 10, "italic")
FONT_LABEL = ("Courier New", 10)
FONT_MONO  = ("Courier New", 9)
FONT_BTN   = ("Georgia", 10, "bold")
FONT_INPUT = ("Georgia", 15)

MODEL_PATH = "calligraphy_model.pkl"



CHARSET  = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?"
CHAR2IDX = {c: i for i, c in enumerate(CHARSET)}
VOCAB    = len(CHARSET)



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(s):
    return s * (1.0 - s)

def tanh_grad(t):
    return 1.0 - t ** 2



def _char_strokes(ch: str):
 
    lo = ch.lower()
    s  = []

    if ch == " ":
        return [(6.0, 0.0, 1.0)]

   
    if lo in "bdfhijklt":
        for _ in range(7):
            s.append((0.1 + random.gauss(0, 0.04),
                       1.3 + random.gauss(0, 0.04), 0.0))
        s.append((0.0, 0.0, 1.0))
        for _ in range(4):
            s.append((0.8 + random.gauss(0, 0.04),
                       random.gauss(0, 0.06), 0.0))
        s.append((0.0, 0.0, 1.0))

    
    elif lo in "gjpqy":
        for i in range(8):
            t = i / 8 * math.pi
            s.append((math.cos(t) * 1.9 + random.gauss(0, 0.05),
                       math.sin(t) * 2.7 + random.gauss(0, 0.05), 0.0))
        s.append((0.0, 0.0, 1.0))
        for _ in range(4):
            s.append((0.2 + random.gauss(0, 0.04),
                      -1.1 + random.gauss(0, 0.04), 0.0))
        s.append((0.0, 0.0, 1.0))

    
    elif lo in "aceos":
        for i in range(10):
            t = i / 10 * 2 * math.pi
            s.append((math.cos(t) * 2.1 + random.gauss(0, 0.06),
                       math.sin(t) * 2.9 + random.gauss(0, 0.06), 0.0))
        s.append((0.0, 0.0, 1.0))

    
    elif lo in "mnuvwxz":
        humps = 2 if lo in "mnw" else 1
        for _ in range(humps):
            for i in range(6):
                t = i / 6 * math.pi
                s.append((1.0 + random.gauss(0, 0.04),
                           math.sin(t) * 3.2 + random.gauss(0, 0.05), 0.0))
        s.append((0.0, 0.0, 1.0))

    
    elif ch.isdigit():
        for i in range(10):
            t = i / 10 * 2 * math.pi
            s.append((math.cos(t) * 1.8 + random.gauss(0, 0.05),
                       math.sin(t) * 2.6 + random.gauss(0, 0.05), 0.0))
        s.append((0.0, 0.0, 1.0))

    
    else:
        for i in range(6):
            s.append((0.8 + random.gauss(0, 0.04),
                       math.sin(i) * 1.1 + random.gauss(0, 0.04), 0.0))
        s.append((0.0, 0.0, 1.0))

    return s


def text_to_strokes(text: str, stroke_len: int = 64):
   
    raw = []
    for ch in text:
        raw.extend(_char_strokes(ch))
    raw = raw[:stroke_len]
    while len(raw) < stroke_len:
        raw.append((0.0, 0.0, 1.0))
    return raw


def make_dataset(n_samples: int = 1000, stroke_len: int = 64):
   
    words = [
        "hello", "world", "robot", "art", "pen", "draw", "write",
        "loop", "line", "curve", "ink", "stroke", "calligraphy",
        "wave", "flow", "brush", "paper", "motion", "craft", "style",
        "letter", "form", "shape", "grace", "beauty", "script",
        "elegant", "fluid", "smooth", "precise",
    ]
    X, Y = [], []
    for _ in range(n_samples):
        word   = random.choice(words)
        chars  = [CHAR2IDX.get(c, 0) for c in word]
        strokes = text_to_strokes(word, stroke_len)

        
        while len(chars) < stroke_len:
            chars.append(chars[-1])
        chars = chars[:stroke_len]

        X.append(chars)
        Y.append(strokes)

    return (np.array(X, dtype=np.int32),
            np.array(Y, dtype=np.float32))



class NumpyLSTM:
 

    def __init__(self, vocab=VOCAB, embed_dim=48,
                 hidden=128, out_dim=3, lr=0.001):
        self.V   = vocab
        self.E   = embed_dim
        self.H   = hidden
        self.out = out_dim
        self.lr  = lr

       
        self.emb = np.random.randn(vocab, embed_dim) * 0.05

       
        self.Wx1 = np.random.randn(embed_dim, 4 * hidden) * 0.05
        self.Wh1 = np.random.randn(hidden,    4 * hidden) * 0.05
        self.b1  = np.zeros(4 * hidden)
        self.b1[hidden:2*hidden] = 1.0   

        
        self.Wx2 = np.random.randn(hidden, 4 * hidden) * 0.05
        self.Wh2 = np.random.randn(hidden, 4 * hidden) * 0.05
        self.b2  = np.zeros(4 * hidden)
        self.b2[hidden:2*hidden] = 1.0

        
        self.Wo = np.random.randn(hidden, out_dim) * 0.05
        self.bo = np.zeros(out_dim)

       
        self._init_adam()

    def _init_adam(self):
        self._t  = 0
        params   = self._param_names()
        self._m  = {k: np.zeros_like(getattr(self, k)) for k in params}
        self._v  = {k: np.zeros_like(getattr(self, k)) for k in params}

    def _param_names(self):
        return ["emb", "Wx1", "Wh1", "b1",
                "Wx2", "Wh2", "b2", "Wo", "bo"]

    def _adam_update(self, grads: dict, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        lr_t = self.lr * math.sqrt(1 - b2**self._t) / (1 - b1**self._t)
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            setattr(self, k,
                    getattr(self, k) - lr_t * self._m[k] /
                    (np.sqrt(self._v[k]) + eps))

   
    @staticmethod
    def _lstm_cell(x, h, c, Wx, Wh, b):
        H = len(h)
        z = x @ Wx + h @ Wh + b
        ig = sigmoid(z[:H]);        fg = sigmoid(z[H:2*H])
        gg = np.tanh(z[2*H:3*H]);   og = sigmoid(z[3*H:4*H])
        c_new = fg * c + ig * gg
        h_new = og * np.tanh(c_new)
        return h_new, c_new, (ig, fg, gg, og, c_new, h_new, h, c, x, z)

    
    def forward(self, char_seq):
       
        T    = len(char_seq)
        H    = self.H
        h1   = np.zeros(H); c1 = np.zeros(H)
        h2   = np.zeros(H); c2 = np.zeros(H)
        outs = np.zeros((T, self.out))
        cache1, cache2, emb_cache = [], [], []

        for t in range(T):
            e = self.emb[char_seq[t]]
            h1, c1, ca1 = self._lstm_cell(e,  h1, c1, self.Wx1, self.Wh1, self.b1)
            h2, c2, ca2 = self._lstm_cell(h1, h2, c2, self.Wx2, self.Wh2, self.b2)
            raw = h2 @ self.Wo + self.bo
            outs[t] = [raw[0], raw[1], sigmoid(raw[2])]
            cache1.append(ca1); cache2.append(ca2)
            emb_cache.append((char_seq[t], e))

        return outs, (cache1, cache2, emb_cache)

   
    def backward(self, char_seq, targets, outputs, cache):
        T = len(char_seq)
        H = self.H
        cache1, cache2, emb_cache = cache

        grads = {k: np.zeros_like(getattr(self, k)) for k in self._param_names()}

        dh2 = np.zeros(H); dc2 = np.zeros(H)
        dh1 = np.zeros(H); dc1 = np.zeros(H)
        total_loss = 0.0

        for t in reversed(range(T)):
            pred = outputs[t]
            tgt  = targets[t]

            
            d_dx  = pred[0] - tgt[0]
            d_dy  = pred[1] - tgt[1]
            d_pen = pred[2] - tgt[2]
            total_loss += d_dx**2 + d_dy**2 + 0.5 * d_pen**2

            dout    = np.array([d_dx, d_dy, d_pen * sigmoid_grad(pred[2])])
            ca2     = cache2[t]
            h2_t    = ca2[5]
            grads["Wo"] += np.outer(h2_t, dout)
            grads["bo"] += dout
            dh2     = dh2 + self.Wo @ dout   

           
            ig2,fg2,gg2,og2,c2_new,h2_new,h2_prev,c2_prev,x2,z2 = ca2
            tanh_c2 = np.tanh(c2_new)
            dog2    = dh2 * tanh_c2 * sigmoid_grad(og2)
            dc2     = dc2 + dh2 * og2 * tanh_grad(tanh_c2)
            dig2    = dc2 * gg2 * sigmoid_grad(ig2)
            dfg2    = dc2 * c2_prev * sigmoid_grad(fg2)
            dgg2    = dc2 * ig2 * tanh_grad(gg2)
            dc2     = dc2 * fg2
            dz2     = np.concatenate([dig2, dfg2, dgg2, dog2])
            grads["Wx2"] += np.outer(x2,     dz2)
            grads["Wh2"] += np.outer(h2_prev, dz2)
            grads["b2"]  += dz2
            dh1      = dh1 + self.Wx2 @ dz2   
            dh2      = self.Wh2 @ dz2          

           
            ca1 = cache1[t]
            ig1,fg1,gg1,og1,c1_new,h1_new,h1_prev,c1_prev,x1,z1 = ca1
            tanh_c1 = np.tanh(c1_new)
            dog1    = dh1 * tanh_c1 * sigmoid_grad(og1)
            dc1     = dc1 + dh1 * og1 * tanh_grad(tanh_c1)
            dig1    = dc1 * gg1 * sigmoid_grad(ig1)
            dfg1    = dc1 * c1_prev * sigmoid_grad(fg1)
            dgg1    = dc1 * ig1 * tanh_grad(gg1)
            dc1     = dc1 * fg1
            dz1     = np.concatenate([dig1, dfg1, dgg1, dog1])
            grads["Wx1"] += np.outer(x1,     dz1)
            grads["Wh1"] += np.outer(h1_prev, dz1)
            grads["b1"]  += dz1
            dh1      = self.Wh1 @ dz1          
            demb     = dz1 @ self.Wx1.T        

            
            idx = emb_cache[t][0]
            grads["emb"][idx] += demb

        
        for k in grads:
            np.clip(grads[k], -5.0, 5.0, out=grads[k])

        self._adam_update(grads)
        return total_loss / T

    def save(self, path=MODEL_PATH):
        data = {k: getattr(self, k) for k in self._param_names()}
        data.update({"V": self.V, "E": self.E, "H": self.H,
                     "out": self.out, "lr": self.lr})
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path=MODEL_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = cls(data["V"], data["E"], data["H"], data["out"], data["lr"])
        for k in m._param_names():
            setattr(m, k, data[k])
        return m



def train_model(log_cb, progress_cb, done_cb,
                epochs=80, n_samples=1000, stroke_len=48, batch=32):
    try:
        log_cb("Generating synthetic stroke dataset...")
        X, Y = make_dataset(n_samples, stroke_len)
        log_cb(f"Dataset: {n_samples} samples × {stroke_len} steps.")

        model     = NumpyLSTM(hidden=128, lr=0.001)
        n         = len(X)
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            idx      = np.random.permutation(n)
            ep_loss  = 0.0
            batches  = 0

            for start in range(0, n, batch):
                b_idx = idx[start:start + batch]
                b_loss = 0.0
                for i in b_idx:
                    char_seq = X[i]
                    targets  = Y[i]
                    outs, cache = model.forward(char_seq)
                    loss = model.backward(char_seq, targets, outs, cache)
                    b_loss += loss
                ep_loss += b_loss / len(b_idx)
                batches += 1

            avg = ep_loss / batches
            if avg < best_loss:
                best_loss = avg
                model.save(MODEL_PATH)

            progress_cb(int(epoch / epochs * 100))
            if epoch % 10 == 0 or epoch == 1:
                log_cb(f"Epoch {epoch:3d}/{epochs}  "
                       f"loss={avg:.5f}  best={best_loss:.5f}")

        log_cb(f"Training complete. Best loss: {best_loss:.5f}")
        log_cb(f"Model saved → {MODEL_PATH}")
        done_cb(success=True)

    except Exception as e:
        import traceback
        log_cb(f"ERROR: {e}\n{traceback.format_exc()}")
        done_cb(success=False, error=str(e))




LETTER_WIDTH = 10.0   
LETTER_HEIGHT= 20.0   
CHAR_SPACING = 14.0   
Y_BASELINE   = 10.0   
FEEDRATE     = 800

def _letter_segments(ch: str):
  
    lo = ch.lower()

   
    if lo == 'a':
        return [[(0,0),(5,20),(10,0)], [(2,8),(8,8)]]
    elif lo == 'b':
        return [[(0,0),(0,20),(7,20),(9,17),(9,13),(7,10),(0,10)],
                [(0,10),(7,10),(9,7),(9,3),(7,0),(0,0)]]
    elif lo == 'c':
        return [[(9,16),(6,20),(3,20),(1,17),(0,13),(0,7),(1,3),(3,0),(6,0),(9,4)]]
    elif lo == 'd':
        return [[(0,0),(0,20),(5,20),(9,16),(9,4),(5,0),(0,0)]]
    elif lo == 'e':
        return [[(9,0),(0,0),(0,20),(9,20)], [(0,10),(7,10)]]
    elif lo == 'f':
        return [[(0,0),(0,20),(9,20)], [(0,12),(6,12)]]
    elif lo == 'g':
        return [[(9,16),(6,20),(3,20),(1,17),(0,13),(0,7),(1,3),(3,0),(6,0),
                 (9,3),(9,10),(5,10)]]
    elif lo == 'h':
        return [[(0,0),(0,20)], [(0,10),(9,10)], [(9,20),(9,0)]]
    elif lo == 'i':
        return [[(2,20),(8,20)], [(5,20),(5,0)], [(2,0),(8,0)]]
    elif lo == 'j':
        return [[(2,20),(8,20)], [(5,20),(5,3),(3,0),(1,0),(0,2)]]
    elif lo == 'k':
        return [[(0,0),(0,20)], [(9,20),(0,10),(9,0)]]
    elif lo == 'l':
        return [[(0,20),(0,0),(9,0)]]
    elif lo == 'm':
        return [[(0,0),(0,20),(5,10),(10,20),(10,0)]]
    elif lo == 'n':
        return [[(0,0),(0,20),(9,0),(9,20)]]
    elif lo == 'o':
        return [[(1,0),(0,4),(0,16),(1,20),(4,20),(6,20),(9,16),(9,4),(6,0),(4,0),(1,0)]]
    elif lo == 'p':
        return [[(0,0),(0,20),(7,20),(9,17),(9,13),(7,10),(0,10)]]
    elif lo == 'q':
        return [[(1,4),(0,8),(0,16),(1,20),(4,20),(6,20),(9,16),(9,4),(6,0),(4,0),(1,4)],
                [(7,3),(10,0)]]
    elif lo == 'r':
        return [[(0,0),(0,20),(7,20),(9,17),(9,13),(7,10),(0,10)], [(4,10),(9,0)]]
    elif lo == 's':
        return [[(9,17),(7,20),(3,20),(1,18),(1,14),(3,12),(7,8),(9,6),
                 (9,3),(7,0),(3,0),(1,3)]]
    elif lo == 't':
        return [[(5,20),(5,0)], [(1,20),(9,20)]]
    elif lo == 'u':
        return [[(0,20),(0,4),(2,0),(5,0),(8,0),(10,4),(10,20)]]
    elif lo == 'v':
        return [[(0,20),(5,0),(10,20)]]
    elif lo == 'w':
        return [[(0,20),(2,0),(5,10),(8,0),(10,20)]]
    elif lo == 'x':
        return [[(0,20),(10,0)], [(10,20),(0,0)]]
    elif lo == 'y':
        return [[(0,20),(5,10),(10,20)], [(5,10),(5,0)]]
    elif lo == 'z':
        return [[(0,20),(10,20),(0,0),(10,0)]]

    
    elif ch == '0':
        return [[(1,0),(0,4),(0,16),(1,20),(4,20),(6,20),(9,16),(9,4),(6,0),(4,0),(1,0)],
                [(2,4),(8,16)]]
    elif ch == '1':
        return [[(2,16),(5,20),(5,0)], [(2,0),(8,0)]]
    elif ch == '2':
        return [[(1,16),(2,20),(7,20),(9,16),(9,12),(1,4),(1,0),(9,0)]]
    elif ch == '3':
        return [[(1,20),(8,20),(9,16),(9,13),(7,11),(4,10),(7,9),(9,7),(9,3),(7,0),(1,0)]]
    elif ch == '4':
        return [[(7,0),(7,20),(0,8),(10,8)]]
    elif ch == '5':
        return [[(9,20),(1,20),(1,12),(7,12),(9,9),(9,3),(7,0),(1,0)]]
    elif ch == '6':
        return [[(8,18),(5,20),(3,20),(1,16),(0,10),(0,4),(2,0),(5,0),(8,0),
                 (9,4),(9,8),(7,11),(4,11),(1,10)]]
    elif ch == '7':
        return [[(0,20),(10,20),(4,0)]]
    elif ch == '8':
        return [[(5,10),(8,13),(8,17),(5,20),(2,17),(2,13),(5,10),
                 (8,7),(8,3),(5,0),(2,3),(2,7),(5,10)]]
    elif ch == '9':
        return [[(1,2),(4,0),(7,0),(9,4),(9,10),(7,13),(4,13),(1,10),
                 (1,6),(3,13),(5,20),(7,18)]]

    
    elif ch == '.':
        return [[(4,0),(5,0),(5,1),(4,1),(4,0)]]
    elif ch == ',':
        return [[(4,1),(5,1),(5,2),(4,0),(3,-1)]]
    elif ch == '!':
        return [[(5,20),(5,6)], [(5,2),(5,0)]]
    elif ch == '?':
        return [[(1,16),(2,20),(7,20),(9,16),(9,12),(5,8),(5,4)], [(5,1),(5,0)]]
    elif ch == ' ':
        return []   
    else:
        
        return [[(1,0),(1,20),(9,20),(9,0),(1,0)]]


def _get_rnn_style(ch: str, model: NumpyLSTM, h1, c1, h2, c2):
  
    idx = CHAR2IDX.get(ch, 0)
    e   = model.emb[idx]
    h1, c1, _ = NumpyLSTM._lstm_cell(e,  h1, c1, model.Wx1, model.Wh1, model.b1)
    h2, c2, _ = NumpyLSTM._lstm_cell(h1, h2, c2, model.Wx2, model.Wh2, model.b2)
    raw = h2 @ model.Wo + model.bo
   
    jitter = float(np.clip(abs(np.tanh(raw[0])) * 0.6, 0.0, 0.6))
    return jitter, h1, c1, h2, c2


def build_gcode(text: str, model: NumpyLSTM,
                progress_cb=None,
                pen_down_angle=90, pen_up_angle=0):
  
    lines = [
        ";  Calligraphy Robot G-Code ",
        ";  RNN  variation",
        "",
        "G21         ; millimeters",
        "G90         ; it is absolute positioning",
        "G28         ; it is home axes",
        f"M3 S{pen_up_angle}      ; pen up",
        "",
    ]

    H  = model.H
    h1 = np.zeros(H); c1 = np.zeros(H)
    h2 = np.zeros(H); c2 = np.zeros(H)

    cursor_x = 10.0   
    total    = max(len(text), 1)

    for i, ch in enumerate(text):

        
        jitter, h1, c1, h2, c2 = _get_rnn_style(ch, model, h1, c1, h2, c2)

        if ch == " ":
            cursor_x += CHAR_SPACING
            lines.append(f"; space")
            lines.append("")
            if progress_cb:
                progress_cb(int((i+1)/total*100))
            continue

        segments = _letter_segments(ch)
        lines.append(f"; ---------- {ch} ----------")

        for seg_idx, seg in enumerate(segments):
            if not seg:
                continue

            
            sx = cursor_x + seg[0][0]
            sy = Y_BASELINE + seg[0][1]
            lines.append(f"M3 S{pen_up_angle}      ; pen up")
            lines.append(f"G0 X{sx:.2f} Y{sy:.2f}")
            lines.append(f"M3 S{pen_down_angle}     ; pen down")

            
            for pt in seg[1:]:
                
                jx = pt[0] + random.gauss(0, jitter)
                jy = pt[1] + random.gauss(0, jitter)
                ax = cursor_x + jx
                ay = Y_BASELINE + jy
                ax = max(0.0, min(ax, 250.0))
                ay = max(0.0, min(ay, 250.0))
                lines.append(f"G1 X{ax:.2f} Y{ay:.2f} F{FEEDRATE}")

        lines.append(f"M3 S{pen_up_angle}      ; pen up after '{ch}'")
        lines.append("")
        cursor_x += CHAR_SPACING

        if progress_cb:
            progress_cb(int((i+1)/total*100))

    lines += [
        f"M3 S{pen_up_angle}      ; this is pen up final",
        "G28         ; returning home",
        ";  End ",
    ]
    return lines


def run_inference(text: str, model: NumpyLSTM,
                  stroke_len=64, progress_cb=None):
    
    return build_gcode(text, model, progress_cb=progress_cb)



class CalligraphyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calligraphy Robot — NumPy LSTM")
        self.configure(bg=BG)
        self.geometry("980x760")
        self.resizable(True, True)
        self.minsize(800, 600)

        self._gcode   = []
        self._running = False
        self._model   = None

        self._build_ui()
        self._check_model_on_start()

   
    def _check_model_on_start(self):
        if os.path.exists(MODEL_PATH):
            try:
                self._model = NumpyLSTM.load(MODEL_PATH)
                self._set_status(
                    f"Model loaded from {MODEL_PATH} — ready to generate.", SUCCESS)
                self._gen_btn.config(state="normal")
                self._train_btn.config(text="  RE-TRAIN MODEL")
            except Exception as e:
                self._set_status(f"Could not load model: {e} — please re-train.", WARNING)
                self._gen_btn.config(state="disabled")
        else:
            self._set_status(
                "No model found — press  TRAIN MODEL  first.", WARNING)
            self._gen_btn.config(state="disabled")

  
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        self._build_header()
        self._build_input_section()
        self._build_progress_row()
        self._build_panels()
        self._build_status_bar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG, pady=14)
        hdr.grid(row=0, column=0, sticky="ew", padx=30)
        tk.Label(hdr, text="CALLIGRAPHY ROBOT",
                 font=FONT_TITLE, bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(hdr, text="NumPy LSTM  ·  G-Code Generator",
                 font=FONT_SUB, bg=BG, fg=TEXT_DIM).pack(
                 side="left", padx=14, pady=4)
        tk.Frame(self, bg=ACCENT, height=1).grid(
            row=0, column=0, sticky="sew")

    def _build_input_section(self):
        outer = tk.Frame(self, bg=BG)
        outer.grid(row=1, column=0, sticky="ew", padx=30, pady=(16, 0))
        outer.columnconfigure(0, weight=1)

        tk.Label(outer, text="Enter text to write:",
                 font=FONT_LABEL, bg=BG, fg=TEXT_DIM).grid(
                 row=0, column=0, sticky="w", pady=(0, 5))

        ef = tk.Frame(outer, bg=ACCENT, padx=2, pady=2)
        ef.grid(row=1, column=0, sticky="ew")
        ef.columnconfigure(0, weight=1)

        self._text_entry = tk.Entry(
            ef, font=FONT_INPUT, bg=SURFACE, fg=TEXT_LIGHT,
            insertbackground=ACCENT, relief="flat", bd=8)
        self._text_entry.grid(row=0, column=0, sticky="ew")
        self._text_entry.bind("<Return>", lambda e: self._on_generate())
        self._text_entry.focus()

        self._char_var = tk.StringVar(value="0 chars")
        tk.Label(outer, textvariable=self._char_var,
                 font=FONT_MONO, bg=BG, fg=TEXT_DIM).grid(
                 row=2, column=0, sticky="e", pady=1)
        self._text_entry.bind("<KeyRelease>",
            lambda e: self._char_var.set(
                f"{len(self._text_entry.get())} chars"))

        btn_row = tk.Frame(outer, bg=BG)
        btn_row.grid(row=3, column=0, sticky="w", pady=(10, 0))

        self._train_btn = self._btn(
            btn_row, "  TRAIN MODEL", SURFACE2, ACCENT, self._on_train)
        self._train_btn.pack(side="left", padx=(0, 10))

        self._gen_btn = self._btn(
            btn_row, "  GENERATE G-CODE", ACCENT, BG,
            self._on_generate, state="disabled")
        self._gen_btn.pack(side="left", padx=(0, 10))

        self._clear_btn = self._btn(
            btn_row, "  CLEAR", SURFACE2, TEXT_DIM, self._on_clear)
        self._clear_btn.pack(side="left", padx=(0, 10))

        self._copy_btn = self._btn(
            btn_row, "  COPY G-CODE", SURFACE2, TEXT_DIM,
            self._on_copy, state="disabled")
        self._copy_btn.pack(side="left")

    def _build_progress_row(self):
        outer = tk.Frame(self, bg=BG)
        outer.grid(row=2, column=0, sticky="ew", padx=30, pady=(12, 0))

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Gold.Horizontal.TProgressbar",
                         troughcolor=SURFACE2, background=ACCENT,
                         bordercolor=SURFACE2,
                         lightcolor=ACCENT, darkcolor=ACCENT2)
        style.configure("Green.Horizontal.TProgressbar",
                         troughcolor=SURFACE2, background=SUCCESS,
                         bordercolor=SURFACE2,
                         lightcolor=SUCCESS, darkcolor="#3a7a3a")

        tk.Label(outer, text="TRAINING",
                 font=FONT_MONO, bg=BG, fg=TEXT_DIM).grid(
                 row=0, column=0, padx=(0, 8))
        self._train_prog = ttk.Progressbar(
            outer, length=300, style="Gold.Horizontal.TProgressbar")
        self._train_prog.grid(row=0, column=1, padx=(0, 6))
        self._train_pct = tk.StringVar(value="")
        tk.Label(outer, textvariable=self._train_pct,
                 font=FONT_MONO, bg=BG, fg=ACCENT, width=5).grid(
                 row=0, column=2)

        tk.Label(outer, text="INFERENCE",
                 font=FONT_MONO, bg=BG, fg=TEXT_DIM).grid(
                 row=0, column=3, padx=(20, 8))
        self._infer_prog = ttk.Progressbar(
            outer, length=200, style="Green.Horizontal.TProgressbar")
        self._infer_prog.grid(row=0, column=4, padx=(0, 6))
        self._infer_pct = tk.StringVar(value="")
        tk.Label(outer, textvariable=self._infer_pct,
                 font=FONT_MONO, bg=BG, fg=SUCCESS, width=5).grid(
                 row=0, column=5)

    def _build_panels(self):
        pane = tk.PanedWindow(self, orient="horizontal",
                               bg=BG, sashwidth=5, sashrelief="flat")
        pane.grid(row=3, column=0, sticky="nsew", padx=30, pady=14)
        self.rowconfigure(3, weight=1)

        
        left = tk.Frame(pane, bg=SURFACE)
        pane.add(left, minsize=260)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        self._panel_hdr(left, "TRAINING LOG", 0)
        self._log_box = scrolledtext.ScrolledText(
            left, font=FONT_MONO, bg=SURFACE, fg="#c8b89a",
            relief="flat", bd=0, state="disabled",
            wrap="none", padx=10, pady=8)
        self._log_box.grid(row=1, column=0, sticky="nsew")

        
        right = tk.Frame(pane, bg=SURFACE)
        pane.add(right, minsize=320)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        hdr_r = tk.Frame(right, bg=SURFACE2)
        hdr_r.grid(row=0, column=0, sticky="ew")
        hdr_r.columnconfigure(1, weight=1)
        tk.Label(hdr_r, text="G-CODE PREVIEW",
                 font=FONT_MONO, bg=SURFACE2, fg=ACCENT,
                 padx=12, pady=7).grid(row=0, column=0, sticky="w")
        self._line_var = tk.StringVar(value="")
        tk.Label(hdr_r, textvariable=self._line_var,
                 font=FONT_MONO, bg=SURFACE2, fg=TEXT_DIM,
                 padx=12).grid(row=0, column=2, sticky="e")

        self._gcode_box = scrolledtext.ScrolledText(
            right, font=FONT_MONO, bg=SURFACE, fg="#a8d8a8",
            selectbackground=ACCENT2, relief="flat", bd=0,
            state="disabled", wrap="none", padx=10, pady=8)
        self._gcode_box.grid(row=1, column=0, sticky="nsew")
        self._gcode_box.tag_config("comment", foreground=TEXT_DIM)
        self._gcode_box.tag_config("command", foreground="#a8d8a8")

    def _build_status_bar(self):
        self._status_var = tk.StringVar(value="Initialising...")
        bar = tk.Frame(self, bg=SURFACE2)
        bar.grid(row=4, column=0, sticky="ew")
        self._status_lbl = tk.Label(
            bar, textvariable=self._status_var,
            font=FONT_MONO, bg=SURFACE2, fg=ACCENT,
            anchor="w", padx=12, pady=5)
        self._status_lbl.pack(fill="x")

   
    def _btn(self, parent, text, bg, fg, cmd, state="normal"):
        return tk.Button(
            parent, text=text, font=FONT_BTN,
            bg=bg, fg=fg,
            activebackground=ACCENT2, activeforeground=BG,
            relief="flat", bd=0, padx=12, pady=7,
            cursor="hand2", command=cmd, state=state)

    def _panel_hdr(self, parent, title, row):
        h = tk.Frame(parent, bg=SURFACE2)
        h.grid(row=row, column=0, sticky="ew")
        tk.Label(h, text=title, font=FONT_MONO,
                 bg=SURFACE2, fg=ACCENT, padx=12, pady=7).pack(side="left")

    def _append_log(self, msg):
        self._log_box.config(state="normal")
        self._log_box.insert("end", msg + "\n")
        self._log_box.see("end")
        self._log_box.config(state="disabled")

    def _set_gcode(self, lines):
        self._gcode_box.config(state="normal")
        self._gcode_box.delete("1.0", "end")
        for line in lines:
            tag = "comment" if line.strip().startswith(";") else "command"
            self._gcode_box.insert("end", line + "\n", tag)
        self._gcode_box.config(state="disabled")
        n = len([l for l in lines if l.strip()])
        self._line_var.set(f"{n} lines" if lines else "")

    def _set_status(self, msg, color=ACCENT):
        self._status_var.set(msg)
        self._status_lbl.config(fg=color)

    def _lock(self):
        self._running = True
        for b in [self._train_btn, self._gen_btn,
                  self._clear_btn, self._copy_btn]:
            b.config(state="disabled")

    def _unlock(self, model_ready=True):
        self._running = False
        self._train_btn.config(state="normal")
        self._gen_btn.config(state="normal" if model_ready else "disabled")
        self._clear_btn.config(state="normal")

    
    def _on_train(self):
        if self._running:
            return
        self._lock()
        self._train_prog["value"] = 0
        self._train_pct.set("0%")
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")
        self._set_status("Training LSTM on synthetic strokes...", WARNING)

        def log_cb(msg):
            self.after(0, lambda m=msg: self._append_log(m))

        def progress_cb(pct):
            self.after(0, lambda p=pct: (
                self._train_prog.__setitem__("value", p),
                self._train_pct.set(f"{p}%")))

        def done_cb(success=True, error=""):
            self.after(0, lambda: self._on_train_done(success, error))

        threading.Thread(
            target=train_model,
            args=(log_cb, progress_cb, done_cb),
            daemon=True).start()

    def _on_train_done(self, success, error=""):
        if success:
            self._model = NumpyLSTM.load(MODEL_PATH)
            self._train_prog["value"] = 100
            self._train_pct.set("100%")
            self._set_status(
                "Training complete — enter text and press GENERATE.", SUCCESS)
            self._train_btn.config(text="  RE-TRAIN MODEL")
            self._unlock(model_ready=True)
        else:
            self._set_status(f"Training failed: {error}", ERROR)
            self._unlock(model_ready=False)

    def _on_generate(self):
        text = self._text_entry.get().strip()
        if not text:
            messagebox.showwarning("Empty Input", "Please enter some text first.")
            return
        if not self._model:
            messagebox.showwarning("No Model", "Train the model first.")
            return
        if self._running:
            return

        self._lock()
        self._infer_prog["value"] = 0
        self._infer_pct.set("0%")
        self._set_gcode([])
        self._set_status("Running LSTM inference...")
        model = self._model

        def progress_cb(pct):
            self.after(0, lambda p=pct: (
                self._infer_prog.__setitem__("value", p),
                self._infer_pct.set(f"{p}%")))

        def worker():
            gcode = run_inference(text, model, progress_cb=progress_cb)
            self.after(0, lambda g=gcode: self._on_generate_done(g))

        threading.Thread(target=worker, daemon=True).start()

    def _on_generate_done(self, gcode):
        self._gcode = gcode
        self._set_gcode(gcode)
        self._infer_prog["value"] = 100
        self._infer_pct.set("100%")
        n = len([l for l in gcode if l.strip()])
        self._set_status(f"G-Code ready — {n} lines generated.", SUCCESS)
        self._unlock(model_ready=True)
        self._copy_btn.config(state="normal")

    def _on_copy(self):
        if not self._gcode:
            return
        self.clipboard_clear()
        self.clipboard_append("\n".join(self._gcode))
        self._set_status("G-Code copied to clipboard!", ACCENT)

    def _on_clear(self):
        self._text_entry.delete(0, "end")
        self._char_var.set("0 chars")
        self._gcode = []
        self._set_gcode([])
        self._train_prog["value"] = 0
        self._infer_prog["value"] = 0
        self._train_pct.set("")
        self._infer_pct.set("")
        self._copy_btn.config(state="disabled")
        self._gen_btn.config(
            state="normal" if self._model else "disabled")
        self._set_status("Cleared — ready for new input.")
        self._text_entry.focus()



if __name__ == "__main__":
    app = CalligraphyApp()
    app.mainloop()