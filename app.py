import re
import math
import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st
from sympy import symbols

# ---------- Page / Theme ----------
st.set_page_config(page_title="SciCalc • Streamlit", page_icon="🧮", layout="centered")
st.title("🧮 Scientific Calculator")
st.caption("Streamlit • SymPy-powered • Plotting • History • Degree/Radian toggle")

# ---------- Session State Defaults ----------
if "display" not in st.session_state:
    st.session_state.display = ""
if "ans" not in st.session_state:
    st.session_state.ans = "0"
if "memory" not in st.session_state:
    st.session_state.memory = 0.0
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"expr":..., "result":...}
if "angle_mode" not in st.session_state:
    st.session_state.angle_mode = "RAD"  # or "DEG"
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = None

# ---------- Angle Mode ----------
col_mode1, col_mode2 = st.columns([1, 3])
with col_mode1:
    st.session_state.angle_mode = st.radio(
        "Angle", ["RAD", "DEG"], index=0, horizontal=True
    )
with col_mode2:
    st.write("")

# ---------- Helpers ----------
x = symbols("x")

def _preprocess(expr: str) -> str:
    """Light preprocessing: ^ to **, percent, factorial (!) into factorial(), implicit × sign cleanup."""
    if not expr:
        return expr

    # Convert ^ to ** for exponent
    expr = expr.replace("^", "**")

    # Replace trailing % to /100 (supports 50% or (a+b)%)
    #  e.g., "50%" -> "(50)/100", "(2*x)%" -> "((2*x))/100"
    expr = re.sub(r"(\))\s*%", r"\1/100", expr)
    expr = re.sub(r"(\d+(\.\d+)?)\s*%", r"(\1)/100", expr)

    # Factorial: turn n! or )! into factorial(n) / factorial( ... )
    # Handles nested parentheses: " (2+3)! " -> " factorial((2+3)) "
    expr = re.sub(r"(\d+|\([^\(\)]*\))\s*!",
                  lambda m: f"factorial({m.group(1)})", expr)

    # Remove accidental unicode multiplication signs
    expr = expr.replace("×", "*").replace("÷", "/")

    return expr

# Degree-mode wrappers (SymPy expects radians)
def sind(z): return sp.sin(sp.pi*z/180)
def cosd(z): return sp.cos(sp.pi*z/180)
def tand(z): return sp.tan(sp.pi*z/180)
def asind(z): return sp.asin(z)*180/sp.pi
def acosd(z): return sp.acos(z)*180/sp.pi
def atand(z): return sp.atan(z)*180/sp.pi

def evaluate(expr: str) -> sp.Expr | str:
    """Safely evaluate with SymPy; returns sympy object or string error."""
    if not expr:
        return ""
    expr_raw = expr
    expr = _preprocess(expr)

    # Allow Ans keyword
    expr = expr.replace("Ans", f"({st.session_state.ans})")

    # Allowed names and functions
    common_locals = {
        "x": x,
        "pi": sp.pi,
        "e": sp.E,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "ln": sp.log,
        "log": sp.log,     # natural log by default; user can write log10
        "log10": sp.log10,
        "exp": sp.exp,
        "floor": sp.floor,
        "ceil": sp.ceiling,
        "gamma": sp.gamma,
        "factorial": sp.factorial,
        # hyperbolics (always rad-based)
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    }

    if st.session_state.angle_mode == "DEG":
        trig_locals = {
            "sin": sind, "cos": cosd, "tan": tand,
            "asin": asind, "acos": acosd, "atan": atand,
        }
    else:
        trig_locals = {
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        }

    locals_dict = {**common_locals, **trig_locals}

    try:
        sym = sp.sympify(expr, locals=locals_dict, convert_xor=True)
        # Try to simplify a bit
        sym_simpl = sp.simplify(sym)
        return sym_simpl
    except Exception as e:
        return f"Error: {str(e)}"

def push_history(expr: str, result_str: str):
    st.session_state.history.insert(0, {"expr": expr, "result": result_str})
    # Keep last 50 entries
    st.session_state.history = st.session_state.history[:50]

def show_result(sym_val) -> str:
    """Return pretty string and render LaTeX."""
    if isinstance(sym_val, str):
        st.error(sym_val)
        return sym_val
    # Numeric try
    try:
        num = sp.N(sym_val, 16)
    except Exception:
        num = sym_val

    # LaTeX pretty
    st.latex(sp.latex(sp.simplify(sym_val)))
    return str(num)

def plot_if_symbolic(sym_val):
    """If expression contains x, draw a plot."""
    if isinstance(sym_val, str):
        return
    free = getattr(sym_val, "free_symbols", set())
    if x in free:
        st.subheader("Plot")
        colA, colB = st.columns(2)
        with colA:
            xmin = st.number_input("x min", value=-10.0)
        with colB:
            xmax = st.number_input("x max", value=10.0)
        if xmax <= xmin:
            st.info("Set xmax greater than xmin to plot.")
            return
        try:
            f = sp.lambdify(x, sym_val, "numpy")
            xs = np.linspace(xmin, xmax, 600)
            ys = f(xs)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(xs, ys)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.grid(True, which="both", alpha=0.3)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"Could not plot: {e}")

# ---------- Display / Input ----------
display_input = st.text_input("Expression", value=st.session_state.display, 
                              key="display_input", 
                              placeholder="Type or use the buttons… e.g., sin(30)+sqrt(2)^2, 5!, log10(100), x^2+2*x+1")

# Update session state when input changes
if display_input != st.session_state.display:
    st.session_state.display = display_input

# Process button clicks
if st.session_state.button_clicked:
    button_action = st.session_state.button_clicked
    if button_action == "backspace":
        st.session_state.display = st.session_state.display[:-1]
    elif button_action == "clear_all":
        st.session_state.display = ""
    elif button_action == "set_ans":
        st.session_state.display += "Ans"
    elif button_action == "mem_recall":
        st.session_state.display += str(st.session_state.memory)
    elif button_action == "mem_clear":
        st.session_state.memory = 0.0
    elif button_action == "mem_plus":
        try:
            val = float(evaluate(st.session_state.display))
            st.session_state.memory += val
        except Exception:
            st.warning("Cannot M+ current expression.")
    elif button_action == "mem_minus":
        try:
            val = float(evaluate(st.session_state.display))
            st.session_state.memory -= val
        except Exception:
            st.warning("Cannot M- current expression.")
    elif button_action == "equals":
        expr = st.session_state.display.strip()
        res = evaluate(expr)
        out = show_result(res)
        if not isinstance(res, str):
            st.session_state.ans = out
            push_history(expr, out)
    elif button_action == "sign_change":
        if st.session_state.display.startswith("-"):
            st.session_state.display = st.session_state.display[1:]
        else:
            st.session_state.display = "-" + st.session_state.display
    else:
        # Regular button that adds text
        if button_action == "div": 
            st.session_state.display += "/"
        elif button_action == "mult": 
            st.session_state.display += "*"
        elif button_action == "pi": 
            st.session_state.display += "pi"
        elif button_action == "square": 
            st.session_state.display += "^2"
        elif button_action == "factorial": 
            st.session_state.display += "!"
        else:
            st.session_state.display += button_action
    
    # Reset the button click
    st.session_state.button_clicked = None
    st.rerun()

# Buttons grid
def create_button_callback(action):
    def callback():
        st.session_state.button_clicked = action
    return callback

# Row 1: Memory + utility
r1 = st.columns(6)
for idx, (label, action) in enumerate([
    ("MC", "mem_clear"), ("MR", "mem_recall"), ("M+", "mem_plus"),
    ("M-", "mem_minus"), ("Ans", "set_ans"), ("C", "clear_all")
]):
    if r1[idx].button(label, on_click=create_button_callback(action)):
        pass

# Row 2: trig/log
r2 = st.columns(7)
for idx, token in enumerate(["sin(", "cos(", "tan(", "asin(", "acos(", "atan(", "ln("]):
    if r2[idx].button(token, on_click=create_button_callback(token)):
        pass

# Row 3: more funcs
r3 = st.columns(7)
for idx, token in enumerate(["log10(", "sqrt(", "(", ")", "^", "factorial", "%"]):
    if r3[idx].button(token, on_click=create_button_callback(token)):
        pass

# Row 4-6: digits and ops
rows = [
    ["7", "8", "9", "÷", "×", "π", "e"],
    ["4", "5", "6", "-", ",", "x", "^2"],
    ["1", "2", "3", "+", ".", "±", "⌫"],
]
for row in rows:
    cols = st.columns(7)
    for i, key in enumerate(row):
        if key == "÷":
            if cols[i].button(key, on_click=create_button_callback("div")):
                pass
        elif key == "×":
            if cols[i].button(key, on_click=create_button_callback("mult")):
                pass
        elif key == "π":
            if cols[i].button(key, on_click=create_button_callback("pi")):
                pass
        elif key == "^2":
            if cols[i].button(key, on_click=create_button_callback("square")):
                pass
        elif key == "±":
            if cols[i].button(key, on_click=create_button_callback("sign_change")):
                pass
        elif key == "⌫":
            if cols[i].button(key, on_click=create_button_callback("backspace")):
                pass
        else:
            if cols[i].button(key, on_click=create_button_callback(key)):
                pass

# Equals row
eq_col1, eq_col2 = st.columns([3, 1])
with eq_col2:
    if st.button("=", on_click=create_button_callback("equals")):
        pass

# Quick tips
with eq_col1:
    st.info("Tips: `^` is power, `!` is factorial, `%` is percent, `Ans` is last result, use `x` to enable plotting.")

# ---------- History ----------
st.subheader("History")
if st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("Download history (CSV)", csv, "calc_history.csv", "text/csv")
else:
    st.caption("No history yet. Calculate something!")

# ---------- Footer ----------
st.divider()
st.caption("Made with ❤️ using Streamlit + SymPy. Degree mode maps sin/cos/tan to degree versions and inverse trig back to degrees.")
