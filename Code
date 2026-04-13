
import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import matplotlib.animation as animation

import matplotlib.patches as mpatches

from scipy.optimize import least_squares

import streamlit as st

import pandas as pd

import io

import tempfile

import os


def forward_kinematics(theta2, E, F, EG, HF, GH, px, py):

    G = E + EG * np.array([np.cos(theta2), np.sin(theta2)])

    d_vec = F - G

    d = np.linalg.norm(d_vec)

    if d < 1e-9 or d > GH + HF or d < abs(GH - HF):

        return None

    a = (GH**2 - HF**2 + d**2) / (2.0 * d)

    h2 = GH**2 - a**2

    if h2 < 0:

        return None

    h = np.sqrt(h2)

    mid = G + a * d_vec / d

    perp = np.array([-d_vec[1], d_vec[0]]) / d

    H = mid - h * perp

    theta3 = np.arctan2(H[1] - G[1], H[0] - G[0])

    I = G + np.array([

        px * np.cos(theta3) - py * np.sin(theta3),

        px * np.sin(theta3) + py * np.cos(theta3),

    ])

    return G, H, I


def check_grashof(sol):

    """

    Grashof condition: s + l <= p + q

    where s = shortest link, l = longest link, p and q = the other two.

    The four links are EG, GH, HF, and the ground link EF.

    Returns (is_grashof, details_dict).

    """

    EF = float(np.linalg.norm(sol['F'] - sol['E']))

    links = {

        'EG': float(sol['EG']),

        'GH': float(sol['GH']),

        'HF': float(sol['HF']),

        'EF': EF,

    }

    sorted_vals = sorted(links.values())

    s = sorted_vals[0]

    l = sorted_vals[3]

    p = sorted_vals[1]

    q = sorted_vals[2]

    s_name = [k for k, v in links.items() if np.isclose(v, s)][0]

    l_name = [k for k, v in links.items() if np.isclose(v, l)][0]

    lhs = s + l

    rhs = p + q

    is_grashof = lhs <= rhs

    return is_grashof, {

        'links': links,

        's': s, 's_name': s_name,

        'l': l, 'l_name': l_name,

        'p': p, 'q': q,

        'lhs': lhs, 'rhs': rhs,

        'excess': lhs - rhs,

    }

def residuals(params, prescribed_pts):

    E = np.array([params[0], params[1]])

    F = np.array([params[2], params[3]])

    EG, HF, GH = params[4], params[5], params[6]

    px, py = params[7], params[8]

    res = []

    for P, theta in zip(prescribed_pts, params[9:13]):

        r = forward_kinematics(theta, E, F, EG, HF, GH, px, py)

        if r is None:

            res.extend([1e4, 1e4])

        else:

            res.extend([r[2][0] - P[0], r[2][1] - P[1]])

    return np.array(res)

def _run_synthesis(points, n_restarts, seed):

    """Single synthesis attempt. Returns solution dict or None."""

    rng = np.random.default_rng(seed)

    pts = np.array(points, dtype=float)

    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()

    avg_r = np.mean(np.linalg.norm(pts - [cx, cy], axis=1))

    spread = max(avg_r, 60)

    canvas = max(spread * 6, 800)

    lo = [cx-canvas, cy-canvas, cx-canvas, cy-canvas,

          5, 5, 5, -canvas, -canvas,

          -np.pi, -np.pi, -np.pi, -np.pi]

    hi = [cx+canvas, cy+canvas, cx+canvas, cy+canvas,

          canvas, canvas, canvas, canvas, canvas,

          np.pi, np.pi, np.pi, np.pi]

    best_result, best_cost = None, np.inf

    for _ in range(n_restarts):

        angle = rng.uniform(0, 2 * np.pi)

        dist = rng.uniform(0.3, 2.5) * spread

        sp = rng.uniform(0.3, 1.5) * spread

        py_sign = rng.choice([-1, 1])

        x0 = [

            cx + dist * np.cos(angle) - sp, cy + dist * np.sin(angle),

            cx + dist * np.cos(angle) + sp, cy + dist * np.sin(angle),

            rng.uniform(0.2, 1.5) * spread,

            rng.uniform(0.2, 1.5) * spread,

            rng.uniform(0.4, 2.0) * spread,

            rng.uniform(-0.6, 0.6) * spread,

            py_sign * rng.uniform(0.2, 1.5) * spread,

            *rng.uniform(-np.pi, np.pi, 4),

        ]

        try:

            res = least_squares(

                residuals, x0, args=(pts,), bounds=(lo, hi),

                method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12,

                max_nfev=8000,

            )

            if res.cost < best_cost:

                best_cost, best_result = res.cost, res

        except Exception:

            pass

    if best_result is None:

        return None

    p = best_result.x

    E = np.array([p[0], p[1]])

    F = np.array([p[2], p[3]])

    EG, HF, GH = p[4], p[5], p[6]

    px_, py_ = p[7], p[8]

    thetas = p[9:13]

    GI = np.sqrt(px_**2 + py_**2)

    IH = np.sqrt((px_ - GH)**2 + py_**2)

    rms = np.sqrt(best_cost * 2 / 4)

    return {

        'E': E, 'F': F,

        'EG': EG, 'HF': HF, 'GH': GH,

        'px': px_, 'py': py_,

        'GI': GI, 'IH': IH,

        'thetas': thetas,

        'rms_error': rms,

        'params': p,

    }

def synthesize_with_grashof(points, n_restarts=200, seed=42, max_grashof_retries=5):

    """

    Synthesise a four-bar linkage and verify the Grashof condition.

    Retries with different seeds up to max_grashof_retries times.

    Returns (sol, grashof_ok, grashof_details, attempts_made).

    """

    last_sol = None

    last_details = None

    for attempt in range(max_grashof_retries):

        current_seed = seed + attempt * 17

        sol = _run_synthesis(points, n_restarts, current_seed)

        if sol is None:

            continue

        if sol['rms_error'] > 1.0:

            continue

        is_grashof, details = check_grashof(sol)

        last_sol = sol

        last_details = details

        if is_grashof:

            return sol, True, details, attempt + 1

    # All retries exhausted

    if last_sol is not None:

        return last_sol, False, last_details, max_grashof_retries

    return None, False, None, max_grashof_retries

def verify(points, sol):

    errors = []

    for P, theta in zip(np.array(points), sol['thetas']):

        r = forward_kinematics(

            theta, sol['E'], sol['F'],

            sol['EG'], sol['HF'], sol['GH'],

            sol['px'], sol['py'],

        )

        errors.append(np.inf if r is None else float(np.linalg.norm(r[2] - np.array(P))))

    return errors

# ─────────────────────────────────────────────────────────────────────────────

# Animation builder

# ─────────────────────────────────────────────────────────────────────────────

def build_animation(points, sol, n_frames=120, fps=30):

    pts = np.array(points)

    E, F = sol['E'], sol['F']

    EG, HF, GH = sol['EG'], sol['HF'], sol['GH']

    px_, py_ = sol['px'], sol['py']

    curve_pts = []

    for i in range(720):

        r = forward_kinematics(i * np.pi / 360, E, F, EG, HF, GH, px_, py_)

        if r:

            curve_pts.append(r[2])

    frames = []

    for theta in np.linspace(0, 2 * np.pi, n_frames + 1)[:-1]:

        r = forward_kinematics(theta, E, F, EG, HF, GH, px_, py_)

        if r:

            frames.append((theta, r[0], r[1], r[2]))

    if not frames:

        return None

    all_x = [E[0], F[0]] + [p[0] for p in pts]

    all_y = [E[1], F[1]] + [p[1] for p in pts]

    if curve_pts:

        all_x += [p[0] for p in curve_pts]

        all_y += [p[1] for p in curve_pts]

    for _, G, H, I in frames:

        all_x += [G[0], H[0], I[0]]

        all_y += [G[1], H[1], I[1]]

    pad = (max(all_x) - min(all_x) + max(all_y) - min(all_y)) * 0.08 + 20

    xlim = (min(all_x) - pad, max(all_x) + pad)

    ylim = (min(all_y) - pad, max(all_y) + pad)

    ORANGE = "#3404AB"

    BLUE = "#0EB02C"

    GREEN = "#FF00EA"

    PURPLE = "#D4FF00"

    AMBER = "#FF0000"

    GRAY = "#000000"

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.set_xlim(xlim); ax.set_ylim(ylim)

    ax.set_aspect('equal')

    ax.set_title('Four-Bar Linkage -- Coupler Path Synthesis', fontsize=13, pad=10)

    ax.grid(True, alpha=0.18, linestyle='--', color='gray')

    ax.set_facecolor('#f9f9f9')

    if curve_pts:

        ax.plot([p[0] for p in curve_pts], [p[1] for p in curve_pts],

                color=AMBER, lw=1.5, ls='--', alpha=0.55, label='Coupler curve', zorder=2)

    ax.plot([E[0], F[0]], [E[1], F[1]], color=GRAY, lw=2, alpha=0.4, zorder=2)

    for pt, lbl in [(E, 'E'), (F, 'F')]:

        ax.scatter(*pt, color=ORANGE, s=150, zorder=9, edgecolors='white', linewidths=1.5)

        ax.annotate(lbl, pt, textcoords='offset points', xytext=(8, 6),

                    fontsize=12, fontweight='bold', color=ORANGE)

    for k, P in enumerate(pts):

        ax.scatter(*P, color=PURPLE, s=130, zorder=10, edgecolors='white', linewidths=1.5)

        ax.annotate('ABCD'[k], P, textcoords='offset points', xytext=(8, 6),

                    fontsize=13, fontweight='bold', color=PURPLE)

    line_EG, = ax.plot([], [], color=ORANGE, lw=3, zorder=5, solid_capstyle='round')

    line_GH, = ax.plot([], [], color=GRAY, lw=2.5, zorder=5, solid_capstyle='round')

    line_HF, = ax.plot([], [], color=ORANGE, lw=3, zorder=5, solid_capstyle='round')

    line_GI, = ax.plot([], [], color=GREEN, lw=1.5, ls=':', zorder=4)

    line_HI, = ax.plot([], [], color=GREEN, lw=1.5, ls=':', zorder=4)

    trail_line, = ax.plot([], [], color=AMBER, lw=2.2, zorder=3, alpha=0.9)

    dot_G = ax.scatter([], [], color=BLUE, s=70, zorder=8, edgecolors='white', linewidths=1.2)

    dot_H = ax.scatter([], [], color=BLUE, s=70, zorder=8, edgecolors='white', linewidths=1.2)

    dot_I = ax.scatter([], [], color=GREEN, s=100, zorder=8, edgecolors='white', linewidths=1.5)

    lbl_G = ax.text(0, 0, 'G', fontsize=9, color=BLUE, fontweight='bold', zorder=11)

    lbl_H = ax.text(0, 0, 'H', fontsize=9, color=BLUE, fontweight='bold', zorder=11)

    lbl_I = ax.text(0, 0, 'I', fontsize=10, color=GREEN, fontweight='bold', zorder=11)

    legend_items = [

        mpatches.Patch(color=PURPLE, label='Target points A to D'),

        mpatches.Patch(color=ORANGE, label='Fixed pivots E, F and cranks EG, HF'),

        mpatches.Patch(color=GRAY, label='Coupler GH'),

        mpatches.Patch(color=GREEN, label='Coupler point I'),

        mpatches.Patch(color=AMBER, label='Coupler curve'),

    ]

    ax.legend(handles=legend_items, loc='upper right', fontsize=8,

              framealpha=0.85, edgecolor='#ccc')

    trail_x, trail_y = [], []

    def init():

        line_EG.set_data([], []); line_GH.set_data([], [])

        line_HF.set_data([], []); line_GI.set_data([], [])

        line_HI.set_data([], []); trail_line.set_data([], [])

        dot_G.set_offsets(np.empty((0, 2)))

        dot_H.set_offsets(np.empty((0, 2)))

        dot_I.set_offsets(np.empty((0, 2)))

        return (line_EG, line_GH, line_HF, line_GI, line_HI,

                trail_line, dot_G, dot_H, dot_I, lbl_G, lbl_H, lbl_I)

    def update(idx):

        _, G, H, I = frames[idx]

        line_EG.set_data([E[0], G[0]], [E[1], G[1]])

        line_GH.set_data([G[0], H[0]], [G[1], H[1]])

        line_HF.set_data([H[0], F[0]], [H[1], F[1]])

        line_GI.set_data([G[0], I[0]], [G[1], I[1]])

        line_HI.set_data([H[0], I[0]], [H[1], I[1]])

        trail_x.append(I[0]); trail_y.append(I[1])

        trail_line.set_data(trail_x, trail_y)

        dot_G.set_offsets([[G[0], G[1]]])

        dot_H.set_offsets([[H[0], H[1]]])

        dot_I.set_offsets([[I[0], I[1]]])

        lbl_G.set_position((G[0] + 6, G[1] + 6))

        lbl_H.set_position((H[0] + 6, H[1] + 6))

        lbl_I.set_position((I[0] + 7, I[1] + 7))

        return (line_EG, line_GH, line_HF, line_GI, line_HI,

                trail_line, dot_G, dot_H, dot_I, lbl_G, lbl_H, lbl_I)

    ani = animation.FuncAnimation(

        fig, update, frames=len(frames),

        init_func=init, interval=1000 // fps, blit=True,

    )

    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tf:

        tmp_path = tf.name

    try:

        ani.save(tmp_path, writer='pillow', fps=fps)

        plt.close(fig)

        with open(tmp_path, 'rb') as f:

            buf = io.BytesIO(f.read())

        buf.seek(0)

    finally:

        if os.path.exists(tmp_path):

            os.unlink(tmp_path)

    return buf

# ─────────────────────────────────────────────────────────────────────────────

# Streamlit UI

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Four-Bar Linkage Synthesizer",
    layout="wide",
)

st.title("ME261: THEORY OF MECHANISMS AND MACHINES")

st.title("4-point coupler curve synthesis")

# ✅ CSS THEME
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #0e5f2c;
        color: white;
    }

    /* Text color */
    html, body, [class*="css"] {
        color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b4d24;
        color: white;
    }

    /* Inputs and widgets */
    .stNumberInput input, .stTextInput input {
        background-color: #1f7a3a;
        color: white;
    }

    /* Buttons */
    .stButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 8px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #2ecc71;
        color: white;
    }

    /* Dataframes */
    .stDataFrame {
        background-color: #145a32;
        color: white;
    }

    /* Tables */
    table {
        color: white !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ DESCRIPTION TEXT (separate call)
st.markdown(
    "Enter **4 coupler-point positions** (A, B, C, D). "
    "The project can help in finding a four-bar linkage whose tracer point **I** passes through all four points given by user, "
    "verifies the Grashof condition, and animates the mechanism."
)
# ── Sidebar inputs ────────────────────────────────────────────────────────────

# ── Main Panel Inputs (replacing sidebar) ─────────────────────────────────────
st.markdown("###  Enter 4 Coordinates (A, B, C, D)")

st.markdown("Enter coordinates as: x,y (one point per line)")
st.markdown("Set the four point coordinates, then just Animate...")
default_text = """200,180
370,140
490,230
310,310"""

user_input = st.text_area(
    "Points (A, B, C, D)",
    value=default_text,
    height=150
)

points = []

try:
    lines = user_input.strip().split("\n")
    for line in lines:
        x_str, y_str = line.split(",")
        points.append([float(x_str.strip()), float(y_str.strip())])
    
    if len(points) != 4:
        st.warning("Please enter exactly 4 points (A, B, C, D).")
        st.stop()

except:
    st.error("Invalid format. Use: x,y per line.")
    st.stop()
    
st.divider()

# 🔒 Hidden solver + animation settings (fixed internally)
n_restarts = 200
max_retries = 5
seed = 42

fps = 30
n_frames = 120
run = st.button("Animate", use_container_width=True)
# ── Static layout ─────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    pass
    

# with col_right:

#     if not run:

#         st.info("Set the four point coordinates, then just Animate...")


if run:

    with st.spinner(f"Synthesising"

                    f"(Pls Wait)…"):

        sol, grashof_ok, grashof_details, attempts = synthesize_with_grashof(

            points,

            n_restarts=int(n_restarts),

            seed=int(seed),

            max_grashof_retries=int(max_retries),

        )


    if sol is None:

        st.error("Optimisation failed entirely. Try different point positions or increase restarts.")

        st.stop()

    # Grashof condition failed

    if not grashof_ok:

        st.error("No valid crank-rocker linkage found..."

                 f"after {attempts} attempt(s).")

        d = grashof_details

        st.markdown(f"""

**Why it failed:**

The Grashof condition requires:

**Shortest link + Longest link ≤ Sum of the other two links**
For the synthesised linkage:

| Link | Length |

|------|--------|

| EG | {d['links']['EG']:.4f} |

| GH | {d['links']['GH']:.4f} |

| HF | {d['links']['HF']:.4f} |

| EF (ground) | {d['links']['EF']:.4f} |

Shortest link **{d['s_name']}** = {d['s']:.4f}

Longest link **{d['l_name']}** = {d['l']:.4f}

**{d['s_name']} + {d['l_name']} = {d['lhs']:.4f}** vs sum of remaining two = **{d['rhs']:.4f}**

{d['lhs']:.4f} > {d['rhs']:.4f} (exceeds by **{d['excess']:.4f}**)

Since no link can make a full 360° rotation, this linkage cannot continuously

visit all four prescribed positions in a single rotation.

**Suggestions:**

- Try placing the four points closer together

- Increase the number of Grashof retries in the sidebar

- Try a different random seed

""")

        st.stop()

    # Solution found and Grashof satisfied

    st.success(f"LINKAGE FOUND....")

    errors = verify(points, sol)

    rms = sol['rms_error']

    with col_left:

        st.subheader("IMP Values")

        res_df = pd.DataFrame({

            "Parameter": ["x_E", "y_E", "x_F", "y_F", "EG", "GI", "IH", "HG", "HF"],

            "Value": [f"{v:.5f}" for v in [

                sol['E'][0], sol['E'][1],

                sol['F'][0], sol['F'][1],

                sol['EG'], sol['GI'], sol['IH'],

                sol['GH'], sol['HF'],

            ]],

        })

        st.dataframe(res_df, use_container_width=True, hide_index=True)

        st.subheader("Grashof Check")

        d = grashof_details

        grashof_df = pd.DataFrame({

            "Link": list(d['links'].keys()),

            "Length": [f"{v:.4f}" for v in d['links'].values()],

        })

        st.dataframe(grashof_df, use_container_width=True, hide_index=True)

        st.markdown(

            f"**{d['s_name']} + {d['l_name']} = {d['lhs']:.4f} "

            f"≤ {d['rhs']:.4f}** -- condition satisfied ✅"

        )


    with col_right:

        st.subheader("Animation")

        with st.spinner("Rendering animation…"):

            gif_buf = build_animation(points, sol, n_frames=int(n_frames), fps=int(fps))

        if gif_buf is None:

            st.warning("Could not render animation -- linkage may not have a full rotation range.")

        else:

            st.image(gif_buf, caption="Four-bar linkage in motion -- coupler point I traces the path",

                     use_container_width=True)

            

