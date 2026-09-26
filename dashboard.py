"""
Mediscope Insight — Single Page Dynamic Dashboard
Run:  python dashboard.py
Open: http://127.0.0.1:8050
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import warnings; warnings.filterwarnings("ignore")

# ── Load & prep ────────────────────────────────────────────────────────────────
df = pd.read_csv("healthcare_normalized.csv")
df["Visit_Date"] = pd.to_datetime(df["Visit_Date"])
df["YearMonth"]  = df["Visit_Date"].dt.to_period("M").astype(str)
df["Age_Years"]  = (df["Age"] * 70 + 1).round().astype(int)
df["Age_Group"]  = pd.cut(df["Age_Years"], bins=[0,14,28,42,56,100],
                           labels=["0–14","15–28","29–42","43–56","57+"])
for c in ["Diabetes","Hypertension","Heart_Disease","Kidney_Disease"]:
    df[c] = df[c].astype(str).str.strip().str.lower() == "true"

# ── Design tokens ──────────────────────────────────────────────────────────────
BG      = "#0D1117"
SURFACE = "#13181F"
CARD    = "#161B24"
BORDER  = "#21293A"
TEXT    = "#E2E8F0"
MUTED   = "#64748B"
SUB     = "#94A3B8"
TEAL    = "#10B981"
TEAL2   = "#059669"
RED     = "#F43F5E"
AMBER   = "#F59E0B"
BLUE    = "#3B82F6"
INDIGO  = "#6366F1"
PINK    = "#EC4899"
GRAY    = "#475569"
PAL     = [TEAL, RED, AMBER, BLUE, INDIGO, PINK, "#34D399", "#FCA5A5"]

FONT = "'Inter','Space Grotesk',system-ui,sans-serif"

# shared axis style
AX = dict(
    gridcolor="rgba(255,255,255,0.05)",
    linecolor=BORDER,
    tickfont=dict(color=SUB, size=10, family=FONT),
    title_font=dict(color=MUTED, size=11, family=FONT),
    zeroline=False,
)

def base_layout(**kw):
    d = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=11),
        margin=dict(l=48, r=20, t=42, b=44),
        hoverlabel=dict(
            bgcolor="#1E2A3A", bordercolor=BORDER,
            font=dict(family=FONT, size=12, color=TEXT)
        ),
    )
    d.update(kw)
    return d

# ── Pre-aggregations ───────────────────────────────────────────────────────────
monthly      = df.groupby("YearMonth").size().reset_index(name="n").sort_values("YearMonth")
m_labels     = [x.replace("-","/") for x in monthly["YearMonth"]]

dept_c       = df["Department"].value_counts()
ecg_c        = df["ECG_Result"].value_counts()
blood_c      = df["Blood_Group"].value_counts().sort_index()
age_c        = df["Age_Group"].value_counts().sort_index()
monthly_hr   = df.groupby("YearMonth")["Heart_Rate"].mean().reset_index().sort_values("YearMonth")
monthly_sbp  = df.groupby("YearMonth")["Systolic_BP"].mean().reset_index().sort_values("YearMonth")
t_labels     = [x.replace("-","/") for x in monthly_hr["YearMonth"]]

NUMERIC_COLS = [
    "BMI","Age","Systolic_BP","Diastolic_BP","Heart_Rate","SpO2",
    "Hemoglobin","Total_Cholesterol","HbA1c","Fasting_Sugar",
    "LDL","HDL","Triglycerides","Creatinine",
]

# ── Time period filtering ──────────────────────────────────────────────────────
from datetime import timedelta

def filter_by_period(data, period="1m"):
    """Filter dataframe by time period: 1w, 1m, 3m, 6m, 1y, all"""
    if period == "all":
        return data
    max_date = data["Visit_Date"].max()
    if period == "1w":
        cutoff = max_date - timedelta(days=7)
    elif period == "1m":
        cutoff = max_date - timedelta(days=30)
    elif period == "3m":
        cutoff = max_date - timedelta(days=90)
    elif period == "6m":
        cutoff = max_date - timedelta(days=180)
    elif period == "1y":
        cutoff = max_date - timedelta(days=365)
    else:
        cutoff = max_date - timedelta(days=30)
    return data[data["Visit_Date"] >= cutoff].copy()

# ── Chart builders ─────────────────────────────────────────────────────────────

def visits_fig():
    # gradient-ish by using opacity ramp on last bar
    colors = [TEAL]*len(monthly)
    colors[-1] = "rgba(16,185,129,0.45)"   # current partial month dimmed
    fig = go.Figure()
    fig.add_bar(
        x=m_labels, y=monthly["n"],
        marker=dict(color=colors, line_width=0,
                    cornerradius=4),
        name="Visits",
        hovertemplate="<b>%{x}</b><br>%{y:,} visits<extra></extra>",
    )
    # reference average line
    avg = monthly["n"].iloc[:-1].mean()
    fig.add_hline(y=avg, line=dict(color=AMBER, width=1.2, dash="dot"),
                  annotation_text=f"  avg {avg:.0f}",
                  annotation_font=dict(color=AMBER, size=10))
    fig.update_layout(
        **base_layout(height=230, margin=dict(l=44, r=20, t=36, b=40)),
        title=dict(text="Monthly Patient Visits", font=dict(size=13, color=TEXT, weight=600), x=0, pad_l=0),
        bargap=0.28,
        xaxis=dict(**AX, tickangle=-30),
        yaxis=dict(**AX, title="Visits"),
    )
    return fig


def dept_fig():
    labels = dept_c.index.tolist()
    values = dept_c.values.tolist()
    colors = [TEAL, RED, BLUE, AMBER]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.68,
        marker=dict(colors=colors, line=dict(color=CARD, width=2)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:,} patients — %{percent}<extra></extra>",
        sort=False,
    ))
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:10px;color:{MUTED}'>patients</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=TEXT, family=FONT),
        align="center",
    )
    fig.update_layout(
        **base_layout(height=230, margin=dict(l=10, r=10, t=36, b=10)),
        title=dict(text="Departments", font=dict(size=13, color=TEXT, weight=600), x=0),
        showlegend=True,
        legend=dict(
            orientation="v", x=1.02, y=0.5, xanchor="left",
            bgcolor="rgba(0,0,0,0)", font=dict(color=SUB, size=10, family=FONT),
        ),
    )
    return fig


def ecg_fig():
    ecg_map = {"Normal": TEAL, "Arrhythmia": RED, "Ischemia": AMBER, "Abnormal ECG": BLUE}
    labels = ecg_c.index.tolist()
    values = ecg_c.values.tolist()
    colors = [ecg_map.get(e, GRAY) for e in labels]
    pcts   = [v/sum(values)*100 for v in values]
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker=dict(color=colors, line_width=0, cornerradius=4),
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
        textfont=dict(color=SUB, size=10, family=FONT),
        hovertemplate="<b>%{y}</b><br>%{x:,} patients<extra></extra>",
    ))
    fig.update_layout(
        **base_layout(height=230, margin=dict(l=100, r=60, t=36, b=36)),
        title=dict(text="ECG Findings", font=dict(size=13, color=TEXT, weight=600), x=0),
        xaxis=dict(**AX, title="Patients"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=BORDER, zeroline=False,
                   tickfont=dict(color=TEXT, size=11, family=FONT),
                   title_font=dict(color=MUTED, size=11, family=FONT)),
    )
    return fig


def diag_fig():
    diag_top = df["Diagnosis"].value_counts().head(5)
    other_n  = len(df) - diag_top.sum()
    labels   = [l[:22]+"…" if len(l)>22 else l for l in diag_top.index] + ["Other"]
    values   = list(diag_top.values) + [other_n]
    colors   = [TEAL, RED, AMBER, BLUE, INDIGO, GRAY]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.60,
        marker=dict(colors=colors, line=dict(color=CARD, width=2)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:,} — %{percent}<extra></extra>",
        sort=False,
    ))
    fig.add_annotation(
        text=f"<b>6,466</b><br><span style='font-size:10px'>total</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color=TEXT, family=FONT), align="center",
    )
    fig.update_layout(
        **base_layout(height=230, margin=dict(l=10, r=10, t=36, b=10)),
        title=dict(text="Diagnosis Split", font=dict(size=13, color=TEXT, weight=600), x=0),
        showlegend=True,
        legend=dict(
            orientation="v", x=1.0, y=0.5, xanchor="left",
            bgcolor="rgba(0,0,0,0)", font=dict(color=SUB, size=9, family=FONT),
        ),
    )
    return fig


def age_fig():
    ages   = age_c.index.astype(str).tolist()
    counts = age_c.values.tolist()
    ramp   = ["#34D399","#10B981","#059669","#047857","#064E3B"]
    fig = go.Figure(go.Bar(
        x=ages, y=counts,
        marker=dict(color=ramp, line_width=0, cornerradius=4),
        hovertemplate="<b>Age %{x}</b><br>%{y:,} patients<extra></extra>",
    ))
    fig.update_layout(
        **base_layout(height=230, margin=dict(l=44, r=20, t=36, b=36)),
        title=dict(text="Age Distribution", font=dict(size=13, color=TEXT, weight=600), x=0),
        bargap=0.32,
        xaxis=dict(**AX),
        yaxis=dict(**AX, title="Patients"),
    )
    return fig


def blood_fig():
    blood_colors = {
        "A+": TEAL,  "A-": "#34D399",
        "B+": BLUE,  "B-": "#93C5FD",
        "AB+": INDIGO,"AB-": "#A5B4FC",
        "O+": RED,   "O-": "#FCA5A5",
    }
    labels = blood_c.index.tolist()
    colors = [blood_colors.get(b, GRAY) for b in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=blood_c.values,
        marker=dict(color=colors, line_width=0, cornerradius=4),
        hovertemplate="<b>%{x}</b>: %{y:,} patients<extra></extra>",
    ))
    fig.update_layout(
        **base_layout(height=230, margin=dict(l=44, r=20, t=36, b=36)),
        title=dict(text="Blood Group Distribution", font=dict(size=13, color=TEXT, weight=600), x=0),
        bargap=0.3,
        xaxis=dict(**AX),
        yaxis=dict(**AX, title="Patients", range=[700, 900]),
    )
    return fig


def vitals_fig(metric="Heart_Rate"):
    opts = {
        "Heart_Rate":  (monthly_hr,  "Heart_Rate",  TEAL,   "Heart Rate (bpm)"),
        "Systolic_BP": (monthly_sbp, "Systolic_BP", RED,    "Systolic BP (mmHg)"),
    }
    data_df, col, color, label = opts.get(metric, opts["Heart_Rate"])
    y    = data_df[col].values
    r, g, b_ch = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
    fig  = go.Figure()
    fig.add_scatter(
        x=t_labels, y=y, mode="lines+markers",
        line=dict(color=color, width=3.5, shape="spline"),
        marker=dict(size=8, color=color, line=dict(width=2, color=CARD)),
        fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b_ch},0.18)",
        hovertemplate="<b>%{x}</b><br>%{y:.1f}<extra></extra>",
        name=label,
    )
    fig.update_layout(
        **base_layout(height=340, margin=dict(l=60, r=30, t=50, b=60)),
        title=dict(text=f"Monthly Avg — {label}", font=dict(size=14, color=TEXT, weight=600), x=0),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor=BORDER, zeroline=False,
                   tickangle=-35, tickfont=dict(color=SUB, size=10, family=FONT),
                   title_font=dict(color=MUTED, size=12, family=FONT), nticks=9),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=BORDER, 
                   tickfont=dict(color=SUB, size=11, family=FONT),
                   title_font=dict(color=MUTED, size=12, family=FONT), title=label),
        showlegend=False,
    )
    return fig


def activity_fig():
    """Physical activity donut."""
    act_c = df["Physical_Activity"].value_counts()
    act_colors = {"High": TEAL, "Moderate": AMBER, "Low": RED}
    colors = [act_colors.get(a, GRAY) for a in act_c.index]
    fig = go.Figure(go.Pie(
        labels=act_c.index, values=act_c.values, hole=0.55,
        marker=dict(colors=colors, line=dict(color=CARD, width=3)),
        textinfo="label+percent", textposition="inside",
        textfont=dict(color=TEXT, size=13, family=FONT, weight=600),
        hovertemplate="<b>%{label}</b><br>%{value:,} patients — %{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>6,466</b>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=MUTED, family=FONT), align="center",
    )
    fig.update_layout(
        **base_layout(height=380, margin=dict(l=20, r=20, t=50, b=20)),
        title=dict(text="Physical Activity", font=dict(size=14, color=TEXT, weight=600), x=0.02),
        showlegend=False,
    )
    return fig


def risk_fig():
    """Risk behaviours horizontal bar."""
    s_count = df["Smoking"].eq("Yes").sum()
    a_count = df["Alcohol"].eq("Yes").sum()
    s_pct = round(s_count / len(df) * 100, 1)
    a_pct = round(a_count / len(df) * 100, 1)
    fig = go.Figure(go.Bar(
        x=[s_pct, a_pct], y=["Smoking", "Alcohol Use"], orientation="h",
        marker=dict(color=[RED, AMBER], line_width=2, line_color=BORDER, cornerradius=6),
        text=[f"{s_pct}%  •  {s_count:,}", f"{a_pct}%  •  {a_count:,}"],
        textposition="outside",
        textfont=dict(color=SUB, size=13, family=FONT),
        hovertemplate="<b>%{y}</b><br>%{x}% of patients (%{text})<extra></extra>",
        width=0.7,
    ))
    fig.update_layout(
        **base_layout(height=380, margin=dict(l=130, r=100, t=50, b=30)),
        title=dict(text="Risk Behaviours", font=dict(size=14, color=TEXT, weight=600), x=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=BORDER, zeroline=False,
                   title="% of patients", range=[0, 50],
                   tickfont=dict(color=SUB, size=12, family=FONT),
                   title_font=dict(color=MUTED, size=12, family=FONT)),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=BORDER, zeroline=False,
                   tickfont=dict(color=TEXT, size=14, family=FONT, weight=600)),
        showlegend=False,
    )
    return fig


def visits_trend_fig(period="1m"):
    """Daily visit trends."""
    filtered_df = filter_by_period(df, period)
    daily = filtered_df.groupby(filtered_df["Visit_Date"].dt.date).size().reset_index(name="visits")
    daily.columns = ["Date", "visits"]
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date")
    
    fig = go.Figure()
    fig.add_scatter(
        x=daily["Date"], y=daily["visits"], mode="lines+markers", name="Daily Visits",
        line=dict(color=TEAL, width=2.5, shape="spline"),
        marker=dict(size=6, color=TEAL, line=dict(width=1, color=CARD)),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.12)",
        hovertemplate="<b>%{x|%b %d}</b><br>%{y} visits<extra></extra>",
    )
    avg = daily["visits"].mean()
    fig.add_hline(y=avg, line=dict(color=AMBER, width=1, dash="dot"),
                  annotation_text=f"avg {avg:.0f}", annotation_font=dict(color=AMBER, size=9))
    
    fig.update_layout(
        **base_layout(height=300, margin=dict(l=50, r=20, t=45, b=55)),
        title=dict(text=f"Patient Visits Trend — {period.replace('1', 'Last ').upper()}", 
                   font=dict(size=13, color=TEXT, weight=600), x=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=BORDER, zeroline=False,
                   tickfont=dict(color=SUB, size=9, family=FONT),
                   title_font=dict(color=MUTED, size=11, family=FONT)),
        yaxis=dict(**AX, title="Visits"),
    )
    return fig


def diagnosis_trend_fig(period="1m"):
    """Top 5 diagnosis trends."""
    filtered_df = filter_by_period(df, period)
    top_diag = filtered_df["Diagnosis"].value_counts().head(5).index.tolist()
    
    fig = go.Figure()
    colors_map = [TEAL, RED, AMBER, BLUE, INDIGO]
    
    for idx, diag in enumerate(top_diag):
        diag_data = filtered_df[filtered_df["Diagnosis"] == diag]
        daily = diag_data.groupby(diag_data["Visit_Date"].dt.date).size().reset_index(name="count")
        daily.columns = ["Date", "count"]
        daily["Date"] = pd.to_datetime(daily["Date"])
        daily = daily.sort_values("Date")
        
        fig.add_scatter(
            x=daily["Date"], y=daily["count"], mode="lines", name=diag[:25],
            line=dict(color=colors_map[idx], width=2.5),
            hovertemplate=f"<b>{diag[:25]}</b><br>%{{x|%b %d}}: %{{y}}<extra></extra>",
        )
    
    fig.update_layout(
        **base_layout(height=300, margin=dict(l=50, r=20, t=45, b=55)),
        title=dict(text=f"Top 5 Diagnosis Trends — {period.replace('1', 'Last ').upper()}", 
                   font=dict(size=13, color=TEXT, weight=600), x=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=BORDER, zeroline=False,
                   tickfont=dict(color=SUB, size=9, family=FONT),
                   title_font=dict(color=MUTED, size=11, family=FONT)),
        yaxis=dict(**AX, title="Cases"),
        hovermode="x unified", showlegend=True,
        legend=dict(x=1, y=1, xanchor="right", yanchor="top",
                    bgcolor="rgba(0,0,0,0.2)", font=dict(color=SUB, size=9, family=FONT)),
    )
    return fig


def department_trend_fig(period="1m"):
    """Department-wise patient distribution trend."""
    filtered_df = filter_by_period(df, period)
    
    fig = go.Figure()
    dept_map = {"Cardiology": RED, "General Medicine": TEAL, "Nephrology": BLUE, "Endocrinology": AMBER}
    
    for dept, color in dept_map.items():
        dept_data = filtered_df[filtered_df["Department"] == dept]
        daily = dept_data.groupby(dept_data["Visit_Date"].dt.date).size().reset_index(name="count")
        daily.columns = ["Date", "count"]
        daily["Date"] = pd.to_datetime(daily["Date"])
        daily = daily.sort_values("Date")
        
        fig.add_scatter(
            x=daily["Date"], y=daily["count"], mode="lines", name=dept,
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{dept}</b><br>%{{x|%b %d}}: %{{y}} patients<extra></extra>",
            stackgroup=None,
        )
    
    fig.update_layout(
        **base_layout(height=300, margin=dict(l=50, r=20, t=45, b=55)),
        title=dict(text=f"Department Trends — {period.replace('1', 'Last ').upper()}", 
                   font=dict(size=13, color=TEXT, weight=600), x=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=BORDER, zeroline=False,
                   tickfont=dict(color=SUB, size=9, family=FONT),
                   title_font=dict(color=MUTED, size=11, family=FONT)),
        yaxis=dict(**AX, title="Patient Count"),
        hovermode="x unified", showlegend=True,
        legend=dict(x=1, y=1, xanchor="right", yanchor="top",
                    bgcolor="rgba(0,0,0,0.2)", font=dict(color=SUB, size=9, family=FONT)),
    )
    return fig


def disease_prevalence_fig(period="1m"):
    """Disease prevalence trends."""
    filtered_df = filter_by_period(df, period)
    
    fig = go.Figure()
    diseases = ["Diabetes", "Hypertension", "Heart_Disease", "Kidney_Disease"]
    disease_labels = ["Diabetes", "Hypertension", "Heart Disease", "Kidney Disease"]
    colors_d = [AMBER, RED, PINK, BLUE]
    
    for disease, label, color in zip(diseases, disease_labels, colors_d):
        disease_data = filtered_df[filtered_df[disease] == True]
        daily = disease_data.groupby(disease_data["Visit_Date"].dt.date).size().reset_index(name="count")
        daily.columns = ["Date", "count"]
        daily["Date"] = pd.to_datetime(daily["Date"])
        daily = daily.sort_values("Date")
        
        fig.add_scatter(
            x=daily["Date"], y=daily["count"], mode="lines", name=label,
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{label}</b><br>%{{x|%b %d}}: %{{y}} cases<extra></extra>",
        )
    
    fig.update_layout(
        **base_layout(height=300, margin=dict(l=50, r=20, t=45, b=55)),
        title=dict(text=f"Disease Prevalence Trends — {period.replace('1', 'Last ').upper()}", 
                   font=dict(size=13, color=TEXT, weight=600), x=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=BORDER, zeroline=False,
                   tickfont=dict(color=SUB, size=9, family=FONT),
                   title_font=dict(color=MUTED, size=11, family=FONT)),
        yaxis=dict(**AX, title="Cases"),
        hovermode="x unified", showlegend=True,
        legend=dict(x=1, y=1, xanchor="right", yanchor="top",
                    bgcolor="rgba(0,0,0,0.2)", font=dict(color=SUB, size=9, family=FONT)),
    )
    return fig


def scatter_fig(x_col="BMI", y_col="Systolic_BP", color_by="Department"):
    sample = df.sample(min(800, len(df)), random_state=42)
    dept_c_map   = {"Cardiology": RED, "General Medicine": TEAL,
                    "Nephrology": BLUE, "Endocrinology": AMBER}
    gender_c_map = {"Male": BLUE, "Female": PINK}
    cmap = dept_c_map if color_by == "Department" else gender_c_map
    colors = [cmap.get(v, GRAY) for v in sample[color_by]]

    fig = go.Figure(go.Scatter(
        x=sample[x_col], y=sample[y_col], mode="markers",
        marker=dict(color=colors, size=5, opacity=0.65,
                    line=dict(width=0)),
        hovertemplate=f"<b>{x_col}</b>: %{{x:.3f}}<br><b>{y_col}</b>: %{{y:.3f}}<extra></extra>",
    ))
    # add legend manually
    for label, color in cmap.items():
        fig.add_scatter(x=[None], y=[None], mode="markers",
                        marker=dict(size=8, color=color),
                        name=label, showlegend=True)
    fig.update_layout(
        **base_layout(height=280, margin=dict(l=52, r=20, t=42, b=50)),
        title=dict(text=f"Scatter — {x_col} vs {y_col}",
                   font=dict(size=13, color=TEXT, weight=600), x=0),
        xaxis=dict(**AX, title=x_col),
        yaxis=dict(**AX, title=y_col),
        showlegend=True,
        legend=dict(orientation="h", x=1, y=1, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", font=dict(color=SUB, size=10, family=FONT)),
    )
    return fig


# ── UI helpers ─────────────────────────────────────────────────────────────────

def card(*children, extra=None):
    s = {"background": CARD, "border": f"1px solid {BORDER}",
         "borderRadius": "12px", "padding": "14px 16px",
         "boxShadow": "0 2px 12px rgba(0,0,0,0.35)"}
    if extra: s.update(extra)
    return html.Div(list(children), style=s)


def kpi(label, value, sub, accent):
    return html.Div([
        html.Div(label, style={"fontSize":"10px","color":MUTED,"letterSpacing":"0.8px",
                               "textTransform":"uppercase","marginBottom":"10px","fontWeight":"500"}),
        html.Div(value, style={"fontSize":"26px","fontWeight":"700","color":TEXT,
                               "letterSpacing":"-0.5px","lineHeight":"1"}),
        html.Div(sub,   style={"fontSize":"11px","color":SUB,"marginTop":"6px"}),
    ], style={"background":CARD,"border":f"1px solid {BORDER}","borderRadius":"12px",
              "padding":"18px 20px","borderLeft":f"3px solid {accent}",
              "boxShadow":"0 2px 12px rgba(0,0,0,0.35)"})


def dd(id_, opts, val, width="148px"):
    return dcc.Dropdown(id=id_,
        options=[{"label": o.replace("_"," "), "value": o} for o in opts],
        value=val, clearable=False,
        style={"width": width, "fontSize":"12px",
               "backgroundColor": SURFACE, "color": TEXT,
               "border": f"1px solid {BORDER}", "borderRadius":"6px"},
        className="ms-dd")


def row(*children, cols, gap="12px", mb="12px"):
    return html.Div(list(children),
        style={"display":"grid","gridTemplateColumns":cols,"gap":gap,"marginBottom":mb})


# ── App ────────────────────────────────────────────────────────────────────────

app = Dash(__name__, title="Mediscope Insight",
           meta_tags=[{"name":"viewport","content":"width=device-width,initial-scale=1"}])

app.index_string = """<!DOCTYPE html><html><head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600&display=swap">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: #0D1117; font-family: 'Inter','Space Grotesk',system-ui,sans-serif; color: #E2E8F0; }
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #0D1117; }
  ::-webkit-scrollbar-thumb { background: #21293A; border-radius: 3px; }

  /* Dropdown dark-mode overrides — Dash v2 (react-select) */
  .ms-dd .Select-control,
  .ms-dd .Select__control,
  .ms-dd [class$="-control"]          { background:#13181F !important; border-color:#21293A !important; color:#E2E8F0 !important; border-radius:6px !important; min-height:34px !important; box-shadow:none !important; }
  .ms-dd .Select-control:hover,
  .ms-dd [class$="-control"]:hover    { border-color:#3B82F6 !important; }
  .ms-dd .Select-placeholder,
  .ms-dd [class$="-placeholder"]      { color:#64748B !important; font-size:12px !important; }
  .ms-dd .Select-value-label,
  .ms-dd [class$="-singleValue"]      { color:#E2E8F0 !important; font-size:12px !important; }
  .ms-dd .Select-input input          { color:#E2E8F0 !important; }
  .ms-dd [class$="-Input"] input      { color:#E2E8F0 !important; }
  .ms-dd .Select-arrow-zone,
  .ms-dd [class$="-indicatorContainer"] { color:#64748B !important; }
  .ms-dd .Select-arrow                { border-top-color:#64748B !important; }
  .ms-dd .Select-menu-outer,
  .ms-dd [class$="-menu"]             { background:#13181F !important; border:1px solid #21293A !important; border-radius:8px !important; box-shadow:0 8px 24px rgba(0,0,0,0.6) !important; z-index:9999 !important; }
  .ms-dd [class$="-MenuList"]         { background:#13181F !important; border-radius:8px !important; padding:4px !important; }
  .ms-dd .Select-option,
  .ms-dd [class$="-option"]           { background:#13181F !important; color:#94A3B8 !important; font-size:12px !important; padding:8px 12px !important; border-radius:4px !important; cursor:pointer !important; }
  .ms-dd .Select-option.is-focused,
  .ms-dd [class$="-option"]:hover     { background:#1C2434 !important; color:#E2E8F0 !important; }
  .ms-dd .Select-option.is-selected,
  .ms-dd [class$="-option"][aria-selected="true"] { background:#132A22 !important; color:#10B981 !important; }
  .ms-dd [class$="-IndicatorsContainer"] { background:#13181F !important; border-radius:0 6px 6px 0 !important; }
  .ms-dd [class$="-indicatorSeparator"] { background:#21293A !important; }
</style>
</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

# ── Layout ─────────────────────────────────────────────────────────────────────

GCFG = {"displayModeBar": False, "responsive": True}

app.layout = html.Div([

    # ── Topbar ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("Mediscope ", style={"fontSize":"20px","fontWeight":"700","color":TEXT,"letterSpacing":"-0.3px"}),
            html.Span("Insight",    style={"fontSize":"20px","fontWeight":"700","color":TEAL,"letterSpacing":"-0.3px"}),
            html.Div("",
                     style={"fontSize":"11px","color":MUTED,"marginTop":"2px","letterSpacing":"0.2px"}),
        ]),
        html.Div([
            html.Span("● LIVE", style={
                "fontSize":"11px","color":TEAL,"fontWeight":"600","letterSpacing":"1px",
                "background":"rgba(16,185,129,0.08)","padding":"5px 12px",
                "borderRadius":"20px","border":f"1px solid rgba(16,185,129,0.3)"}),
            html.Span("6,466 patients · 4 departments",
                      style={"fontSize":"11px","color":MUTED,"marginLeft":"14px"}),
        ], style={"display":"flex","alignItems":"center"}),
    ], style={
        "background":SURFACE,"borderBottom":f"1px solid {BORDER}",
        "padding":"16px 28px","display":"flex","justifyContent":"space-between","alignItems":"center",
        "position":"sticky","top":"0","zIndex":"100",
    }),

    # ── Main content ─────────────────────────────────────────────────────────
    html.Div([

        # Row 0 — KPIs
        row(
            kpi("Total Patients",  "6,466", "across 4 departments",        TEAL),
            kpi("Cardiac Cases",   "1,000", "15.5% of all visits",         RED),
            kpi("Hypertension",    "1,633", "25.3% prevalence",            AMBER),
            kpi("ECG Anomalies",   "1,000", "arrhythmia · ischemia",       BLUE),
            kpi("Smoking Rate",    "21.2%", "1,372 patients affected",     RED),
            kpi("Avg Monthly",     "523",   "patient visits per month",    TEAL),
            cols="repeat(6,1fr)",
        ),

        # Row 1 — Visits bar | Dept donut | ECG horizontal bar
        row(
            card(dcc.Graph(figure=visits_fig(), config=GCFG)),
            card(dcc.Graph(figure=dept_fig(),   config=GCFG)),
            card(dcc.Graph(figure=ecg_fig(),    config=GCFG)),
            cols="1.6fr 1fr 1fr",
        ),

        # Row 2 — Diagnosis donut | Age bars | Blood group bars
        row(
            card(dcc.Graph(figure=diag_fig(),  config=GCFG)),
            card(dcc.Graph(figure=age_fig(),   config=GCFG)),
            card(dcc.Graph(figure=blood_fig(), config=GCFG)),
            cols="1fr 1fr 1fr",
        ),

        # Row 3a — Activity | Risk (two columns, wider)
        row(
            card(dcc.Graph(figure=activity_fig(), config=GCFG)),
            card(dcc.Graph(figure=risk_fig(),     config=GCFG)),
            cols="1.2fr 1.2fr",
        ),

        # Row 3b — Vitals trend (full width, interactive)
        card(
            html.Div([
                html.Span("Vital Metric:",
                          style={"fontSize":"12px","color":MUTED,"marginRight":"10px","fontWeight":"600"}),
                dd("vital-dd", ["Heart_Rate","Systolic_BP"], "Heart_Rate", width="180px"),
            ], style={"display":"flex","alignItems":"center","marginBottom":"14px"}),
            dcc.Graph(id="vitals-chart", figure=vitals_fig(), config=GCFG),
            extra={"marginBottom":"0"},
        ),

        # Row 3c — Time Period Selector for Trends
        html.Div([
            html.Span("Trend Analysis — Select Period:",
                      style={"fontSize":"13px","color":TEXT,"fontWeight":"600","marginRight":"16px"}),
            html.Div([
                html.Button("Last Week", id="period-1w", n_clicks=0,
                           style={"padding":"8px 14px","margin":"0 4px","fontSize":"11px","fontWeight":"500",
                                  "background":TEAL,"color":TEXT,"border":"none","borderRadius":"6px","cursor":"pointer"}),
                html.Button("Last Month", id="period-1m", n_clicks=1,
                           style={"padding":"8px 14px","margin":"0 4px","fontSize":"11px","fontWeight":"500",
                                  "background":SURFACE,"color":SUB,"border":f"1px solid {BORDER}","borderRadius":"6px","cursor":"pointer"}),
                html.Button("Last Quarter", id="period-3m", n_clicks=0,
                           style={"padding":"8px 14px","margin":"0 4px","fontSize":"11px","fontWeight":"500",
                                  "background":SURFACE,"color":SUB,"border":f"1px solid {BORDER}","borderRadius":"6px","cursor":"pointer"}),
                html.Button("Last 6 Months", id="period-6m", n_clicks=0,
                           style={"padding":"8px 14px","margin":"0 4px","fontSize":"11px","fontWeight":"500",
                                  "background":SURFACE,"color":SUB,"border":f"1px solid {BORDER}","borderRadius":"6px","cursor":"pointer"}),
                html.Button("Last Year", id="period-1y", n_clicks=0,
                           style={"padding":"8px 14px","margin":"0 4px","fontSize":"11px","fontWeight":"500",
                                  "background":SURFACE,"color":SUB,"border":f"1px solid {BORDER}","borderRadius":"6px","cursor":"pointer"}),
                html.Button("All Time", id="period-all", n_clicks=0,
                           style={"padding":"8px 14px","margin":"0 4px","fontSize":"11px","fontWeight":"500",
                                  "background":SURFACE,"color":SUB,"border":f"1px solid {BORDER}","borderRadius":"6px","cursor":"pointer"}),
            ], style={"display":"flex","gap":"6px","flexWrap":"wrap"}),
        ], style={"background":CARD,"border":f"1px solid {BORDER}","borderRadius":"12px",
                  "padding":"16px","display":"flex","alignItems":"center","gap":"16px",
                  "marginBottom":"12px","flexWrap":"wrap"}),
        dcc.Store(id="period-store", data="1m"),

        # Row 4 — Trends (4 interactive charts)
        row(
            card(dcc.Graph(id="visits-trend-chart", config=GCFG)),
            card(dcc.Graph(id="diag-trend-chart", config=GCFG)),
            cols="1fr 1fr",
        ),

        row(
            card(dcc.Graph(id="dept-trend-chart", config=GCFG)),
            card(dcc.Graph(id="disease-trend-chart", config=GCFG)),
            cols="1fr 1fr",
        ),

        # Row 5 — Scatter explorer
        card(
            html.Div([
                html.Span("Scatter Explorer",
                          style={"fontSize":"13px","color":TEXT,"fontWeight":"600","marginRight":"20px"}),
                html.Span("X:", style={"fontSize":"12px","color":MUTED,"marginRight":"6px"}),
                dd("sc-x", NUMERIC_COLS, "BMI", "148px"),
                html.Span("Y:", style={"fontSize":"12px","color":MUTED,"margin":"0 6px 0 14px"}),
                dd("sc-y", NUMERIC_COLS, "Systolic_BP", "148px"),
                html.Span("Color:", style={"fontSize":"12px","color":MUTED,"margin":"0 6px 0 14px"}),
                dd("sc-col", ["Department","Gender"], "Department", "140px"),
            ], style={"display":"flex","alignItems":"center","marginBottom":"6px"}),
            dcc.Graph(id="scatter-chart", figure=scatter_fig(), config=GCFG),
            extra={"marginBottom":"0"},
        ),

        # Footer
        html.Div("",
                 style={"textAlign":"center","fontSize":"11px","color":MUTED,
                        "padding":"20px 0 8px","letterSpacing":"0.3px"}),

    ], style={"padding":"20px 28px","maxWidth":"1600px","margin":"0 auto"}),

], style={"background":BG,"minHeight":"100vh"})


# ── Callbacks ──────────────────────────────────────────────────────────────────

@app.callback(Output("vitals-chart","figure"), Input("vital-dd","value"))
def cb_vitals(m): return vitals_fig(m)

@app.callback(Output("scatter-chart","figure"),
              Input("sc-x","value"), Input("sc-y","value"), Input("sc-col","value"))
def cb_scatter(x, y, c): return scatter_fig(x, y, c)

# Time period selection callbacks
@app.callback(Output("period-store","data"),
              Input("period-1w","n_clicks"), Input("period-1m","n_clicks"),
              Input("period-3m","n_clicks"), Input("period-6m","n_clicks"),
              Input("period-1y","n_clicks"), Input("period-all","n_clicks"))
def update_period(c1, c2, c3, c4, c5, c6):
    periods = ["1w", "1m", "3m", "6m", "1y", "all"]
    clicks = [c1, c2, c3, c4, c5, c6]
    max_click = max(clicks)
    if max_click == 0:
        return "1m"
    return periods[clicks.index(max_click)]

# Update period button styles
@app.callback(
    [Output("period-1w","style"), Output("period-1m","style"),
     Output("period-3m","style"), Output("period-6m","style"),
     Output("period-1y","style"), Output("period-all","style")],
    Input("period-store","data"))
def update_button_styles(period):
    periods = ["1w", "1m", "3m", "6m", "1y", "all"]
    styles = []
    for p in periods:
        if p == period:
            styles.append({"padding":"8px 14px","margin":"0 4px","fontSize":"11px","fontWeight":"500",
                          "background":TEAL,"color":TEXT,"border":"none","borderRadius":"6px","cursor":"pointer"})
        else:
            styles.append({"padding":"8px 14px","margin":"0 4px","fontSize":"11px","fontWeight":"500",
                          "background":SURFACE,"color":SUB,"border":f"1px solid {BORDER}","borderRadius":"6px","cursor":"pointer"})
    return styles

# Trend charts callbacks
@app.callback(Output("visits-trend-chart","figure"), Input("period-store","data"))
def cb_visits_trend(period): return visits_trend_fig(period)

@app.callback(Output("diag-trend-chart","figure"), Input("period-store","data"))
def cb_diag_trend(period): return diagnosis_trend_fig(period)

@app.callback(Output("dept-trend-chart","figure"), Input("period-store","data"))
def cb_dept_trend(period): return department_trend_fig(period)

@app.callback(Output("disease-trend-chart","figure"), Input("period-store","data"))
def cb_disease_trend(period): return disease_prevalence_fig(period)


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  ┌─────────────────────────────────────┐")
    print("  │  Mediscope Insight                  │")
    print("  │  → http://127.0.0.1:8050            │")
    print("  └─────────────────────────────────────┘\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
