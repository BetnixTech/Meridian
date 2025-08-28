"""
Meridian Ultimate Bayesian Multi-KPI MMM Dashboard
Run: streamlit run meridian_ultimate_multi_kpi.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pymc3 as pm
import arviz as az
from fpdf import FPDF
import base64

st.title("Meridian Ultimate Multi-KPI Bayesian MMM Framework")

# -------------------------------
# 1. Load Data
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV with marketing channels + KPIs", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using sample data (Sales & Leads as KPIs)")
    df = pd.DataFrame({
        'TV':[200,220,250,300,280,310,330],
        'Digital':[50,60,65,80,70,90,100],
        'Radio':[30,35,40,50,45,55,60],
        'Promotions':[20,25,30,35,30,40,45],
        'Sales':[500,550,600,700,650,750,800],
        'Leads':[50,55,60,70,65,75,80]
    })
st.dataframe(df)

channels = [c for c in df.columns if c.lower() not in ['sales','leads']]
kpis = ['Sales','Leads']
X = df[channels]
y = df[kpis]

# -------------------------------
# 2. Channel-specific Adstock & Diminishing Returns
# -------------------------------
st.header("1. Channel-specific Adstock & Diminishing Returns")
adstock = {}
diminish = {}
X_adstocked = X.copy()
for c in channels:
    ad_factor = st.slider(f"{c} Adstock",0.0,0.9,0.5)
    diminish_factor = st.slider(f"{c} Diminishing Returns",0.0,1.0,0.5)
    adstock[c] = ad_factor
    diminish[c] = diminish_factor
    # Apply adstock
    result = np.zeros_like(X[c])
    result[0] = X[c].iloc[0]
    for t in range(1,len(X[c])):
        result[t] = X[c].iloc[t] + ad_factor*result[t-1]
    # Apply diminishing returns (log)
    X_adstocked[c] = np.log1p(result)*(1-diminish_factor) + result*diminish_factor

# -------------------------------
# 3. Seasonality & Holiday Effects
# -------------------------------
st.header("2. Seasonality & Holidays")
seasonality_factor = st.slider("Seasonality Amplitude",0.0,0.5,0.1)
holiday_factor = st.slider("Holiday Effect Amplitude",0.0,0.5,0.1)
time_index = np.arange(len(df))
seasonality = 1 + seasonality_factor*np.sin(2*np.pi*time_index/12)
holidays = 1 + holiday_factor*((time_index%12==11).astype(float)) # e.g., month 12 is holiday
X_adstocked['Seasonality'] = seasonality
X_adstocked['Holiday'] = holidays
channels_aug = channels + ['Seasonality','Holiday']

# -------------------------------
# 4. Bayesian Multi-KPI MMM
# -------------------------------
st.header("3. Bayesian Multi-KPI MMM")
trace_dict = {}
for kpi in kpis:
    st.subheader(f"KPI: {kpi}")
    with pm.Model() as model:
        alpha = pm.Normal("alpha",mu=0,sigma=100)
        betas = pm.Normal("betas",mu=0,sigma=10,shape=len(channels_aug))
        sigma = pm.HalfNormal("sigma",sigma=10)
        mu = alpha + pm.math.dot(X_adstocked[channels_aug].values,betas)
        y_obs = pm.Normal("y_obs",mu=mu,sigma=sigma,observed=y[kpi].values)
        trace = pm.sample(1000,tune=1000,cores=1,progressbar=False)
    trace_dict[kpi] = trace
    contrib = pd.Series(trace['betas'].mean(axis=0)[:len(channels)], index=channels)
    st.write(contrib)
    st.write(f"Base {kpi}: {trace['alpha'].mean():.2f}")

# -------------------------------
# 5. Multi-Scenario Simulation
# -------------------------------
st.header("4. Multi-Scenario Simulation")
n_scenarios = st.slider("Number of Scenarios",1,3,2)
scenarios=[]
scenario_names=[]
for i in range(n_scenarios):
    st.subheader(f"Scenario {i+1}")
    scenario={}
    for c in channels:
        scenario[c]=st.number_input(f"{c} spend S{i+1}", float(X[c].mean()), key=f"{c}_{i}")
    scenarios.append(scenario)
    scenario_names.append(f"Scenario {i+1}")

scenario_df = pd.DataFrame(scenarios)
scenario_adstocked = scenario_df.copy()
for c in channels:
    scenario_adstocked[c] = np.log1p(scenario_df[c]*(1+adstock[c]))*(1-diminish[c]) + scenario_df[c]*(diminish[c])
scenario_adstocked['Seasonality'] = 1
scenario_adstocked['Holiday'] = 1

for kpi in kpis:
    st.subheader(f"{kpi} Predictions")
    pred_sales=[]
    for i in range(n_scenarios):
        pred = trace_dict[kpi]['alpha'][:,None]+np.dot(trace_dict[kpi]['betas'], scenario_adstocked.iloc[i].values)
        mean = pred.mean()
        lower = np.percentile(pred,5)
        upper = np.percentile(pred,95)
        pred_sales.append((mean,lower,upper))
        st.write(f"{scenario_names[i]} {kpi}: {mean:.2f} [{lower:.2f},{upper:.2f}]")
    # Plot
    fig, ax = plt.subplots()
    for i,(mean,lower,upper) in enumerate(pred_sales):
        ax.bar(i,mean,yerr=[[mean-lower],[upper-mean]],capsize=5,label=scenario_names[i])
    ax.set_xticks(range(n_scenarios))
    ax.set_xticklabels(scenario_names)
    ax.set_ylabel(kpi)
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# 6. Multi-Period Forecast
# -------------------------------
st.header("5. Multi-Period Forecast")
periods = st.slider("Forecast Periods",1,12,6)
forecast_df = pd.DataFrame({c: np.mean(X[c]) for c in channels}, index=range(periods))
forecast_adstocked = forecast_df.copy()
for c in channels:
    forecast_adstocked[c] = np.log1p(forecast_df[c]*(1+adstock[c]))*(1-diminish[c]) + forecast_df[c]*diminish[c]
forecast_adstocked['Seasonality'] = 1 + seasonality_factor*np.sin(2*np.pi*np.arange(periods)/12)
forecast_adstocked['Holiday'] = 1 + holiday_factor*((np.arange(periods)%12==11).astype(float))

for kpi in kpis:
    pred = trace_dict[kpi]['alpha'][:,None]+np.dot(trace_dict[kpi]['betas'],forecast_adstocked.T)
    mean = pred.mean(axis=0)
    lower = np.percentile(pred,5,axis=0)
    upper = np.percentile(pred,95,axis=0)
    fig, ax = plt.subplots()
    ax.plot(range(periods),mean,label="Mean Forecast")
    ax.fill_between(range(periods),lower,upper,color='skyblue',alpha=0.3,label="90% CI")
    ax.set_title(f"{kpi} Forecast")
    ax.set_xlabel("Period")
    ax.set_ylabel(kpi)
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# 7. ROI-Constrained Budget Optimization
# -------------------------------
st.header("6. ROI-Constrained Allocation Optimization")
total_budget = st.number_input("Total Budget",500.0)
roi_threshold = st.slider("Minimum ROI per channel",0.0,5.0,1.0)

def objective(x):
    scenario_opt = pd.DataFrame([x],columns=channels)
    scenario_opt_ad = scenario_opt.copy()
    for c in channels:
        scenario_opt_ad[c] = np.log1p(scenario_opt[c]*(1+adstock[c]))*(1-diminish[c]) + scenario_opt[c]*diminish[c]
    scenario_opt_ad['Seasonality']=1
    scenario_opt_ad['Holiday']=1
    total_pred = 0
    for kpi in kpis:
        total_pred += trace_dict[kpi]['alpha'].mean() + np.dot(trace_dict[kpi]['betas'].mean(axis=0), scenario_opt_ad.iloc[0].values)
    return -total_pred

bounds = [(0,total_budget)]*len(channels)
constraints = {'type':'eq','fun':lambda x: total_budget-sum(x)}
res = minimize(objective,x0=[total_budget/len(channels)]*len(channels),bounds=bounds,constraints=constraints)

optimized_scenario = pd.Series(res.x,index=channels)
st.subheader("Optimized Allocation")
st.write(optimized_scenario)

# -------------------------------
# 8. Export Full PDF Report
# -------------------------------
st.header("7. Export PDF Report")
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial","B",16)
pdf.cell(0,10,"Meridian Ultimate Multi-KPI Bayesian MMM Report",ln=True)
pdf.ln(5)
for i,kpi in enumerate(kpis):
    pdf.set_font("Arial","",12)
    pdf.cell(0,10,f"KPI: {kpi}",ln=True)
    for j,(mean,lower,upper) in enumerate(pred_sales):
        pdf.cell(0,10,f"{scenario_names[j]} {kpi}: {mean:.2f} [{lower:.2f},{upper:.2f}]",ln=True)
pdf.ln(5)
pdf.cell(0,10,"Optimized Allocation:",ln=True)
for c in channels:
    pdf.cell(0,10,f"{c}: {optimized_scenario[c]:.2f}",ln=True)
pdf_file="meridian_ultimate_multi_kpi_report.pdf"
pdf.output(pdf_file)
with open(pdf_file,"rb") as f:
    b64 = base64.b64encode(f.read()).decode()
st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="meridian_ultimate_multi_kpi_report.pdf">Download PDF Report</a>',unsafe_allow_html=True)
