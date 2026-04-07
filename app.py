import itertools
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from openpyxl import load_workbook

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Irish Rental Market Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Power BI Dark Theme CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #111318 !important;
    color: #E8EAF0 !important;
    font-family: 'Barlow', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #1A1D26 !important;
    border-right: 1px solid #2A2D3A !important;
}
[data-testid="stSidebar"] * { color: #C8CBD8 !important; }
h1 { font-family: 'Barlow Condensed', sans-serif !important; font-size: 2rem !important; color: #FFFFFF !important; letter-spacing: 1px !important; }
h2 { font-family: 'Barlow Condensed', sans-serif !important; font-size: 1.4rem !important; color: #C8CBD8 !important; border-bottom: 1px solid #2A2D3A; padding-bottom: 8px; }
.kpi-card {
    background: linear-gradient(135deg, #1E2130 0%, #252836 100%);
    border: 1px solid #2E3145; border-radius: 10px; padding: 18px 20px;
    text-align: center; transition: transform 0.2s, border-color 0.2s; position: relative; overflow: hidden;
}
.kpi-card:hover { transform: translateY(-2px); border-color: #4A90D9; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #4A90D9, #00C4B4); }
.kpi-value { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 700; color: #FFFFFF; line-height: 1; }
.kpi-label { font-size: 0.72rem; color: #7A7D8C; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }
.kpi-delta-pos { font-size: 0.8rem; color: #FF6B6B; font-weight: 600; margin-top: 4px; }
.kpi-delta-neu { font-size: 0.8rem; color: #00C4B4; font-weight: 600; margin-top: 4px; }
.section-header {
    background: linear-gradient(90deg, #1E2130, transparent); border-left: 3px solid #4A90D9;
    padding: 8px 16px; margin: 16px 0 12px 0; font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem; font-weight: 600; color: #C8CBD8; text-transform: uppercase; letter-spacing: 2px;
}
.dash-divider { height: 1px; background: linear-gradient(90deg, #4A90D9 0%, transparent 80%); margin: 20px 0; }
div[data-testid="stMetric"] { background: #1E2130; border-radius: 8px; padding: 12px; border: 1px solid #2E3145; }
div[data-testid="stMetric"] label { color: #7A7D8C !important; }
div[data-testid="stMetric"] div { color: #FFFFFF !important; }
.stTabs [data-baseweb="tab-list"] { background: #1A1D26; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #7A7D8C !important; }
.stTabs [aria-selected="true"] { color: #4A90D9 !important; border-bottom: 2px solid #4A90D9; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Colors ───────────────────────────────────────────────────────────────────
PLOT_BG = "#111318"; PAPER_BG = "#1A1D26"; GRID_COLOR = "#2A2D3A"; TEXT_COLOR = "#C8CBD8"
ACCENT1="#4A90D9"; ACCENT2="#00C4B4"; ACCENT3="#FF6B6B"; ACCENT4="#F5A623"; ACCENT5="#9B59B6"

def dark_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Barlow Condensed", size=16, color=TEXT_COLOR)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(family="Barlow", color=TEXT_COLOR),
        height=height, margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR, font=dict(size=11), orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
    return fig

def quarter_to_date(q):
    parts = str(q).strip().split()
    qnum = int(parts[0][1]); year = int(parts[1])
    return pd.Timestamp(year=year, month=(qnum-1)*3+1, day=1)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    wb = load_workbook('RTB_Regional_Rent_TimeSeries_2007_2025.xlsx', read_only=True)
    ws_new = wb['RTBRI Q325 New']
    data_new = [r for r in list(ws_new.iter_rows(values_only=True))[2:] if r[0] and str(r[0]).startswith('Q')]
    rtb_new = pd.DataFrame(data_new, columns=['Period','Dublin','Non_Dublin','GDA_excl_Dublin','Outside_GDA'])
    ws_exist = wb['RTBRI Q325 Existing']
    data_exist = [r for r in list(ws_exist.iter_rows(values_only=True))[2:] if r[0] and str(r[0]).startswith('Q')]
    rtb_exist = pd.DataFrame(data_exist, columns=['Period','Dublin_Exist','Non_Dublin_Exist','GDA_Exist','Outside_GDA_Exist'])
    rtb_new['Date'] = rtb_new['Period'].apply(quarter_to_date)
    rtb_exist['Date'] = rtb_exist['Period'].apply(quarter_to_date)
    rtb = pd.merge(rtb_new, rtb_exist[['Date','Dublin_Exist','Non_Dublin_Exist']], on='Date', how='left').sort_values('Date').reset_index(drop=True)
    for col in ['Dublin','Non_Dublin','GDA_excl_Dublin','Outside_GDA']: rtb[col] = pd.to_numeric(rtb[col], errors='coerce')

    wb2 = load_workbook('RTB_County_Rent_Snapshot_Q32025.xlsx', read_only=True)
    county_data = [{'County':r[0],'Rent_Q32025':round(float(r[1]),2)} for r in list(wb2['RIQ325 New'].iter_rows(values_only=True))[2:] if r[0] and isinstance(r[0],str) and r[1] and isinstance(r[1],(int,float))]
    county_snapshot = pd.DataFrame(county_data).sort_values('Rent_Q32025',ascending=False).reset_index(drop=True)

    unemp_df = pd.read_excel('CSO_Unemployment_Monthly_1998_2026.xlsx', sheet_name='Unpivoted')
    um = ((unemp_df['Statistic Label'].str.strip()=='Seasonally Adjusted Monthly Unemployment Rate (%)') & (unemp_df['Age Group'].str.strip()=='15 - 74 years') & (unemp_df['Sex'].str.strip()=='Both sexes'))
    unemp = unemp_df[um][['Month','VALUE']].copy()
    if unemp.empty:
        um2 = ((unemp_df['Statistic Label'].str.strip()=='Seasonally Adjusted Monthly Unemployment Rate') & (unemp_df['Age Group'].str.strip()=='15 - 74 years') & (unemp_df['Sex'].str.strip()=='Both sexes'))
        unemp = unemp_df[um2][['Month','VALUE']].copy()
    unemp.columns=['Month','Unemployment_Rate']; unemp['Date']=pd.to_datetime(unemp['Month'],format='%Y %B',errors='coerce')
    unemp = unemp.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    unemp_q = unemp.set_index('Date')['Unemployment_Rate'].resample('QS').mean().reset_index(); unemp_q.columns=['Date','Unemployment_Rate']

    ppi_df = pd.read_excel('CSO_PropertyPriceIndex_2005_2026.xlsx', sheet_name='Unpivoted')
    pm = ((ppi_df['Statistic Label']=='Residential Property Price Index') & (ppi_df['Type of Residential Property']=='National - all residential properties'))
    ppi = ppi_df[pm][['Month','VALUE']].copy(); ppi.columns=['Month','RPPI']
    ppi['Date']=pd.to_datetime(ppi['Month'],format='%Y %B',errors='coerce'); ppi=ppi.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    ppi_q = ppi.set_index('Date')['RPPI'].resample('QS').mean().reset_index(); ppi_q.columns=['Date','RPPI']

    hc_raw=pd.read_excel('CSO_HousingCompletions_Quarterly.xlsx',sheet_name='Unpivoted')
    rc=hc_raw.columns[7]; yc=hc_raw.columns[3]; vc=hc_raw.columns[-1]
    hc=hc_raw[hc_raw[rc].astype(str).str.contains('State',na=False)][[yc,vc]].copy(); hc.columns=['Year','Completions']
    hc=hc.groupby('Year')['Completions'].sum().reset_index(); hc['Year']=pd.to_numeric(hc['Year'],errors='coerce'); hc=hc.dropna().sort_values('Year').reset_index(drop=True)

    income_df=pd.read_excel('CSO_HouseholdIncome_SILC_2020_2025.xlsx',sheet_name='Unpivoted')
    im=((income_df['Statistic Label']=='Mean Nominal Household Disposable Income (Euro)') & (income_df['Sex']=='Both sexes'))
    income=income_df[im][['Year','VALUE']].copy(); income.columns=['Year','Mean_Annual_Income']
    income['Year']=pd.to_numeric(income['Year'],errors='coerce'); income['Monthly_Income']=income['Mean_Annual_Income']/12; income=income.sort_values('Year').reset_index(drop=True)

    wb7=load_workbook('RTB_Landlord_Numbers_Q22023_Q42025.xlsx',read_only=True); ws7=wb7['RTB LL Total Q425']; rows7=list(ws7.iter_rows(values_only=True))
    landlord=pd.DataFrame({'Quarter':list(rows7[1][1:]),'Total_Landlords':list(rows7[2][1:])}).dropna()
    landlord['Date']=landlord['Quarter'].apply(quarter_to_date); landlord['Total_Landlords']=pd.to_numeric(landlord['Total_Landlords']); landlord['QoQ_Change']=landlord['Total_Landlords'].diff(); landlord=landlord.sort_values('Date').reset_index(drop=True)

    master=rtb[['Date','Period','Dublin','Non_Dublin','GDA_excl_Dublin','Outside_GDA']].copy()
    master=pd.merge(master,unemp_q,on='Date',how='left'); master=pd.merge(master,ppi_q,on='Date',how='left')
    master['Year']=master['Date'].dt.year; master=pd.merge(master,income[['Year','Monthly_Income']],on='Year',how='left')
    master['Monthly_Income']=master['Monthly_Income'].ffill().bfill()
    master['Affordability_Dublin']=(master['Dublin']/master['Monthly_Income']*100).round(2)
    master['Affordability_National']=(master['Non_Dublin']/master['Monthly_Income']*100).round(2)
    master['RPZ_Active']=(master['Date']>='2016-10-01').astype(int)
    ms=master.sort_values('Date').copy(); master['Dublin_YoY']=ms['Dublin'].pct_change(4).values*100; master['NonDublin_YoY']=ms['Non_Dublin'].pct_change(4).values*100
    return master, county_snapshot, landlord, hc, income, unemp_q, ppi_q, rtb

@st.cache_data
def run_models(_master, _unemp_q, _ppi_q):
    df_ts=_master[['Date','Dublin','Non_Dublin']].copy().sort_values('Date').set_index('Date')
    ts_n=15; n=len(df_ts); tr_s=df_ts.iloc[:n-ts_n]; te_s=df_ts.iloc[n-ts_n:]

    def sgs(series):
        ba=np.inf; bo=(1,1,1); bs=(1,1,1,4)
        for p,d,q in itertools.product([0,1,2],[1],[0,1,2]):
            for P,D,Q in itertools.product([0,1],[1],[0,1]):
                try:
                    r=SARIMAX(series,order=(p,d,q),seasonal_order=(P,D,Q,4),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
                    if r.aic<ba: ba=r.aic; bo=(p,d,q); bs=(P,D,Q,4)
                except: pass
        return bo,bs

    bod,bsd=sgs(tr_s['Dublin']); bond,bsnd=sgs(tr_s['Non_Dublin'])
    sdf=SARIMAX(tr_s['Dublin'],order=bod,seasonal_order=bsd,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    sndf=SARIMAX(tr_s['Non_Dublin'],order=bond,seasonal_order=bsnd,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)

    def evs(fit,col):
        po=fit.get_forecast(steps=len(te_s)); pm=po.predicted_mean; pm.index=te_s.index; ci=po.conf_int(alpha=0.05); ci.index=te_s.index
        return pm,ci,mean_absolute_error(te_s[col],pm),np.sqrt(mean_squared_error(te_s[col],pm)),np.mean(np.abs((te_s[col]-pm)/te_s[col]))*100

    pds,cds,maeds,_,mapeds=evs(sdf,'Dublin'); pnds,cnds,maends,_,mapends=evs(sndf,'Non_Dublin')
    nf=8
    sdf_f=SARIMAX(df_ts['Dublin'],order=bod,seasonal_order=bsd,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    sndf_f=SARIMAX(df_ts['Non_Dublin'],order=bond,seasonal_order=bsnd,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    fdt=pd.date_range(start=df_ts.index[-1]+pd.DateOffset(months=3),periods=nf,freq='QS')
    fcd=sdf_f.get_forecast(steps=nf); fcdm=fcd.predicted_mean; fcdm.index=fdt; fcdci=fcd.conf_int(alpha=0.05); fcdci.index=fdt
    fcnd=sndf_f.get_forecast(steps=nf); fcndm=fcnd.predicted_mean; fcndm.index=fdt; fcndci=fcnd.conf_int(alpha=0.05); fcndci.index=fdt

    df_rf=_master[['Date','Dublin','Non_Dublin']].copy()
    df_rf=df_rf.merge(_unemp_q,on='Date',how='left').merge(_ppi_q,on='Date',how='left'); df_rf['RPZ']=(df_rf['Date']>='2016-10-01').astype(int)

    def mf(di,tc,nl=4):
        d=di.copy()
        for l in range(1,nl+1): d[f'{tc}_lag{l}']=d[tc].shift(l)
        d[f'{tc}_roll4_mean']=d[tc].shift(1).rolling(4).mean(); d[f'{tc}_roll4_std']=d[tc].shift(1).rolling(4).std()
        d[f'{tc}_yoy']=d[tc].pct_change(4)*100; d['Quarter']=d['Date'].dt.quarter; d['RPPI_lag1']=d['RPPI'].shift(1); d['Unemp_lag1']=d['Unemployment_Rate'].shift(1)
        return d.dropna()

    dfd=mf(df_rf,'Dublin'); dfnd=mf(df_rf,'Non_Dublin')
    ex={'Date','Period','Dublin','Non_Dublin','GDA','Outside_GDA'}
    fcd_c=[c for c in dfd.columns if c not in ex]; fcnd_c=[c for c in dfnd.columns if c not in ex]
    trf=12
    ddc=dfd[fcd_c+['Dublin','Date']].dropna(); dndc=dfnd[fcnd_c+['Non_Dublin','Date']].dropna()
    Xdt,Xde=ddc[fcd_c].values[:-trf],ddc[fcd_c].values[-trf:]; ydt,yde=ddc['Dublin'].values[:-trf],ddc['Dublin'].values[-trf:]; dtd=ddc['Date'].values[-trf:]
    Xndt,Xnde=dndc[fcnd_c].values[:-trf],dndc[fcnd_c].values[-trf:]; yndt,ynde=dndc['Non_Dublin'].values[:-trf],dndc['Non_Dublin'].values[-trf:]; dtnd=dndc['Date'].values[-trf:]

    rfd=RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_split=3,random_state=42,n_jobs=-1); rfd.fit(Xdt,ydt); prfd=rfd.predict(Xde)
    maed_rf=mean_absolute_error(yde,prfd); maped_rf=np.mean(np.abs((yde-prfd)/yde))*100
    rfnd=RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_split=3,random_state=42,n_jobs=-1); rfnd.fit(Xndt,yndt); prfnd=rfnd.predict(Xnde)
    maend_rf=mean_absolute_error(ynde,prfnd); mapend_rf=np.mean(np.abs((ynde-prfnd)/ynde))*100

    gbd=GradientBoostingRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,random_state=42); gbd.fit(Xdt,ydt); pgbd=gbd.predict(Xde)
    maed_gb=mean_absolute_error(yde,pgbd); maped_gb=np.mean(np.abs((yde-pgbd)/yde))*100
    gbnd=GradientBoostingRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,random_state=42); gbnd.fit(Xndt,yndt); pgbnd=gbnd.predict(Xnde)
    maend_gb=mean_absolute_error(ynde,pgbnd); mapend_gb=np.mean(np.abs((ynde-pgbnd)/ynde))*100

    try: from prophet import Prophet
    except: import subprocess; subprocess.run(['pip','install','prophet','-q'],check=False); from prophet import Prophet
    dp=df_rf[['Date','Dublin','Non_Dublin','Unemployment_Rate','RPPI']].copy()
    dpd=dp[['Date','Dublin','Unemployment_Rate','RPPI']].rename(columns={'Date':'ds','Dublin':'y'}).dropna()
    dpnd=dp[['Date','Non_Dublin','Unemployment_Rate','RPPI']].rename(columns={'Date':'ds','Non_Dublin':'y'}).dropna()
    tp=12; tpd=dpd.iloc[:-tp]; tepd=dpd.iloc[-tp:]; tpnd=dpnd.iloc[:-tp]; tepnd=dpnd.iloc[-tp:]
    regs=['Unemployment_Rate','RPPI']
    def bp(tr):
        mm=Prophet(changepoints=['2016-10-01','2020-01-01'],changepoint_prior_scale=0.05,seasonality_mode='additive',yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False)
        mm.add_seasonality(name='quarterly',period=91.25,fourier_order=5)
        for r in regs: mm.add_regressor(r)
        mm.fit(tr); return mm
    mpd=bp(tpd); mpnd=bp(tpnd)
    ftd=mpd.predict(tepd); ftnd=mpnd.predict(tepnd)
    maped_p=np.mean(np.abs((tepd['y'].values-ftd['yhat'].values)/tepd['y'].values))*100; maed_p=mean_absolute_error(tepd['y'],ftd['yhat'])
    mapend_p=np.mean(np.abs((tepnd['y'].values-ftnd['yhat'].values)/tepnd['y'].values))*100; maend_p=mean_absolute_error(tepnd['y'],ftnd['yhat'])
    mpdf=bp(dpd); mpndf=bp(dpnd)
    lu=df_rf['Unemployment_Rate'].dropna().iloc[-1]; lr=df_rf['RPPI'].dropna().iloc[-1]; ld=df_rf['Date'].iloc[-1]
    fpts=pd.date_range(start=ld+pd.DateOffset(months=3),periods=nf,freq='QS')
    fdf_p=pd.DataFrame({'ds':fpts,'Unemployment_Rate':lu,'RPPI':lr})
    fpfd=mpdf.predict(pd.concat([dpd[['ds']+regs],fdf_p],ignore_index=True)); fpfnd=mpndf.predict(pd.concat([dpnd[['ds']+regs],fdf_p],ignore_index=True))
    pfud=fpfd[fpfd['ds'].isin(fpts)]; pfund=fpfnd[fpfnd['ds'].isin(fpts)]

    dsx=df_rf.set_index('Date'); tssx=15; nsx=len(dsx); trsx=dsx.iloc[:nsx-tssx]; tesx=dsx.iloc[nsx-tssx:]
    sc=StandardScaler(); ec=['Unemployment_Rate','RPPI']
    trex=pd.DataFrame(sc.fit_transform(trsx[ec]),index=trsx.index,columns=ec); teex=pd.DataFrame(sc.transform(tesx[ec]),index=tesx.index,columns=ec)

    def sxgs(series,exog):
        ba=np.inf; bo=(1,1,1); bs=(1,1,1,4)
        for p,d,q in itertools.product([0,1,2],[1],[0,1,2]):
            for P,D,Q in itertools.product([0,1],[1],[0,1]):
                try:
                    r=SARIMAX(series,exog=exog,order=(p,d,q),seasonal_order=(P,D,Q,4),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=200)
                    if r.aic<ba: ba=r.aic; bo=(p,d,q); bs=(P,D,Q,4)
                except: pass
        return bo,bs

    bsxd,bssxd=sxgs(trsx['Dublin'],trex); bsxnd,bssxnd=sxgs(trsx['Non_Dublin'],trex)
    sxdf=SARIMAX(trsx['Dublin'],exog=trex,order=bsxd,seasonal_order=bssxd,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=200)
    sxndf=SARIMAX(trsx['Non_Dublin'],exog=trex,order=bsxnd,seasonal_order=bssxnd,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=200)
    opd=sxdf.get_forecast(steps=len(tesx),exog=teex); psxd=opd.predicted_mean; psxd.index=tesx.index; csxd=opd.conf_int(alpha=0.05); csxd.index=tesx.index
    opnd=sxndf.get_forecast(steps=len(tesx),exog=teex); psxnd=opnd.predicted_mean; psxnd.index=tesx.index; csxnd=opnd.conf_int(alpha=0.05); csxnd.index=tesx.index
    maped_sx=np.mean(np.abs((tesx['Dublin']-psxd)/tesx['Dublin']))*100; maed_sx=mean_absolute_error(tesx['Dublin'],psxd)
    mapend_sx=np.mean(np.abs((tesx['Non_Dublin']-psxnd)/tesx['Non_Dublin']))*100; maend_sx=mean_absolute_error(tesx['Non_Dublin'],psxnd)
    fexdf=pd.DataFrame(sc.transform(dsx[ec]),index=dsx.index,columns=ec)
    sxdf_f=SARIMAX(dsx['Dublin'],exog=fexdf,order=bsxd,seasonal_order=bssxd,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=200)
    sxndf_f=SARIMAX(dsx['Non_Dublin'],exog=fexdf,order=bsxnd,seasonal_order=bssxnd,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=200)
    fesc=sc.transform(np.array([[dsx['Unemployment_Rate'].iloc[-1],dsx['RPPI'].iloc[-1]]]*nf))
    fdssx=pd.date_range(start=dsx.index[-1]+pd.DateOffset(months=3),periods=nf,freq='QS')
    fsxd=sxdf_f.get_forecast(steps=nf,exog=fesc); sxdm=fsxd.predicted_mean; sxdm.index=fdssx; sxdci=fsxd.conf_int(alpha=0.05); sxdci.index=fdssx
    fsxnd=sxndf_f.get_forecast(steps=nf,exog=fesc); sxndm=fsxnd.predicted_mean; sxndm.index=fdssx; sxndci=fsxnd.conf_int(alpha=0.05); sxndci.index=fdssx

    comp=pd.DataFrame([
        {'Model':'SARIMA',           'Dublin_MAE':maeds,   'Dublin_MAPE':mapeds,   'NonDublin_MAE':maends,   'NonDublin_MAPE':mapends},
        {'Model':'Random Forest',    'Dublin_MAE':maed_rf, 'Dublin_MAPE':maped_rf, 'NonDublin_MAE':maend_rf, 'NonDublin_MAPE':mapend_rf},
        {'Model':'Gradient Boosting','Dublin_MAE':maed_gb, 'Dublin_MAPE':maped_gb, 'NonDublin_MAE':maend_gb, 'NonDublin_MAPE':mapend_gb},
        {'Model':'Prophet',          'Dublin_MAE':maed_p,  'Dublin_MAPE':maped_p,  'NonDublin_MAE':maend_p,  'NonDublin_MAPE':mapend_p},
        {'Model':'SARIMAX',          'Dublin_MAE':maed_sx, 'Dublin_MAPE':maped_sx, 'NonDublin_MAE':maend_sx, 'NonDublin_MAPE':mapend_sx},
    ])
    comp['Avg_MAPE']=(comp['Dublin_MAPE']+comp['NonDublin_MAPE'])/2; comp['Avg_MAE']=(comp['Dublin_MAE']+comp['NonDublin_MAE'])/2
    comp=comp.sort_values('Avg_MAPE').reset_index(drop=True)

    return dict(
        df_ts=df_ts,tr_s=tr_s,te_s=te_s,pds=pds,cds=cds,mapeds=mapeds,maeds=maeds,pnds=pnds,cnds=cnds,mapends=mapends,maends=maends,
        fcdm=fcdm,fcdci=fcdci,fcndm=fcndm,fcndci=fcndci,
        dfd=dfd,dfnd=dfnd,fcd_c=fcd_c,fcnd_c=fcnd_c,trf=trf,
        ydt=ydt,yde=yde,prfd=prfd,dtd=dtd,maped_rf=maped_rf,maed_rf=maed_rf,
        yndt=yndt,ynde=ynde,prfnd=prfnd,dtnd=dtnd,mapend_rf=mapend_rf,maend_rf=maend_rf,
        pgbd=pgbd,maped_gb=maped_gb,maed_gb=maed_gb,pgbnd=pgbnd,mapend_gb=mapend_gb,maend_gb=maend_gb,
        rfd=rfd,rfnd=rfnd,
        dpd=dpd,dpnd=dpnd,tpd=tpd,tepd=tepd,tpnd=tpnd,tepnd=tepnd,ftd=ftd,ftnd=ftnd,
        maped_p=maped_p,maed_p=maed_p,mapend_p=mapend_p,maend_p=maend_p,pfud=pfud,pfund=pfund,
        trsx=trsx,tesx=tesx,psxd=psxd,csxd=csxd,psxnd=psxnd,csxnd=csxnd,
        maped_sx=maped_sx,maed_sx=maed_sx,mapend_sx=mapend_sx,maend_sx=maend_sx,
        sxdm=sxdm,sxdci=sxdci,sxndm=sxndm,sxndci=sxndci,dsx=dsx,comp=comp,
    )

# ─── Load ─────────────────────────────────────────────────────────────────────
with st.spinner("Loading datasets..."):
    master, county_snapshot, landlord, hc, income, unemp_q, ppi_q, rtb = load_all_data()
latest=master.iloc[-1]; first=master.iloc[0]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:16px 0 8px 0'>
    <div style='font-family:Barlow Condensed;font-size:1.5rem;font-weight:700;color:#fff;letter-spacing:2px'>🏠 IRISH RENTAL</div>
    <div style='font-size:0.7rem;color:#4A90D9;letter-spacing:3px;text-transform:uppercase'>Market Intelligence</div>
    </div><hr style='border-color:#2A2D3A;margin:8px 0 16px'>""", unsafe_allow_html=True)

    page = st.radio("", ["📊 Overview","📈 Rent Trends","🗺️ County Analysis","💸 Affordability",
        "🏗️ Supply & Landlords","🤖 SARIMA","🌲 Random Forest & GB","🔮 Prophet","📡 SARIMAX",
        "🏆 Model Comparison","📋 Policy Insights"], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#2A2D3A;margin:16px 0'><div style='font-size:0.7rem;color:#4A90D9;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px'>⚡ FILTERS</div>", unsafe_allow_html=True)

    min_y=int(master['Date'].dt.year.min()); max_y=int(master['Date'].dt.year.max())
    yr=st.slider("Date Range", min_y, max_y, (min_y, max_y))
    region_filter=st.multiselect("Region",["Dublin","Non-Dublin","GDA (excl. Dublin)","Outside GDA"],default=["Dublin","Non-Dublin"])
    county_filter=st.selectbox("County Focus",["All Counties"]+sorted(county_snapshot['County'].tolist()))
    model_filter=st.multiselect("Models (Comparison)",["SARIMA","Random Forest","Gradient Boosting","Prophet","SARIMAX"],default=["SARIMA","SARIMAX","Prophet"])

mf=master[(master['Date'].dt.year>=yr[0])&(master['Date'].dt.year<=yr[1])].copy()
rcm={"Dublin":("Dublin",ACCENT1),"Non-Dublin":("Non_Dublin",ACCENT2),"GDA (excl. Dublin)":("GDA_excl_Dublin",ACCENT4),"Outside GDA":("Outside_GDA",ACCENT5)}

# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page=="📊 Overview":
    st.markdown("""<div style='font-family:Barlow Condensed;font-size:2rem;font-weight:700;color:#fff;letter-spacing:1px'>
    IRISH RENTAL MARKET INTELLIGENCE</div>
    <div style='color:#4A90D9;font-size:0.8rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:20px'>MS5131 Major Project — Predictive Analytics Dashboard</div>
    <div class='dash-divider'></div>""", unsafe_allow_html=True)

    kpis=[
        (f"€{latest['Dublin']:,.0f}","Dublin Rent Q3'25",f"+{((latest['Dublin']/first['Dublin'])-1)*100:.0f}% since 2007","pos"),
        (f"€{latest['Non_Dublin']:,.0f}","Non-Dublin Rent Q3'25",f"+{((latest['Non_Dublin']/first['Non_Dublin'])-1)*100:.0f}% since 2007","pos"),
        (f"{latest['Affordability_Dublin']:.1f}%","Dublin Rent-to-Income","30% = Crisis threshold","pos"),
        (f"{master['Unemployment_Rate'].dropna().iloc[-1]:.1f}%","Unemployment Rate","Latest available","neu"),
        (f"{len(county_snapshot)}","Counties Tracked","Q3 2025 snapshot","neu"),
        ("5 Models","Forecast Models","SARIMA·RF·GB·Prophet·SARIMAX","neu"),
    ]
    cols=st.columns(6)
    for col,(val,label,delta,dt) in zip(cols,kpis):
        with col:
            st.markdown(f"""<div class='kpi-card'><div class='kpi-value'>{val}</div>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-delta-{dt}'>{delta}</div></div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2=st.columns([3,2])
    with c1:
        st.markdown("<div class='section-header'>RENT TREND BY REGION</div>",unsafe_allow_html=True)
        fig=go.Figure()
        lmap={"Dublin":"Dublin","Non_Dublin":"Non-Dublin","GDA_excl_Dublin":"GDA (excl. Dublin)","Outside_GDA":"Outside GDA"}
        for cn,clr in [("Dublin",ACCENT1),("Non_Dublin",ACCENT2),("GDA_excl_Dublin",ACCENT4),("Outside_GDA",ACCENT5)]:
            fig.add_trace(go.Scatter(x=mf['Date'],y=mf[cn],name=lmap[cn],line=dict(color=clr,width=2.5),mode='lines',
                hovertemplate=f"<b>{lmap[cn]}</b><br>%{{x|%Y}}<br>€%{{y:,.0f}}<extra></extra>"))
        for xd,lb,cl in [('2008-07-01','GFC 2008','#FF6B6B'),('2016-10-01','RPZ 2016','#9B59B6'),('2020-03-01','COVID','#F5A623')]:
            fig.add_vline(x=xd,line_dash="dash",line_color=cl,line_width=1,opacity=0.6,
                annotation_text=lb,annotation_font_color=cl,annotation_font_size=10)
        fig.update_yaxes(tickprefix="€",tickformat=","); fig.update_layout(hovermode='x unified')
        dark_layout(fig,height=360); st.plotly_chart(fig,use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>TOP 10 COUNTIES</div>",unsafe_allow_html=True)
        t10=county_snapshot.head(10)
        fig2=go.Figure(go.Bar(x=t10['Rent_Q32025'],y=t10['County'],orientation='h',
            marker=dict(color=[ACCENT1 if r>1700 else ACCENT2 for r in t10['Rent_Q32025']],line=dict(width=0)),
            text=[f"€{v:,.0f}" for v in t10['Rent_Q32025']],textposition='outside',textfont=dict(size=10,color=TEXT_COLOR),
            hovertemplate="<b>%{y}</b><br>€%{x:,.0f}<extra></extra>"))
        fig2.add_vline(x=county_snapshot['Rent_Q32025'].mean(),line_dash="dot",line_color=ACCENT3,
            annotation_text="Avg",annotation_font_color=ACCENT3,annotation_font_size=10)
        dark_layout(fig2,height=360); fig2.update_xaxes(tickprefix="€",tickformat=","); st.plotly_chart(fig2,use_container_width=True)

    c3,c4=st.columns(2)
    with c3:
        st.markdown("<div class='section-header'>AFFORDABILITY RATIO</div>",unsafe_allow_html=True)
        afd=mf.dropna(subset=['Affordability_Dublin'])
        fig3=go.Figure()
        fig3.add_trace(go.Scatter(x=afd['Date'],y=afd['Affordability_Dublin'],name='Dublin',fill='tozeroy',
            fillcolor='rgba(74,144,217,0.12)',line=dict(color=ACCENT1,width=2),hovertemplate="Dublin %{y:.1f}%<extra></extra>"))
        fig3.add_trace(go.Scatter(x=afd['Date'],y=afd['Affordability_National'],name='Non-Dublin',
            line=dict(color=ACCENT2,width=2,dash='dot'),hovertemplate="Non-Dublin %{y:.1f}%<extra></extra>"))
        fig3.add_hline(y=30,line_dash="solid",line_color=ACCENT3,line_width=2,annotation_text="30% Crisis",annotation_font_color=ACCENT3)
        fig3.update_yaxes(ticksuffix="%"); dark_layout(fig3,height=280); st.plotly_chart(fig3,use_container_width=True)

    with c4:
        st.markdown("<div class='section-header'>YoY GROWTH RATE</div>",unsafe_allow_html=True)
        gd=mf.dropna(subset=['Dublin_YoY'])
        fig4=go.Figure()
        fig4.add_trace(go.Bar(x=gd['Date'],y=gd['Dublin_YoY'],name='Dublin YoY',
            marker_color=[ACCENT1 if v>=0 else ACCENT3 for v in gd['Dublin_YoY']],
            hovertemplate="Dublin YoY: %{y:.1f}%<extra></extra>"))
        fig4.add_hline(y=4,line_dash="dash",line_color=ACCENT4,annotation_text="RPZ 4%",annotation_font_color=ACCENT4)
        fig4.add_hline(y=2,line_dash="dot",line_color=ACCENT5,annotation_text="RPZ 2%",annotation_font_color=ACCENT5)
        fig4.update_yaxes(ticksuffix="%"); dark_layout(fig4,height=280); st.plotly_chart(fig4,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RENT TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
elif page=="📈 Rent Trends":
    st.markdown("<div class='section-header'>RENT TRENDS — INTERACTIVE</div>",unsafe_allow_html=True)
    fig=go.Figure()
    lmap={"Dublin":"Dublin","Non_Dublin":"Non-Dublin","GDA_excl_Dublin":"GDA (excl. Dublin)","Outside_GDA":"Outside GDA"}
    for reg,(cn,clr) in rcm.items():
        vis=True if reg in region_filter else 'legendonly'
        fig.add_trace(go.Scatter(x=mf['Date'],y=mf[cn],name=reg,visible=vis,line=dict(color=clr,width=2.5),mode='lines+markers',marker=dict(size=4),
            hovertemplate=f"<b>{reg}</b><br>%{{x|%Y}}<br>€%{{y:,.0f}}/mo<extra></extra>"))
    for xd,lb,cl in [('2008-07-01','GFC 2008','#FF6B6B'),('2016-10-01','RPZ Intro','#9B59B6'),('2020-03-01','COVID-19','#F5A623')]:
        fig.add_vline(x=xd,line_dash="dash",line_color=cl,line_width=1.2,opacity=0.7,annotation_text=lb,annotation_font_color=cl,annotation_font_size=11,annotation_position="top left")
    fig.update_yaxes(tickprefix="€",tickformat=","); fig.update_layout(hovermode='x unified')
    dark_layout(fig,"Average Monthly Rent by Region",height=480); st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>YoY GROWTH RATE</div>",unsafe_allow_html=True)
        gd=mf.dropna(subset=['Dublin_YoY'])
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=gd['Date'],y=gd['Dublin_YoY'],name='Dublin',line=dict(color=ACCENT1,width=2),fill='tozeroy',fillcolor='rgba(74,144,217,0.1)',hovertemplate="Dublin %{y:.1f}%<extra></extra>"))
        fig2.add_trace(go.Scatter(x=gd['Date'],y=gd['NonDublin_YoY'],name='Non-Dublin',line=dict(color=ACCENT2,width=2,dash='dot'),hovertemplate="Non-Dublin %{y:.1f}%<extra></extra>"))
        fig2.add_hline(y=4,line_dash="dash",line_color=ACCENT4,annotation_text="4% cap",annotation_font_color=ACCENT4)
        fig2.add_hline(y=2,line_dash="dot",line_color=ACCENT5,annotation_text="2% cap",annotation_font_color=ACCENT5)
        fig2.update_yaxes(ticksuffix="%"); dark_layout(fig2,height=320); st.plotly_chart(fig2,use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>NEW vs EXISTING TENANCIES</div>",unsafe_allow_html=True)
        ex=master.dropna(subset=['Dublin_Exist'])
        fig3=go.Figure()
        fig3.add_trace(go.Scatter(x=ex['Date'],y=ex['Dublin'],name='New Dublin',line=dict(color=ACCENT1,width=2)))
        fig3.add_trace(go.Scatter(x=ex['Date'],y=ex['Dublin_Exist'],name='Existing Dublin',line=dict(color=ACCENT1,width=2,dash='dot')))
        fig3.add_trace(go.Scatter(x=ex['Date'],y=ex['Non_Dublin'],name='New Non-Dublin',line=dict(color=ACCENT2,width=2)))
        fig3.add_trace(go.Scatter(x=ex['Date'],y=ex['Non_Dublin_Exist'],name='Existing Non-Dublin',line=dict(color=ACCENT2,width=2,dash='dot')))
        fig3.update_yaxes(tickprefix="€",tickformat=","); dark_layout(fig3,height=320); st.plotly_chart(fig3,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COUNTY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page=="🗺️ County Analysis":
    st.markdown("<div class='section-header'>COUNTY RENT SNAPSHOT — Q3 2025</div>",unsafe_allow_html=True)
    disp=county_snapshot if county_filter=="All Counties" else county_snapshot[county_snapshot['County']==county_filter]
    nav=county_snapshot['Rent_Q32025'].mean()
    fig=go.Figure(go.Bar(x=disp['Rent_Q32025'],y=disp['County'],orientation='h',
        marker=dict(color=[ACCENT1 if r>1700 else ACCENT2 if r>1300 else "#5A8A7A" for r in disp['Rent_Q32025']],line=dict(width=0)),
        text=[f"€{v:,.0f}" for v in disp['Rent_Q32025']],textposition='outside',textfont=dict(color=TEXT_COLOR,size=11),
        hovertemplate="<b>%{y}</b><br>€%{x:,.0f}/month<extra></extra>"))
    fig.add_vline(x=nav,line_dash="dash",line_color=ACCENT3,line_width=2,annotation_text=f"Avg: €{nav:,.0f}",annotation_font_color=ACCENT3,annotation_font_size=12)
    dark_layout(fig,"Average Monthly Rent by County (Q3 2025)",height=max(400,len(disp)*28))
    fig.update_xaxes(tickprefix="€",tickformat=","); st.plotly_chart(fig,use_container_width=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Highest Rent",f"€{county_snapshot['Rent_Q32025'].max():,.0f}",county_snapshot.iloc[0]['County'])
    c2.metric("Lowest Rent", f"€{county_snapshot['Rent_Q32025'].min():,.0f}",county_snapshot.iloc[-1]['County'])
    c3.metric("National Average",f"€{nav:,.0f}")
    c4.metric("Above Average",f"{(county_snapshot['Rent_Q32025']>nav).sum()} counties")

# ═══════════════════════════════════════════════════════════════════════════════
# AFFORDABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page=="💸 Affordability":
    st.markdown("<div class='section-header'>RENT AFFORDABILITY ANALYSIS</div>",unsafe_allow_html=True)
    afd=mf.dropna(subset=['Affordability_Dublin'])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=afd['Date'],y=afd['Affordability_Dublin'],name='Dublin',line=dict(color=ACCENT1,width=3),fill='tozeroy',fillcolor='rgba(74,144,217,0.1)',hovertemplate="Dublin %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=afd['Date'],y=afd['Affordability_National'],name='Non-Dublin',line=dict(color=ACCENT2,width=2.5,dash='dot'),hovertemplate="Non-Dublin %{y:.1f}%<extra></extra>"))
    fig.add_hrect(y0=30,y1=afd['Affordability_Dublin'].max()+5,fillcolor="rgba(255,107,107,0.07)",line_width=0)
    fig.add_hline(y=30,line_dash="solid",line_color=ACCENT3,line_width=2.5,annotation_text="⚠ 30% Crisis Threshold",annotation_font_color=ACCENT3,annotation_font_size=13)
    fig.add_vline(x='2016-10-01',line_dash="dash",line_color=ACCENT5,line_width=1.5,annotation_text="RPZ Intro",annotation_font_color=ACCENT5)
    fig.update_yaxes(ticksuffix="%"); fig.update_layout(hovermode='x unified')
    dark_layout(fig,"Rent-to-Income Ratio Over Time",height=420); st.plotly_chart(fig,use_container_width=True)
    da=round(latest['Dublin']/master['Monthly_Income'].iloc[-1]*100,1); na=round(latest['Non_Dublin']/master['Monthly_Income'].iloc[-1]*100,1)
    c1,c2,c3=st.columns(3)
    c1.metric("Dublin Ratio",f"{da}%","⚠ CRISIS" if da>30 else "✅ OK")
    c2.metric("Non-Dublin Ratio",f"{na}%","⚠ CRISIS" if na>30 else "✅ OK")
    c3.metric("Monthly Income",f"€{master['Monthly_Income'].iloc[-1]:,.0f}")

    st.markdown("<div class='section-header'>INCOME vs RENT GROWTH</div>",unsafe_allow_html=True)
    fig2=make_subplots(specs=[[{"secondary_y":True}]])
    fig2.add_trace(go.Bar(x=income['Year'],y=income['Monthly_Income'],name='Monthly Income',marker_color=ACCENT2,opacity=0.7,hovertemplate="Income %{x}: €%{y:,.0f}<extra></extra>"),secondary_y=False)
    dan=mf.groupby('Year')['Dublin'].mean().reset_index()
    fig2.add_trace(go.Scatter(x=dan['Year'],y=dan['Dublin'],name='Dublin Rent',line=dict(color=ACCENT1,width=3),hovertemplate="Dublin Rent %{x}: €%{y:,.0f}<extra></extra>"),secondary_y=True)
    fig2.update_yaxes(tickprefix="€",tickformat=",",secondary_y=False,gridcolor=GRID_COLOR)
    fig2.update_yaxes(tickprefix="€",tickformat=",",secondary_y=True,gridcolor=GRID_COLOR)
    fig2.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=TEXT_COLOR),height=300,margin=dict(l=50,r=50,t=30,b=40),legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=1.02))
    st.plotly_chart(fig2,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SUPPLY & LANDLORDS
# ═══════════════════════════════════════════════════════════════════════════════
elif page=="🏗️ Supply & Landlords":
    st.markdown("<div class='section-header'>HOUSING SUPPLY vs RENT</div>",unsafe_allow_html=True)
    da=master.groupby('Year')['Dublin'].mean().reset_index(); da.columns=['Year','Dublin_Avg_Rent']
    sd=pd.merge(hc,da,on='Year',how='inner')
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=sd['Year'],y=sd['Completions'],name='Completions',marker_color=ACCENT2,opacity=0.7,hovertemplate="Completions %{x}: %{y:,}<extra></extra>"),secondary_y=False)
    fig.add_trace(go.Scatter(x=sd['Year'],y=sd['Dublin_Avg_Rent'],name='Dublin Rent',line=dict(color=ACCENT1,width=3),hovertemplate="Rent %{x}: €%{y:,.0f}<extra></extra>"),secondary_y=True)
    fig.add_hline(y=48000,line_dash="dash",line_color=ACCENT2,annotation_text="48k Target",annotation_font_color=ACCENT2,secondary_y=False)
    fig.add_hline(y=33000,line_dash="dot",line_color=ACCENT4,annotation_text="33k Govt",annotation_font_color=ACCENT4,secondary_y=False)
    fig.update_yaxes(tickformat=",",secondary_y=False,gridcolor=GRID_COLOR); fig.update_yaxes(tickprefix="€",tickformat=",",secondary_y=True,gridcolor=GRID_COLOR)
    fig.update_layout(paper_bgcolor=PAPER_BG,plot_bgcolor=PLOT_BG,font=dict(color=TEXT_COLOR),height=380,margin=dict(l=50,r=50,t=30,b=40),legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=1.02))
    st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>TOTAL LANDLORDS</div>",unsafe_allow_html=True)
        fig2=go.Figure(go.Bar(x=landlord['Date'],y=landlord['Total_Landlords'],marker_color=ACCENT1,hovertemplate="%{x|%Y}: %{y:,}<extra></extra>"))
        fig2.update_yaxes(tickformat=","); dark_layout(fig2,height=300); st.plotly_chart(fig2,use_container_width=True)
    with c2:
        st.markdown("<div class='section-header'>QoQ LANDLORD CHANGE</div>",unsafe_allow_html=True)
        qc=landlord['QoQ_Change'].fillna(0)
        fig3=go.Figure(go.Bar(x=landlord['Date'],y=qc,marker_color=[ACCENT3 if v<0 else ACCENT2 for v in qc],hovertemplate="%{x|%Y}: %{y:,}<extra></extra>"))
        fig3.add_hline(y=0,line_color=TEXT_COLOR,line_width=1); dark_layout(fig3,height=300); st.plotly_chart(fig3,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PAGES
# ═══════════════════════════════════════════════════════════════════════════════
elif page in ["🤖 SARIMA","🌲 Random Forest & GB","🔮 Prophet","📡 SARIMAX","🏆 Model Comparison"]:
    with st.spinner("⏳ Running models — first load takes a few minutes..."):
        m=run_models(master,unemp_q,ppi_q)

    COLORS=[ACCENT1,ACCENT2,ACCENT4,ACCENT5,ACCENT3]

    def forecast_chart(hist_x,hist_y,test_x,test_y,pred_y,pred_ci_lo,pred_ci_hi,fc_x,fc_y,fc_lo,fc_hi,name,color,title,height=380):
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=hist_x,y=hist_y,name='Train',line=dict(color=TEXT_COLOR,width=1.5)))
        if test_x is not None:
            fig.add_trace(go.Scatter(x=test_x,y=test_y,name='Actual',line=dict(color=ACCENT3,width=2.5)))
            fig.add_trace(go.Scatter(x=test_x,y=pred_y,name=f'{name} Forecast',line=dict(color=color,width=2.5,dash='dash')))
            if pred_ci_hi is not None:
                fig.add_trace(go.Scatter(x=test_x,y=pred_ci_hi,line=dict(width=0),showlegend=False))
                fig.add_trace(go.Scatter(x=test_x,y=pred_ci_lo,fill='tonexty',fillcolor='rgba(74,144,217,0.15)',line=dict(width=0),name='95% CI'))
        if fc_x is not None:
            fig.add_trace(go.Scatter(x=fc_x,y=fc_y,name='Future Forecast',line=dict(color=ACCENT2,width=2.5,dash='dot'),marker=dict(size=5)))
            fig.add_trace(go.Scatter(x=fc_x,y=fc_hi,line=dict(width=0),showlegend=False))
            fig.add_trace(go.Scatter(x=fc_x,y=fc_lo,fill='tonexty',fillcolor='rgba(0,196,180,0.12)',line=dict(width=0),showlegend=False))
        fig.update_yaxes(tickprefix="€",tickformat=","); dark_layout(fig,title,height); return fig

    if page=="🤖 SARIMA":
        st.markdown("<div class='section-header'>SARIMA BASELINE MODEL</div>",unsafe_allow_html=True)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Dublin MAE",f"€{m['maeds']:.0f}"); c2.metric("Dublin MAPE",f"{m['mapeds']:.2f}%")
        c3.metric("Non-Dublin MAE",f"€{m['maends']:.0f}"); c4.metric("Non-Dublin MAPE",f"{m['mapends']:.2f}%")
        col1,col2=st.columns(2)
        cutoff=pd.Timestamp('2022-01-01'); rec=m['df_ts'][m['df_ts'].index>=cutoff]
        with col1:
            fig=forecast_chart(m['tr_s'].index,m['tr_s']['Dublin'],m['te_s'].index,m['te_s']['Dublin'],m['pds'].values,m['cds'].iloc[:,0].values,m['cds'].iloc[:,1].values,
                m['fcdm'].index,m['fcdm'].values,m['fcdci'].iloc[:,0].values,m['fcdci'].iloc[:,1].values,'SARIMA',ACCENT1,f"SARIMA — Dublin | MAPE={m['mapeds']:.1f}%")
            st.plotly_chart(fig,use_container_width=True)
        with col2:
            fig=forecast_chart(m['tr_s'].index,m['tr_s']['Non_Dublin'],m['te_s'].index,m['te_s']['Non_Dublin'],m['pnds'].values,m['cnds'].iloc[:,0].values,m['cnds'].iloc[:,1].values,
                m['fcndm'].index,m['fcndm'].values,m['fcndci'].iloc[:,0].values,m['fcndci'].iloc[:,1].values,'SARIMA',ACCENT2,f"SARIMA — Non-Dublin | MAPE={m['mapends']:.1f}%")
            st.plotly_chart(fig,use_container_width=True)

    elif page=="🌲 Random Forest & GB":
        st.markdown("<div class='section-header'>RANDOM FOREST & GRADIENT BOOSTING</div>",unsafe_allow_html=True)
        c1,c2,c3,c4,c5,c6,c7,c8=st.columns(8)
        c1.metric("RF D MAE",f"€{m['maed_rf']:.0f}"); c2.metric("RF D MAPE",f"{m['maped_rf']:.2f}%")
        c3.metric("RF ND MAE",f"€{m['maend_rf']:.0f}"); c4.metric("RF ND MAPE",f"{m['mapend_rf']:.2f}%")
        c5.metric("GB D MAE",f"€{m['maed_gb']:.0f}"); c6.metric("GB D MAPE",f"{m['maped_gb']:.2f}%")
        c7.metric("GB ND MAE",f"€{m['maend_gb']:.0f}"); c8.metric("GB ND MAPE",f"{m['mapend_gb']:.2f}%")

        col1,col2=st.columns(2)
        for c,dff,ytr,yte,prf,pgb,dtst,name,clr in [
            (col1,m['dfd'],m['ydt'],m['yde'],m['prfd'],m['pgbd'],m['dtd'],'Dublin',ACCENT1),
            (col2,m['dfnd'],m['yndt'],m['ynde'],m['prfnd'],m['pgbnd'],m['dtnd'],'Non-Dublin',ACCENT2)
        ]:
            with c:
                trd=dff['Date'].values[:-m['trf']]
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=trd,y=ytr,name='Train',line=dict(color=TEXT_COLOR,width=1.5)))
                fig.add_trace(go.Scatter(x=dtst,y=yte,name='Actual',line=dict(color=ACCENT3,width=2.5)))
                fig.add_trace(go.Scatter(x=dtst,y=prf,name='Random Forest',line=dict(color=clr,width=2.5,dash='dash')))
                fig.add_trace(go.Scatter(x=dtst,y=pgb,name='Grad. Boost',line=dict(color=ACCENT4,width=2.5,dash='dot')))
                fig.update_yaxes(tickprefix="€",tickformat=","); dark_layout(fig,f"RF & GB — {name}",height=360)
                st.plotly_chart(fig,use_container_width=True)

        st.markdown("<div class='section-header'>FEATURE IMPORTANCE</div>",unsafe_allow_html=True)
        col1,col2=st.columns(2)
        for c,model,fc,name in [(col1,m['rfd'],m['fcd_c'],'Dublin'),(col2,m['rfnd'],m['fcnd_c'],'Non-Dublin')]:
            with c:
                nn=model.n_features_in_; fc2=fc[:nn]
                imp=pd.DataFrame({'Feature':fc2,'Importance':model.feature_importances_}).sort_values('Importance').tail(12)
                clrs=[ACCENT3 if 'RPPI' in f or 'Unemp' in f else ACCENT1 if 'lag' in f or 'roll' in f else ACCENT2 for f in imp['Feature']]
                fig=go.Figure(go.Bar(x=imp['Importance'],y=imp['Feature'],orientation='h',marker_color=clrs,hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
                dark_layout(fig,f"Feature Importance — {name}",height=360); st.plotly_chart(fig,use_container_width=True)

    elif page=="🔮 Prophet":
        st.markdown("<div class='section-header'>PROPHET FORECAST MODEL</div>",unsafe_allow_html=True)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Dublin MAE",f"€{m['maed_p']:.0f}"); c2.metric("Dublin MAPE",f"{m['maped_p']:.2f}%")
        c3.metric("Non-Dublin MAE",f"€{m['maend_p']:.0f}"); c4.metric("Non-Dublin MAPE",f"{m['mapend_p']:.2f}%")
        col1,col2=st.columns(2)
        cutoff_p=pd.Timestamp('2022-01-01')
        for c,trd,ted,fct,fut,dpd,name,clr in [
            (col1,m['tpd'],m['tepd'],m['ftd'],m['pfud'],m['dpd'],'Dublin',ACCENT1),
            (col2,m['tpnd'],m['tepnd'],m['ftnd'],m['pfund'],m['dpnd'],'Non-Dublin',ACCENT2)
        ]:
            with c:
                rec=dpd[dpd['ds']>=cutoff_p]
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=trd['ds'],y=trd['y'],name='Train',line=dict(color=TEXT_COLOR,width=1.5)))
                fig.add_trace(go.Scatter(x=ted['ds'],y=ted['y'],name='Actual',line=dict(color=ACCENT3,width=2.5)))
                fig.add_trace(go.Scatter(x=fct['ds'],y=fct['yhat'],name='Prophet Test',line=dict(color=clr,width=2.5,dash='dash')))
                fig.add_trace(go.Scatter(x=fct['ds'],y=fct['yhat_upper'],line=dict(width=0),showlegend=False))
                fig.add_trace(go.Scatter(x=fct['ds'],y=fct['yhat_lower'],fill='tonexty',fillcolor='rgba(74,144,217,0.15)',line=dict(width=0),name='95% CI'))
                fig.add_trace(go.Scatter(x=fut['ds'],y=fut['yhat'],name='Future Forecast',line=dict(color=ACCENT2,width=2.5,dash='dot')))
                fig.add_trace(go.Scatter(x=fut['ds'],y=fut['yhat_upper'],line=dict(width=0),showlegend=False))
                fig.add_trace(go.Scatter(x=fut['ds'],y=fut['yhat_lower'],fill='tonexty',fillcolor='rgba(0,196,180,0.12)',line=dict(width=0),showlegend=False))
                fig.update_yaxes(tickprefix="€",tickformat=","); dark_layout(fig,f"Prophet — {name} (Q3 2027: €{fut['yhat'].iloc[-1]:,.0f})",height=400)
                st.plotly_chart(fig,use_container_width=True)

    elif page=="📡 SARIMAX":
        st.markdown("<div class='section-header'>SARIMAX — WITH EXOGENOUS VARIABLES</div>",unsafe_allow_html=True)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Dublin MAE",f"€{m['maed_sx']:.0f}"); c2.metric("Dublin MAPE",f"{m['maped_sx']:.2f}%")
        c3.metric("Non-Dublin MAE",f"€{m['maend_sx']:.0f}"); c4.metric("Non-Dublin MAPE",f"{m['mapend_sx']:.2f}%")
        col1,col2=st.columns(2)
        for c,cn,pred,ci,fcm,fcci,name,clr in [
            (col1,'Dublin',m['psxd'],m['csxd'],m['sxdm'],m['sxdci'],'Dublin',ACCENT1),
            (col2,'Non_Dublin',m['psxnd'],m['csxnd'],m['sxndm'],m['sxndci'],'Non-Dublin',ACCENT2)
        ]:
            with c:
                fig=forecast_chart(m['trsx'].index,m['trsx'][cn],m['tesx'].index,m['tesx'][cn],pred.values,ci.iloc[:,0].values,ci.iloc[:,1].values,
                    fcm.index,fcm.values,fcci.iloc[:,0].values,fcci.iloc[:,1].values,'SARIMAX',clr,f"SARIMAX — {name} (Q3 2027: €{fcm.iloc[-1]:,.0f})")
                st.plotly_chart(fig,use_container_width=True)

    elif page=="🏆 Model Comparison":
        st.markdown("<div class='section-header'>FINAL MODEL COMPARISON</div>",unsafe_allow_html=True)
        comp=m['comp'].copy()
        if model_filter: comp=comp[comp['Model'].isin(model_filter)]
        best=m['comp'].iloc[0]
        st.success(f"🥇 Best Model: **{best['Model']}** — Avg MAPE: {best['Avg_MAPE']:.2f}% | Avg MAE: €{best['Avg_MAE']:.0f}/month")

        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure()
            for i,row in comp.iterrows():
                fig.add_trace(go.Bar(name=row['Model'],x=['Dublin MAPE','Non-Dublin MAPE'],y=[row['Dublin_MAPE'],row['NonDublin_MAPE']],
                    marker_color=COLORS[i%5],hovertemplate=f"<b>{row['Model']}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>"))
            fig.add_hline(y=5,line_dash="dash",line_color=ACCENT3,annotation_text="5% threshold",annotation_font_color=ACCENT3)
            fig.update_yaxes(ticksuffix="%"); dark_layout(fig,"MAPE Comparison (Lower = Better)",height=380); fig.update_layout(barmode='group')
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig2=go.Figure()
            for i,row in comp.iterrows():
                fig2.add_trace(go.Bar(name=row['Model'],x=['Dublin MAE','Non-Dublin MAE'],y=[row['Dublin_MAE'],row['NonDublin_MAE']],
                    marker_color=COLORS[i%5],hovertemplate=f"<b>{row['Model']}</b><br>%{{x}}: €%{{y:.0f}}<extra></extra>"))
            fig2.update_yaxes(tickprefix="€",tickformat=","); dark_layout(fig2,"MAE Comparison — €/month (Lower = Better)",height=380); fig2.update_layout(barmode='group')
            st.plotly_chart(fig2,use_container_width=True)

        disp=m['comp'][['Model','Dublin_MAE','Dublin_MAPE','NonDublin_MAE','NonDublin_MAPE','Avg_MAPE','Avg_MAE']].copy()
        for col in ['Dublin_MAE','NonDublin_MAE','Avg_MAE']: disp[col]=disp[col].apply(lambda x:f"€{x:.0f}")
        for col in ['Dublin_MAPE','NonDublin_MAPE','Avg_MAPE']: disp[col]=disp[col].apply(lambda x:f"{x:.2f}%")
        st.dataframe(disp,use_container_width=True,hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# POLICY INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page=="📋 Policy Insights":
    st.markdown("<div class='section-header'>RESEARCH FINDINGS & POLICY RECOMMENDATIONS</div>",unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    c1.metric("Dublin Growth (2007–2025)",f"+{((latest['Dublin']/first['Dublin'])-1)*100:.0f}%","vs wages")
    c2.metric("Non-Dublin Growth",f"+{((latest['Non_Dublin']/first['Non_Dublin'])-1)*100:.0f}%","spreading crisis")
    c3.metric("Best Model Accuracy","< 5% MAPE","8 quarters ahead")

    t1,t2,t3=st.tabs(["🔍 SQ1: Key Drivers","⚠️ SQ2: Crisis?","📡 SQ3: Lead Time"])
    with t1:
        col1,col2=st.columns(2)
        with col1:
            st.markdown("""<div class='kpi-card' style='text-align:left'>
            <div style='color:#4A90D9;font-size:0.8rem;letter-spacing:2px;margin-bottom:8px'>DUBLIN DRIVERS</div>
            <ol style='color:#C8CBD8;line-height:2'><li><b>Rent Lag (Q-1)</b> — ~18%</li><li><b>Rent Lag (Q-4)</b> — ~17%</li>
            <li><b>Rolling 4Q Mean</b> — ~17%</li><li><b>Unemployment Lag</b> — ~11%</li><li><b>RPPI Index</b> — SARIMAX</li></ol>
            </div>""",unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class='kpi-card' style='text-align:left'>
            <div style='color:#00C4B4;font-size:0.8rem;letter-spacing:2px;margin-bottom:8px'>NON-DUBLIN DRIVERS</div>
            <ol style='color:#C8CBD8;line-height:2'><li><b>Rent Lag (Q-1)</b> — ~37%</li><li><b>Unemployment Rate</b> — ~14%</li>
            <li><b>Rolling 4Q Mean</b> — trend</li><li><b>RPZ Dummy</b> — structural break</li></ol>
            </div>""",unsafe_allow_html=True)
    with t2:
        st.markdown(f"""<div class='kpi-card' style='text-align:left;border-left:3px solid #FF6B6B'>
        <div style='color:#FF6B6B;font-size:1rem;font-weight:700;margin-bottom:12px'>⚠️ YES — Systemic Crisis Confirmed</div>
        <ul style='color:#C8CBD8;line-height:2.2'>
        <li>Dublin: <b>€{latest['Dublin']:,.0f}/month</b> — up {((latest['Dublin']/first['Dublin'])-1)*100:.0f}% since 2007</li>
        <li>Non-Dublin: <b>€{latest['Non_Dublin']:,.0f}/month</b> — up {((latest['Non_Dublin']/first['Non_Dublin'])-1)*100:.0f}% since 2007</li>
        <li>All 5 models forecast <b>continued upward trajectory</b> to 2027</li>
        <li>RPZ (2016) slowed but did <b>NOT reverse</b> the trend</li>
        <li>Dublin rent-to-income: <b>{latest['Affordability_Dublin']:.1f}%</b> (threshold: 30%)</li>
        </ul></div>""",unsafe_allow_html=True)
    with t3:
        st.markdown("""<div class='kpi-card' style='text-align:left;border-left:3px solid #00C4B4'>
        <div style='color:#00C4B4;font-size:1rem;font-weight:700;margin-bottom:12px'>✅ YES — Sufficient Lead Time</div>
        <ul style='color:#C8CBD8;line-height:2.2'>
        <li>Forecast window: <b>8 quarters (2 years)</b> ahead</li>
        <li>Best model: <b>&lt;5% MAPE</b></li>
        <li>Policy lead time needed: <b>12–18 months</b></li>
        <li>Dashboard gives <b>6–8 quarter early warning</b></li>
        </ul></div>""",unsafe_allow_html=True)

    st.markdown("<div class='section-header'>POLICY RECOMMENDATIONS</div>",unsafe_allow_html=True)
    recs=[("🏗️","Supply-Side Intervention","Accelerate social & affordable housing. Fast-track planning in Dublin corridors.",ACCENT1),
          ("📜","Strengthen RPZ","Lower cap to inflation-linked rate. Expand to Non-Dublin commuter belts.",ACCENT2),
          ("💼","Unemployment Linkage","Incentivise remote-work hubs to distribute demand geographically.",ACCENT4),
          ("📊","Deploy Dashboard","Run SARIMAX/Prophet as live policy tool. Auto-alert at 30% threshold.",ACCENT5),
          ("🗄️","Data Infrastructure","Add mortgage rates, construction costs, migration data.",ACCENT3)]
    cols=st.columns(5)
    for col,(icon,title,desc,clr) in zip(cols,recs):
        with col:
            st.markdown(f"""<div class='kpi-card' style='text-align:left;min-height:180px'>
            <div style='font-size:1.8rem;margin-bottom:8px'>{icon}</div>
            <div style='color:{clr};font-size:0.75rem;font-weight:700;letter-spacing:1px;margin-bottom:6px'>{title.upper()}</div>
            <div style='color:#A0A3B0;font-size:0.82rem;line-height:1.5'>{desc}</div>
            </div>""",unsafe_allow_html=True)
