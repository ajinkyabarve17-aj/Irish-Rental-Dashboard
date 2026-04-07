import itertools
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st
from openpyxl import load_workbook

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Irish Rental Market Dashboard",
    page_icon="🏠",
    layout="wide",
)

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ── Helper ────────────────────────────────────────────────────────────────────
def quarter_to_date(q):
    parts = str(q).strip().split()
    qnum  = int(parts[0][1])
    year  = int(parts[1])
    month = (qnum - 1) * 3 + 1
    return pd.Timestamp(year=year, month=month, day=1)

# ── Data loading (cached so it only runs once) ────────────────────────────────
@st.cache_data
def load_all_data():
    # Dataset 1 — RTB Regional Rent
    wb = load_workbook('RTB_Regional_Rent_TimeSeries_2007_2025.xlsx', read_only=True)
    ws_new   = wb['RTBRI Q325 New']
    rows_new = list(ws_new.iter_rows(values_only=True))
    data_new = [r for r in rows_new[2:] if r[0] and str(r[0]).startswith('Q')]
    rtb_new  = pd.DataFrame(data_new, columns=['Period', 'Dublin', 'Non_Dublin', 'GDA_excl_Dublin', 'Outside_GDA'])

    ws_exist   = wb['RTBRI Q325 Existing']
    rows_exist = list(ws_exist.iter_rows(values_only=True))
    data_exist = [r for r in rows_exist[2:] if r[0] and str(r[0]).startswith('Q')]
    rtb_exist  = pd.DataFrame(data_exist, columns=['Period', 'Dublin_Exist', 'Non_Dublin_Exist', 'GDA_Exist', 'Outside_GDA_Exist'])

    rtb_new['Date']   = rtb_new['Period'].apply(quarter_to_date)
    rtb_exist['Date'] = rtb_exist['Period'].apply(quarter_to_date)

    rtb = pd.merge(rtb_new, rtb_exist[['Date', 'Dublin_Exist', 'Non_Dublin_Exist']], on='Date', how='left')
    rtb = rtb.sort_values('Date').reset_index(drop=True)
    for col in ['Dublin', 'Non_Dublin', 'GDA_excl_Dublin', 'Outside_GDA']:
        rtb[col] = pd.to_numeric(rtb[col], errors='coerce')

    # Dataset 2 — County Snapshot
    wb2       = load_workbook('RTB_County_Rent_Snapshot_Q32025.xlsx', read_only=True)
    ws_county = wb2['RIQ325 New']
    rows_county = list(ws_county.iter_rows(values_only=True))
    county_data = []
    for row in rows_county[2:]:
        if row[0] and isinstance(row[0], str) and row[1] and isinstance(row[1], (int, float)):
            county_data.append({'County': row[0], 'Rent_Q32025': round(float(row[1]), 2)})
    county_snapshot = (pd.DataFrame(county_data)
                       .sort_values('Rent_Q32025', ascending=False)
                       .reset_index(drop=True))

    # Dataset 3 — Unemployment
    unemp_df    = pd.read_excel('CSO_Unemployment_Monthly_1998_2026.xlsx', sheet_name='Unpivoted')
    unemp_label = 'Seasonally Adjusted Monthly Unemployment Rate (%)'
    unemp_mask  = (
        (unemp_df['Statistic Label'].str.strip() == unemp_label) &
        (unemp_df['Age Group'].str.strip() == '15 - 74 years') &
        (unemp_df['Sex'].str.strip() == 'Both sexes')
    )
    unemp = unemp_df[unemp_mask][['Month', 'VALUE']].copy()
    if unemp.empty:
        unemp_mask = (
            (unemp_df['Statistic Label'].str.strip() == 'Seasonally Adjusted Monthly Unemployment Rate') &
            (unemp_df['Age Group'].str.strip() == '15 - 74 years') &
            (unemp_df['Sex'].str.strip() == 'Both sexes')
        )
        unemp = unemp_df[unemp_mask][['Month', 'VALUE']].copy()
    unemp.columns = ['Month', 'Unemployment_Rate']
    unemp['Date'] = pd.to_datetime(unemp['Month'], format='%Y %B', errors='coerce')
    unemp = unemp.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    unemp_q = unemp.set_index('Date')['Unemployment_Rate'].resample('QS').mean().reset_index()
    unemp_q.columns = ['Date', 'Unemployment_Rate']

    # Dataset 4 — Property Price Index
    ppi_df   = pd.read_excel('CSO_PropertyPriceIndex_2005_2026.xlsx', sheet_name='Unpivoted')
    ppi_mask = (
        (ppi_df['Statistic Label'] == 'Residential Property Price Index') &
        (ppi_df['Type of Residential Property'] == 'National - all residential properties')
    )
    ppi = ppi_df[ppi_mask][['Month', 'VALUE']].copy()
    ppi.columns = ['Month', 'RPPI']
    ppi['Date'] = pd.to_datetime(ppi['Month'], format='%Y %B', errors='coerce')
    ppi = ppi.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    ppi_q = ppi.set_index('Date')['RPPI'].resample('QS').mean().reset_index()
    ppi_q.columns = ['Date', 'RPPI']

    # Dataset 5 — Housing Completions
    hc_raw     = pd.read_excel('CSO_HousingCompletions_Quarterly.xlsx', sheet_name='Unpivoted')
    region_col = hc_raw.columns[7]
    year_col   = hc_raw.columns[3]
    value_col  = hc_raw.columns[-1]
    hc = hc_raw[hc_raw[region_col].astype(str).str.contains('State', na=False)][[year_col, value_col]].copy()
    hc.columns = ['Year', 'Completions']
    hc = hc.groupby('Year')['Completions'].sum().reset_index()
    hc['Year'] = pd.to_numeric(hc['Year'], errors='coerce')
    hc = hc.dropna().sort_values('Year').reset_index(drop=True)

    # Dataset 6 — Household Income
    income_df   = pd.read_excel('CSO_HouseholdIncome_SILC_2020_2025.xlsx', sheet_name='Unpivoted')
    income_mask = (
        (income_df['Statistic Label'] == 'Mean Nominal Household Disposable Income (Euro)') &
        (income_df['Sex'] == 'Both sexes')
    )
    income = income_df[income_mask][['Year', 'VALUE']].copy()
    income.columns = ['Year', 'Mean_Annual_Income']
    income['Year'] = pd.to_numeric(income['Year'], errors='coerce')
    income['Monthly_Income'] = income['Mean_Annual_Income'] / 12
    income = income.sort_values('Year').reset_index(drop=True)

    # Dataset 7 — Landlord Numbers
    wb7   = load_workbook('RTB_Landlord_Numbers_Q22023_Q42025.xlsx', read_only=True)
    ws7   = wb7['RTB LL Total Q425']
    rows7 = list(ws7.iter_rows(values_only=True))
    landlord_quarters = list(rows7[1][1:])
    landlord_values   = list(rows7[2][1:])
    landlord = pd.DataFrame({'Quarter': landlord_quarters, 'Total_Landlords': landlord_values}).dropna()
    landlord['Date']            = landlord['Quarter'].apply(quarter_to_date)
    landlord['Total_Landlords'] = pd.to_numeric(landlord['Total_Landlords'])
    landlord['QoQ_Change']      = landlord['Total_Landlords'].diff()
    landlord = landlord.sort_values('Date').reset_index(drop=True)

    # Master dataset
    master = rtb[['Date', 'Period', 'Dublin', 'Non_Dublin', 'GDA_excl_Dublin', 'Outside_GDA']].copy()
    master = pd.merge(master, unemp_q, on='Date', how='left')
    master = pd.merge(master, ppi_q,   on='Date', how='left')
    master['Year'] = master['Date'].dt.year
    master = pd.merge(master, income[['Year', 'Monthly_Income']], on='Year', how='left')
    master['Monthly_Income'] = master['Monthly_Income'].ffill().bfill()
    master['Affordability_Dublin']   = (master['Dublin']     / master['Monthly_Income'] * 100).round(2)
    master['Affordability_National'] = (master['Non_Dublin'] / master['Monthly_Income'] * 100).round(2)
    master['RPZ_Active'] = (master['Date'] >= '2016-10-01').astype(int)
    master_s = master.sort_values('Date').copy()
    master['Dublin_YoY']    = master_s['Dublin'].pct_change(4).values * 100
    master['NonDublin_YoY'] = master_s['Non_Dublin'].pct_change(4).values * 100

    return master, county_snapshot, landlord, hc, income, unemp_q, ppi_q, rtb

# ── ML Models (cached separately — these are slow) ───────────────────────────
@st.cache_data
def run_models(_master, _unemp_q, _ppi_q):
    # ── SARIMA ────────────────────────────────────────────────────────────────
    df_ts = _master[['Date', 'Dublin', 'Non_Dublin']].copy().sort_values('Date').set_index('Date')
    test_size_s = 15
    n_ts    = len(df_ts)
    train_s = df_ts.iloc[:n_ts - test_size_s]
    test_s  = df_ts.iloc[n_ts - test_size_s:]

    def sarima_grid_search(series, label):
        p_vals = [0, 1, 2]; d_vals = [1]; q_vals = [0, 1, 2]
        P_vals = [0, 1];    D_vals = [1]; Q_vals = [0, 1]; s = 4
        best_aic = np.inf; best_order = (1, 1, 1); best_seasonal = (1, 1, 1, 4)
        for p, d, q in itertools.product(p_vals, d_vals, q_vals):
            for P, D, Q in itertools.product(P_vals, D_vals, Q_vals):
                try:
                    m = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, s),
                                enforce_stationarity=False, enforce_invertibility=False)
                    r = m.fit(disp=False)
                    if r.aic < best_aic:
                        best_aic = r.aic; best_order = (p, d, q); best_seasonal = (P, D, Q, s)
                except Exception:
                    continue
        return best_order, best_seasonal

    best_order_d,  best_seasonal_d  = sarima_grid_search(train_s['Dublin'],    'Dublin')
    best_order_nd, best_seasonal_nd = sarima_grid_search(train_s['Non_Dublin'],'Non-Dublin')

    sarima_d_fit  = SARIMAX(train_s['Dublin'],    order=best_order_d,  seasonal_order=best_seasonal_d,
                             enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_nd_fit = SARIMAX(train_s['Non_Dublin'],order=best_order_nd, seasonal_order=best_seasonal_nd,
                             enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    def eval_sarima(fit, test_col):
        po = fit.get_forecast(steps=len(test_s))
        pm = po.predicted_mean; pm.index = test_s.index
        ci = po.conf_int(alpha=0.05); ci.index = test_s.index
        mae  = mean_absolute_error(test_s[test_col], pm)
        rmse = np.sqrt(mean_squared_error(test_s[test_col], pm))
        mape = np.mean(np.abs((test_s[test_col] - pm) / test_s[test_col])) * 100
        return pm, ci, mae, rmse, mape

    pred_d_s,  ci_d_s,  mae_d_s,  rmse_d_s,  mape_d_s  = eval_sarima(sarima_d_fit,  'Dublin')
    pred_nd_s, ci_nd_s, mae_nd_s, rmse_nd_s, mape_nd_s = eval_sarima(sarima_nd_fit, 'Non_Dublin')

    # Future SARIMA forecast
    sarima_d_full  = SARIMAX(df_ts['Dublin'],    order=best_order_d,  seasonal_order=best_seasonal_d,
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_nd_full = SARIMAX(df_ts['Non_Dublin'],order=best_order_nd, seasonal_order=best_seasonal_nd,
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    n_forecast   = 8
    future_dates = pd.date_range(start=df_ts.index[-1] + pd.DateOffset(months=3), periods=n_forecast, freq='QS')
    fc_d  = sarima_d_full.get_forecast(steps=n_forecast);  fc_d_mean = fc_d.predicted_mean;  fc_d_mean.index = future_dates
    fc_nd = sarima_nd_full.get_forecast(steps=n_forecast); fc_nd_mean= fc_nd.predicted_mean; fc_nd_mean.index= future_dates
    fc_d_ci  = fc_d.conf_int(alpha=0.05);  fc_d_ci.index  = future_dates
    fc_nd_ci = fc_nd.conf_int(alpha=0.05); fc_nd_ci.index = future_dates

    # ── Random Forest + Gradient Boosting ────────────────────────────────────
    df_rf = _master[['Date', 'Dublin', 'Non_Dublin']].copy()
    df_rf = df_rf.merge(_unemp_q, on='Date', how='left').merge(_ppi_q, on='Date', how='left')
    df_rf['RPZ'] = (df_rf['Date'] >= '2016-10-01').astype(int)

    def make_features(df_in, target_col, n_lags=4):
        d = df_in.copy()
        for lag in range(1, n_lags + 1):
            d[f'{target_col}_lag{lag}'] = d[target_col].shift(lag)
        d[f'{target_col}_roll4_mean'] = d[target_col].shift(1).rolling(4).mean()
        d[f'{target_col}_roll4_std']  = d[target_col].shift(1).rolling(4).std()
        d[f'{target_col}_yoy']        = d[target_col].pct_change(4) * 100
        d['Quarter']    = d['Date'].dt.quarter
        d['RPPI_lag1']  = d['RPPI'].shift(1)
        d['Unemp_lag1'] = d['Unemployment_Rate'].shift(1)
        return d.dropna()

    df_feat_d  = make_features(df_rf, 'Dublin')
    df_feat_nd = make_features(df_rf, 'Non_Dublin')
    exclude    = {'Date', 'Period', 'Dublin', 'Non_Dublin', 'GDA', 'Outside_GDA'}
    feat_cols_d  = [c for c in df_feat_d.columns  if c not in exclude]
    feat_cols_nd = [c for c in df_feat_nd.columns if c not in exclude]

    test_size_rf = 12
    df_d_clean  = df_feat_d[feat_cols_d  + ['Dublin',     'Date']].dropna()
    df_nd_clean = df_feat_nd[feat_cols_nd + ['Non_Dublin', 'Date']].dropna()

    X_train_d, X_test_d   = df_d_clean[feat_cols_d].values[:-test_size_rf],  df_d_clean[feat_cols_d].values[-test_size_rf:]
    y_train_d, y_test_d   = df_d_clean['Dublin'].values[:-test_size_rf],     df_d_clean['Dublin'].values[-test_size_rf:]
    dates_test_d          = df_d_clean['Date'].values[-test_size_rf:]

    X_train_nd, X_test_nd = df_nd_clean[feat_cols_nd].values[:-test_size_rf], df_nd_clean[feat_cols_nd].values[-test_size_rf:]
    y_train_nd, y_test_nd = df_nd_clean['Non_Dublin'].values[:-test_size_rf], df_nd_clean['Non_Dublin'].values[-test_size_rf:]
    dates_test_nd         = df_nd_clean['Date'].values[-test_size_rf:]

    rf_d = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_split=3, random_state=42, n_jobs=-1)
    rf_d.fit(X_train_d, y_train_d); pred_d_rf = rf_d.predict(X_test_d)
    mae_d_rf  = mean_absolute_error(y_test_d, pred_d_rf)
    rmse_d_rf = np.sqrt(mean_squared_error(y_test_d, pred_d_rf))
    mape_d_rf = np.mean(np.abs((y_test_d - pred_d_rf) / y_test_d)) * 100

    rf_nd = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_split=3, random_state=42, n_jobs=-1)
    rf_nd.fit(X_train_nd, y_train_nd); pred_nd_rf = rf_nd.predict(X_test_nd)
    mae_nd_rf  = mean_absolute_error(y_test_nd, pred_nd_rf)
    rmse_nd_rf = np.sqrt(mean_squared_error(y_test_nd, pred_nd_rf))
    mape_nd_rf = np.mean(np.abs((y_test_nd - pred_nd_rf) / y_test_nd)) * 100

    gb_d = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    gb_d.fit(X_train_d, y_train_d); pred_d_gb = gb_d.predict(X_test_d)
    mae_d_gb  = mean_absolute_error(y_test_d, pred_d_gb)
    rmse_d_gb = np.sqrt(mean_squared_error(y_test_d, pred_d_gb))
    mape_d_gb = np.mean(np.abs((y_test_d - pred_d_gb) / y_test_d)) * 100

    gb_nd = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    gb_nd.fit(X_train_nd, y_train_nd); pred_nd_gb = gb_nd.predict(X_test_nd)
    mae_nd_gb  = mean_absolute_error(y_test_nd, pred_nd_gb)
    rmse_nd_gb = np.sqrt(mean_squared_error(y_test_nd, pred_nd_gb))
    mape_nd_gb = np.mean(np.abs((y_test_nd - pred_nd_gb) / y_test_nd)) * 100

    # ── Prophet ───────────────────────────────────────────────────────────────
    try:
        from prophet import Prophet
    except ImportError:
        import subprocess; subprocess.run(['pip', 'install', 'prophet', '-q'], check=False)
        from prophet import Prophet

    df_p_base = df_rf[['Date', 'Dublin', 'Non_Dublin', 'Unemployment_Rate', 'RPPI']].copy()
    df_p_d    = df_p_base[['Date', 'Dublin',     'Unemployment_Rate', 'RPPI']].rename(columns={'Date':'ds','Dublin':'y'}).dropna()
    df_p_nd   = df_p_base[['Date', 'Non_Dublin', 'Unemployment_Rate', 'RPPI']].rename(columns={'Date':'ds','Non_Dublin':'y'}).dropna()

    test_size_p = 12
    train_p_d   = df_p_d.iloc[:-test_size_p];  test_p_d  = df_p_d.iloc[-test_size_p:]
    train_p_nd  = df_p_nd.iloc[:-test_size_p]; test_p_nd = df_p_nd.iloc[-test_size_p:]

    known_cp = ['2016-10-01', '2020-01-01']
    def build_prophet(train_df, regressors):
        m = Prophet(changepoints=known_cp, changepoint_prior_scale=0.05,
                    seasonality_mode='additive', yearly_seasonality=False,
                    weekly_seasonality=False, daily_seasonality=False)
        m.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        for r in regressors: m.add_regressor(r)
        m.fit(train_df); return m

    regressors = ['Unemployment_Rate', 'RPPI']
    m_p_d  = build_prophet(train_p_d,  regressors)
    m_p_nd = build_prophet(train_p_nd, regressors)

    fc_test_d  = m_p_d.predict(test_p_d)
    fc_test_nd = m_p_nd.predict(test_p_nd)
    mae_d_p   = mean_absolute_error(test_p_d['y'],  fc_test_d['yhat'])
    rmse_d_p  = np.sqrt(mean_squared_error(test_p_d['y'], fc_test_d['yhat']))
    mape_d_p  = np.mean(np.abs((test_p_d['y'].values - fc_test_d['yhat'].values) / test_p_d['y'].values)) * 100
    mae_nd_p  = mean_absolute_error(test_p_nd['y'],  fc_test_nd['yhat'])
    rmse_nd_p = np.sqrt(mean_squared_error(test_p_nd['y'], fc_test_nd['yhat']))
    mape_nd_p = np.mean(np.abs((test_p_nd['y'].values - fc_test_nd['yhat'].values) / test_p_nd['y'].values)) * 100

    # Prophet future forecast
    m_p_d_full  = build_prophet(df_p_d,  regressors)
    m_p_nd_full = build_prophet(df_p_nd, regressors)
    last_unemp = df_rf['Unemployment_Rate'].dropna().iloc[-1]
    last_rppi  = df_rf['RPPI'].dropna().iloc[-1]
    last_date  = df_rf['Date'].iloc[-1]
    future_dates_p = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=n_forecast, freq='QS')
    future_df_p    = pd.DataFrame({'ds': future_dates_p, 'Unemployment_Rate': last_unemp, 'RPPI': last_rppi})
    future_d_full  = pd.concat([df_p_d[['ds'] + regressors], future_df_p], ignore_index=True)
    future_nd_full = pd.concat([df_p_nd[['ds'] + regressors], future_df_p], ignore_index=True)
    fc_fut_d  = m_p_d_full.predict(future_d_full)
    fc_fut_nd = m_p_nd_full.predict(future_nd_full)
    p_future_d  = fc_fut_d[fc_fut_d['ds'].isin(future_dates_p)]
    p_future_nd = fc_fut_nd[fc_fut_nd['ds'].isin(future_dates_p)]

    # ── SARIMAX ───────────────────────────────────────────────────────────────
    df_sx = df_rf.set_index('Date')
    test_size_sx = 15; n_sx = len(df_sx)
    train_sx = df_sx.iloc[:n_sx - test_size_sx]; test_sx = df_sx.iloc[n_sx - test_size_sx:]
    scaler    = StandardScaler()
    exog_cols = ['Unemployment_Rate', 'RPPI']
    train_exog = pd.DataFrame(scaler.fit_transform(train_sx[exog_cols]), index=train_sx.index, columns=exog_cols)
    test_exog  = pd.DataFrame(scaler.transform(test_sx[exog_cols]),      index=test_sx.index,  columns=exog_cols)

    def sarimax_gs(series, exog_df, label):
        p_vals=[0,1,2]; d_vals=[1]; q_vals=[0,1,2]; P_vals=[0,1]; D_vals=[1]; Q_vals=[0,1]; s=4
        best_aic=np.inf; bo=(1,1,1); bs=(1,1,1,4)
        for p,d,q in itertools.product(p_vals,d_vals,q_vals):
            for P,D,Q in itertools.product(P_vals,D_vals,Q_vals):
                try:
                    m=SARIMAX(series,exog=exog_df,order=(p,d,q),seasonal_order=(P,D,Q,s),
                              enforce_stationarity=False,enforce_invertibility=False)
                    r=m.fit(disp=False,maxiter=200)
                    if r.aic<best_aic: best_aic=r.aic; bo=(p,d,q); bs=(P,D,Q,s)
                except: continue
        return bo, bs

    bo_sx_d,  bs_sx_d  = sarimax_gs(train_sx['Dublin'],    train_exog, 'Dublin')
    bo_sx_nd, bs_sx_nd = sarimax_gs(train_sx['Non_Dublin'],train_exog, 'Non-Dublin')

    sx_d_fit  = SARIMAX(train_sx['Dublin'],    exog=train_exog, order=bo_sx_d,  seasonal_order=bs_sx_d,
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
    sx_nd_fit = SARIMAX(train_sx['Non_Dublin'],exog=train_exog, order=bo_sx_nd, seasonal_order=bs_sx_nd,
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)

    pred_d_sx_obj  = sx_d_fit.get_forecast(steps=len(test_sx), exog=test_exog)
    pred_d_sx      = pred_d_sx_obj.predicted_mean;   pred_d_sx.index   = test_sx.index
    pred_d_sx_ci   = pred_d_sx_obj.conf_int(alpha=0.05); pred_d_sx_ci.index = test_sx.index
    pred_nd_sx_obj = sx_nd_fit.get_forecast(steps=len(test_sx), exog=test_exog)
    pred_nd_sx     = pred_nd_sx_obj.predicted_mean;  pred_nd_sx.index  = test_sx.index
    pred_nd_sx_ci  = pred_nd_sx_obj.conf_int(alpha=0.05); pred_nd_sx_ci.index = test_sx.index

    mae_d_sx   = mean_absolute_error(test_sx['Dublin'],    pred_d_sx)
    rmse_d_sx  = np.sqrt(mean_squared_error(test_sx['Dublin'],    pred_d_sx))
    mape_d_sx  = np.mean(np.abs((test_sx['Dublin']    - pred_d_sx)  / test_sx['Dublin']))    * 100
    mae_nd_sx  = mean_absolute_error(test_sx['Non_Dublin'], pred_nd_sx)
    rmse_nd_sx = np.sqrt(mean_squared_error(test_sx['Non_Dublin'], pred_nd_sx))
    mape_nd_sx = np.mean(np.abs((test_sx['Non_Dublin'] - pred_nd_sx) / test_sx['Non_Dublin'])) * 100

    # SARIMAX future
    full_exog_df = pd.DataFrame(scaler.transform(df_sx[exog_cols]), index=df_sx.index, columns=exog_cols)
    sx_d_full  = SARIMAX(df_sx['Dublin'],    exog=full_exog_df, order=bo_sx_d,  seasonal_order=bs_sx_d,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
    sx_nd_full = SARIMAX(df_sx['Non_Dublin'],exog=full_exog_df, order=bo_sx_nd, seasonal_order=bs_sx_nd,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)

    future_exog_scaled = scaler.transform(np.array([[df_sx['Unemployment_Rate'].iloc[-1], df_sx['RPPI'].iloc[-1]]] * n_forecast))
    future_dates_sx = pd.date_range(start=df_sx.index[-1] + pd.DateOffset(months=3), periods=n_forecast, freq='QS')
    fc_sx_d  = sx_d_full.get_forecast(steps=n_forecast, exog=future_exog_scaled)
    fc_sx_nd = sx_nd_full.get_forecast(steps=n_forecast, exog=future_exog_scaled)
    sx_d_fc_mean = fc_sx_d.predicted_mean;  sx_d_fc_mean.index  = future_dates_sx
    sx_nd_fc_mean= fc_sx_nd.predicted_mean; sx_nd_fc_mean.index = future_dates_sx
    sx_d_fc_ci   = fc_sx_d.conf_int(alpha=0.05);  sx_d_fc_ci.index  = future_dates_sx
    sx_nd_fc_ci  = fc_sx_nd.conf_int(alpha=0.05); sx_nd_fc_ci.index = future_dates_sx

    # ── Model comparison table ────────────────────────────────────────────────
    comp_df = pd.DataFrame([
        {'Model': 'SARIMA',            'Dublin_MAE': mae_d_s,   'Dublin_RMSE': rmse_d_s,   'Dublin_MAPE': mape_d_s,
         'NonDublin_MAE': mae_nd_s,   'NonDublin_RMSE': rmse_nd_s,   'NonDublin_MAPE': mape_nd_s},
        {'Model': 'Random Forest',     'Dublin_MAE': mae_d_rf,  'Dublin_RMSE': rmse_d_rf,  'Dublin_MAPE': mape_d_rf,
         'NonDublin_MAE': mae_nd_rf,  'NonDublin_RMSE': rmse_nd_rf,  'NonDublin_MAPE': mape_nd_rf},
        {'Model': 'Gradient Boosting', 'Dublin_MAE': mae_d_gb,  'Dublin_RMSE': rmse_d_gb,  'Dublin_MAPE': mape_d_gb,
         'NonDublin_MAE': mae_nd_gb,  'NonDublin_RMSE': rmse_nd_gb,  'NonDublin_MAPE': mape_nd_gb},
        {'Model': 'Prophet',           'Dublin_MAE': mae_d_p,   'Dublin_RMSE': rmse_d_p,   'Dublin_MAPE': mape_d_p,
         'NonDublin_MAE': mae_nd_p,   'NonDublin_RMSE': rmse_nd_p,   'NonDublin_MAPE': mape_nd_p},
        {'Model': 'SARIMAX',           'Dublin_MAE': mae_d_sx,  'Dublin_RMSE': rmse_d_sx,  'Dublin_MAPE': mape_d_sx,
         'NonDublin_MAE': mae_nd_sx,  'NonDublin_RMSE': rmse_nd_sx,  'NonDublin_MAPE': mape_nd_sx},
    ])
    comp_df['Avg_MAPE'] = (comp_df['Dublin_MAPE'] + comp_df['NonDublin_MAPE']) / 2
    comp_df['Avg_MAE']  = (comp_df['Dublin_MAE']  + comp_df['NonDublin_MAE'])  / 2
    comp_df = comp_df.sort_values('Avg_MAPE').reset_index(drop=True)

    return {
        'df_ts': df_ts, 'train_s': train_s, 'test_s': test_s,
        'pred_d_s': pred_d_s, 'ci_d_s': ci_d_s, 'mape_d_s': mape_d_s, 'mae_d_s': mae_d_s,
        'pred_nd_s': pred_nd_s,'ci_nd_s': ci_nd_s,'mape_nd_s':mape_nd_s,'mae_nd_s':mae_nd_s,
        'fc_d_mean': fc_d_mean, 'fc_d_ci': fc_d_ci, 'fc_nd_mean': fc_nd_mean, 'fc_nd_ci': fc_nd_ci,
        'df_feat_d': df_feat_d, 'df_feat_nd': df_feat_nd,
        'feat_cols_d': feat_cols_d, 'feat_cols_nd': feat_cols_nd,
        'test_size_rf': test_size_rf,
        'y_train_d': y_train_d, 'y_test_d': y_test_d, 'pred_d_rf': pred_d_rf, 'dates_test_d': dates_test_d,
        'mape_d_rf': mape_d_rf, 'mae_d_rf': mae_d_rf,
        'y_train_nd': y_train_nd,'y_test_nd':y_test_nd,'pred_nd_rf':pred_nd_rf,'dates_test_nd':dates_test_nd,
        'mape_nd_rf': mape_nd_rf,'mae_nd_rf':mae_nd_rf,
        'pred_d_gb': pred_d_gb, 'mape_d_gb': mape_d_gb, 'mae_d_gb': mae_d_gb,
        'pred_nd_gb':pred_nd_gb,'mape_nd_gb':mape_nd_gb,'mae_nd_gb':mae_nd_gb,
        'rf_d': rf_d, 'rf_nd': rf_nd,
        'df_p_d': df_p_d, 'df_p_nd': df_p_nd,
        'train_p_d': train_p_d,'test_p_d': test_p_d,'fc_test_d': fc_test_d,'mape_d_p':mape_d_p,'mae_d_p':mae_d_p,
        'train_p_nd':train_p_nd,'test_p_nd':test_p_nd,'fc_test_nd':fc_test_nd,'mape_nd_p':mape_nd_p,'mae_nd_p':mae_nd_p,
        'p_future_d': p_future_d,'p_future_nd':p_future_nd,
        'train_sx': train_sx,'test_sx': test_sx,
        'pred_d_sx': pred_d_sx,'pred_d_sx_ci': pred_d_sx_ci,'mape_d_sx':mape_d_sx,'mae_d_sx':mae_d_sx,
        'pred_nd_sx':pred_nd_sx,'pred_nd_sx_ci':pred_nd_sx_ci,'mape_nd_sx':mape_nd_sx,'mae_nd_sx':mae_nd_sx,
        'sx_d_fc_mean': sx_d_fc_mean,'sx_d_fc_ci': sx_d_fc_ci,
        'sx_nd_fc_mean':sx_nd_fc_mean,'sx_nd_fc_ci':sx_nd_fc_ci,
        'df_sx': df_sx,
        'comp_df': comp_df,
        'mape_d_sx': mape_d_sx, 'mape_d_p': mape_d_p,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🏠 Irish Rental Market Analysis Dashboard")
st.markdown("**MS5131 Major Project** — Predictive Analytics for Irish Rental Prices (2007–2027)")

# Load data
with st.spinner("Loading datasets..."):
    master, county_snapshot, landlord, hc, income, unemp_q, ppi_q, rtb = load_all_data()

latest = master.iloc[-1]
first  = master.iloc[0]

# ── KPI row ───────────────────────────────────────────────────────────────────
st.subheader("📊 Key Metrics — Q3 2025")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Dublin Rent",      f"€{latest['Dublin']:,.0f}/mo",     f"+{((latest['Dublin']/first['Dublin'])-1)*100:.0f}% since 2007")
k2.metric("Non-Dublin Rent",  f"€{latest['Non_Dublin']:,.0f}/mo", f"+{((latest['Non_Dublin']/first['Non_Dublin'])-1)*100:.0f}% since 2007")
k3.metric("Dublin Affordability", f"{latest['Affordability_Dublin']:.1f}%", "Threshold: 30%")
k4.metric("Unemployment Rate",f"{master['Unemployment_Rate'].dropna().iloc[-1]:.1f}%")
k5.metric("Counties Analysed", f"{len(county_snapshot)}")

st.divider()

# ── Sidebar navigation ────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    ["📈 EDA — Rent Trends",
     "🗺️ County Snapshot",
     "💸 Affordability",
     "📉 Macroeconomic Factors",
     "🏗️ Supply & Landlords",
     "🤖 SARIMA Model",
     "🌲 Random Forest & GB",
     "🔮 Prophet",
     "📡 SARIMAX",
     "🏆 Model Comparison",
     "📋 Policy Conclusions"],
)

# ════════════════════════════════════════════════════════════════════
# PAGE 1 — EDA Rent Trends
# ════════════════════════════════════════════════════════════════════
if page == "📈 EDA — Rent Trends":
    st.header("Irish Rental Market: Rent by Region (2007–2025)")

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(master['Date'], master['Dublin'],          color='#1F3864', linewidth=2.5, label='Dublin')
    ax.plot(master['Date'], master['GDA_excl_Dublin'], color='#2E5FA3', linewidth=2,   label='GDA (excl. Dublin)')
    ax.plot(master['Date'], master['Non_Dublin'],      color='#1A7A4A', linewidth=2,   label='Non-Dublin')
    ax.plot(master['Date'], master['Outside_GDA'],     color='#BA7517', linewidth=2,   label='Outside GDA')
    for xdate, label, color, ypos_frac in [
        ('2008-07-01',' GFC\n 2008','red',0.95),
        ('2016-10-01',' RPZ\n Intro','purple',0.85),
        ('2020-03-01',' COVID\n 2020','orange',0.75),
    ]:
        ax.axvline(pd.Timestamp(xdate), color=color, linestyle='--', alpha=0.5, linewidth=1.2)
        ax.text(pd.Timestamp(xdate), master['Dublin'].max()*ypos_frac, label, fontsize=9, color=color)
    ax.set_title('Standardised Average Rent by Region (2007–2025)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Average Monthly Rent (€)')
    ax.legend(fontsize=11); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'€{int(x):,}'))
    ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.subheader("Year-on-Year Growth Rate — Did RPZ Work?")
    growth_data = master.dropna(subset=['Dublin_YoY'])
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(growth_data['Date'], growth_data['Dublin_YoY'],    color='#1F3864', linewidth=2.5, label='Dublin YoY %')
    ax.plot(growth_data['Date'], growth_data['NonDublin_YoY'], color='#1A7A4A', linewidth=2, linestyle='--', label='Non-Dublin YoY %')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(4, color='red',    linestyle=':', linewidth=1.5, label='RPZ 4% Cap')
    ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, label='RPZ 2% Cap (2021)')
    ax.axvline(pd.Timestamp('2016-10-01'), color='purple', linestyle='--', linewidth=1.5)
    ax.axvspan(pd.Timestamp('2016-10-01'), master['Date'].max(), alpha=0.05, color='purple')
    ax.text(pd.Timestamp('2017-01-01'), growth_data['Dublin_YoY'].max()*0.9, 'RPZ Period', fontsize=10, color='purple')
    ax.set_title('Annual Rent Growth Rate', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Year-on-Year Growth (%)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'{x:.1f}%'))
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════
# PAGE 2 — County Snapshot
# ════════════════════════════════════════════════════════════════════
elif page == "🗺️ County Snapshot":
    st.header("Average Monthly Rent by County — Q3 2025")
    fig, ax = plt.subplots(figsize=(14, 9))
    bar_colors = ['#1F3864' if r > 1700 else '#2E5FA3' if r > 1300 else '#1A7A4A'
                  for r in county_snapshot['Rent_Q32025']]
    bars = ax.barh(county_snapshot['County'], county_snapshot['Rent_Q32025'], color=bar_colors, edgecolor='white')
    nat_avg = county_snapshot['Rent_Q32025'].mean()
    ax.axvline(nat_avg, color='red', linestyle='--', linewidth=1.5, label=f'National Avg: €{nat_avg:,.0f}')
    for bar, val in zip(bars, county_snapshot['Rent_Q32025']):
        ax.text(val+20, bar.get_y()+bar.get_height()/2, f'€{val:,.0f}', va='center', fontsize=9)
    ax.set_title('Average Monthly Rent by County — Q3 2025', fontsize=14, fontweight='bold')
    ax.set_xlabel('Standardised Average Rent (€)'); ax.legend(fontsize=11); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.dataframe(county_snapshot.style.format({'Rent_Q32025': '€{:,.0f}'}), use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# PAGE 3 — Affordability
# ════════════════════════════════════════════════════════════════════
elif page == "💸 Affordability":
    st.header("Rent-to-Income Ratio Over Time")
    afford_data = master.dropna(subset=['Affordability_Dublin'])
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(afford_data['Date'], afford_data['Affordability_Dublin'],   color='#1F3864', linewidth=2.5, label='Dublin Affordability Ratio')
    ax.plot(afford_data['Date'], afford_data['Affordability_National'], color='#1A7A4A', linewidth=2, linestyle='--', label='Non-Dublin Affordability Ratio')
    ax.axhline(30, color='red', linestyle='-', linewidth=2, label='30% Affordability Threshold')
    ax.fill_between(afford_data['Date'], 30, afford_data['Affordability_Dublin'],
                    where=afford_data['Affordability_Dublin'] > 30, alpha=0.15, color='red', label='Above Threshold')
    ax.axvline(pd.Timestamp('2016-10-01'), color='purple', linestyle='--', alpha=0.5)
    ax.text(pd.Timestamp('2016-10-01'), afford_data['Affordability_Dublin'].max()*0.97, ' RPZ Intro', fontsize=9, color='purple')
    ax.set_title('Is Dublin Already in a Rental Crisis?', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Rent as % of Monthly Income')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'{x:.0f}%'))
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    st.pyplot(fig); plt.close()

    dublin_afford    = round(latest['Dublin']     / master['Monthly_Income'].iloc[-1] * 100, 1)
    nondublin_afford = round(latest['Non_Dublin'] / master['Monthly_Income'].iloc[-1] * 100, 1)
    c1, c2 = st.columns(2)
    c1.metric("Dublin Rent-to-Income",     f"{dublin_afford}%",     delta="CRISIS" if dublin_afford > 30 else "Below threshold", delta_color="inverse")
    c2.metric("Non-Dublin Rent-to-Income", f"{nondublin_afford}%",  delta="CRISIS" if nondublin_afford > 30 else "Below threshold", delta_color="inverse")

# ════════════════════════════════════════════════════════════════════
# PAGE 4 — Macroeconomic Factors
# ════════════════════════════════════════════════════════════════════
elif page == "📉 Macroeconomic Factors":
    st.header("Correlation Heatmap — Rent & Macroeconomic Variables")
    corr_cols = ['Dublin','Non_Dublin','GDA_excl_Dublin','Outside_GDA','Unemployment_Rate','RPPI']
    corr_matrix = master[corr_cols].dropna().corr().round(2)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', mask=mask, ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={'size': 12})
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold'); plt.tight_layout()
    st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════
# PAGE 5 — Supply & Landlords
# ════════════════════════════════════════════════════════════════════
elif page == "🏗️ Supply & Landlords":
    st.header("Housing Supply vs Dublin Rent")
    dublin_annual = master.groupby('Year')['Dublin'].mean().reset_index()
    dublin_annual.columns = ['Year','Dublin_Avg_Rent']
    supply_demand = pd.merge(hc, dublin_annual, on='Year', how='inner')
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()
    ax1.bar(supply_demand['Year'], supply_demand['Completions'], color='#2E5FA3', alpha=0.6, label='Housing Completions')
    ax2.plot(supply_demand['Year'], supply_demand['Dublin_Avg_Rent'], color='#C0392B', linewidth=2.5, marker='o', markersize=5, label='Dublin Avg Rent')
    ax1.axhline(48000, color='green',  linestyle='--', linewidth=1.5, label='48k Target')
    ax1.axhline(33000, color='orange', linestyle='--', linewidth=1.5, label='33k Target')
    ax1.set_title('Housing Supply vs Dublin Rent', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year'); ax1.set_ylabel('Housing Completions', color='#2E5FA3')
    ax2.set_ylabel('Dublin Avg Monthly Rent (€)', color='#C0392B')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'€{int(x):,}'))
    lines1,labels1=ax1.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2,labels1+labels2,fontsize=10,loc='upper left')
    ax1.grid(axis='y',alpha=0.3); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.header("Landlord Numbers")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.bar(landlord['Date'], landlord['Total_Landlords'], color='#2E5FA3', width=60)
    ax1.set_title('Total Number of Private Landlords', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Landlords')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'{int(x):,}')); ax1.grid(axis='y', alpha=0.3)
    qoq_colors = ['#C0392B' if x < 0 else '#1A7A4A' for x in landlord['QoQ_Change'].fillna(0)]
    ax2.bar(landlord['Date'], landlord['QoQ_Change'], color=qoq_colors, width=60)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('Quarter-on-Quarter Change in Landlord Numbers', fontsize=13, fontweight='bold')
    ax2.set_ylabel('QoQ Change'); ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════
# MODEL PAGES — run models on demand
# ════════════════════════════════════════════════════════════════════
elif page in ["🤖 SARIMA Model","🌲 Random Forest & GB","🔮 Prophet","📡 SARIMAX","🏆 Model Comparison"]:

    with st.spinner("⏳ Running all forecasting models (this takes a few minutes on first load)..."):
        m = run_models(master, unemp_q, ppi_q)

    # ── SARIMA ───────────────────────────────────────────────────────────────
    if page == "🤖 SARIMA Model":
        st.header("SARIMA Baseline Model")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        for ax, tr_col, pred, ci, name, mae, mape in [
            (ax1,'Dublin',    m['pred_d_s'], m['ci_d_s'],  'Dublin',     m['mae_d_s'],  m['mape_d_s']),
            (ax2,'Non_Dublin',m['pred_nd_s'],m['ci_nd_s'], 'Non-Dublin', m['mae_nd_s'], m['mape_nd_s']),
        ]:
            ax.plot(m['train_s'].index, m['train_s'][tr_col], color='#1F3864', linewidth=2, label='Historical (Train)')
            ax.plot(m['test_s'].index,  m['test_s'][tr_col],  color='#C0392B', linewidth=2, label='Actual (Test)')
            ax.plot(pred.index, pred.values, color='#1A7A4A', linewidth=2, linestyle='--', label=f'SARIMA (MAPE={mape:.1f}%)')
            ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2, color='#1A7A4A', label='95% CI')
            ax.axvline(m['test_s'].index[0], color='gray', linestyle=':', linewidth=1.5)
            ax.set_title(f'SARIMA — {name}: Actual vs Predicted  |  MAE=€{mae:.0f}/mo, MAPE={mape:.1f}%', fontsize=13, fontweight='bold')
            ax.set_ylabel('Monthly Rent (€)'); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'€{int(x):,}'))
            ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("SARIMA Future Forecast (Q4 2025 — Q3 2027)")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        cutoff = pd.Timestamp('2022-01-01')
        recent = m['df_ts'][m['df_ts'].index >= cutoff]
        for ax, col, fc_mean, fc_ci, name in [
            (ax1,'Dublin',    m['fc_d_mean'], m['fc_d_ci'], 'Dublin'),
            (ax2,'Non_Dublin',m['fc_nd_mean'],m['fc_nd_ci'],'Non-Dublin'),
        ]:
            ax.plot(recent.index, recent[col], color='#1F3864', linewidth=2.5, marker='o', markersize=4, label='Historical')
            ax.plot(fc_mean.index, fc_mean.values, color='#1A7A4A', linewidth=2.5, linestyle='--', marker='s', markersize=5, label='SARIMA Forecast')
            ax.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.2, color='#1A7A4A', label='95% CI')
            ax.axvline(m['df_ts'].index[-1], color='gray', linestyle=':', linewidth=1.5)
            ax.annotate(f'€{fc_mean.iloc[-1]:,.0f}', xy=(fc_mean.index[-1], fc_mean.iloc[-1]), xytext=(0,10), textcoords='offset points', fontsize=10, fontweight='bold', color='#1A7A4A')
            ax.set_title(f'SARIMA — {name} Forecast', fontsize=13, fontweight='bold')
            ax.set_ylabel('Monthly Rent (€)'); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'€{int(x):,}'))
            ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Random Forest & GB ───────────────────────────────────────────────────
    elif page == "🌲 Random Forest & GB":
        st.header("Random Forest — Actual vs Predicted")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        for ax, df_feat, y_train, y_test, pred, dates_test, name, mae, mape in [
            (ax1, m['df_feat_d'],  m['y_train_d'],  m['y_test_d'],  m['pred_d_rf'],  m['dates_test_d'],  'Dublin',     m['mae_d_rf'],  m['mape_d_rf']),
            (ax2, m['df_feat_nd'], m['y_train_nd'], m['y_test_nd'], m['pred_nd_rf'], m['dates_test_nd'], 'Non-Dublin', m['mae_nd_rf'], m['mape_nd_rf']),
        ]:
            train_dates = df_feat['Date'].values[:-m['test_size_rf']]
            ax.plot(train_dates, y_train, color='#1F3864', linewidth=2, label='Historical (Train)')
            ax.plot(dates_test,  y_test,  color='#C0392B', linewidth=2, label='Actual (Test)')
            ax.plot(dates_test,  pred,    color='#117A65', linewidth=2, linestyle='--', label=f'RF Forecast (MAPE={mape:.1f}%)')
            ax.axvline(dates_test[0], color='gray', linestyle=':', linewidth=1.5)
            ax.set_title(f'Random Forest — {name}  |  MAE=€{mae:.0f}/mo, MAPE={mape:.1f}%', fontsize=13, fontweight='bold')
            ax.set_ylabel('Monthly Rent (€)'); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'€{int(x):,}'))
            ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.header("Feature Importance")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        for ax, model, feat_cols, name in [
            (ax1, m['rf_d'],  m['feat_cols_d'],  'Dublin'),
            (ax2, m['rf_nd'], m['feat_cols_nd'], 'Non-Dublin'),
        ]:
            n_feat = model.n_features_in_
            fc_use = feat_cols[:n_feat]
            imp = pd.DataFrame({'Feature': fc_use, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True).tail(12)
            bar_colors = ['#C0392B' if 'RPPI' in f or 'Unemp' in f else '#2E5FA3' if 'lag' in f or 'roll' in f else '#1A7A4A' for f in imp['Feature']]
            ax.barh(imp['Feature'], imp['Importance'], color=bar_colors, edgecolor='white')
            ax.set_title(f'Feature Importance — {name}\n(Red=Macro, Blue=Lag, Green=Other)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Importance Score'); ax.grid(axis='x', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.header("Gradient Boosting")
        c1, c2 = st.columns(2)
        c1.metric("Dublin MAE",      f"€{m['mae_d_gb']:.0f}")
        c1.metric("Dublin MAPE",     f"{m['mape_d_gb']:.2f}%")
        c2.metric("Non-Dublin MAE",  f"€{m['mae_nd_gb']:.0f}")
        c2.metric("Non-Dublin MAPE", f"{m['mape_nd_gb']:.2f}%")

    # ── Prophet ──────────────────────────────────────────────────────────────
    elif page == "🔮 Prophet":
        st.header("Prophet — Actual vs Predicted")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        for ax, train_df, test_df, forecast, name, mae, mape in [
            (ax1, m['train_p_d'],  m['test_p_d'],  m['fc_test_d'],  'Dublin',     m['mae_d_p'],  m['mape_d_p']),
            (ax2, m['train_p_nd'], m['test_p_nd'], m['fc_test_nd'], 'Non-Dublin', m['mae_nd_p'], m['mape_nd_p']),
        ]:
            ax.plot(train_df['ds'], train_df['y'],       color='#1F3864', linewidth=2, label='Historical (Train)')
            ax.plot(test_df['ds'],  test_df['y'],         color='#C0392B', linewidth=2, label='Actual (Test)')
            ax.plot(forecast['ds'], forecast['yhat'],     color='#1A7A4A', linewidth=2, linestyle='--', label=f'Prophet (MAPE={mape:.1f}%)')
            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, color='#1A7A4A', label='95% CI')
            ax.set_title(f'Prophet — {name}  |  MAE=€{mae:.0f}/mo, MAPE={mape:.1f}%', fontsize=12, fontweight='bold')
            ax.set_ylabel('Monthly Rent (€)'); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'€{int(x):,}'))
            ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.header("Prophet Future Forecast")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        cutoff_p = pd.Timestamp('2022-01-01')
        for ax, df_p, future_pred, name in [
            (ax1, m['df_p_d'],  m['p_future_d'],  'Dublin'),
            (ax2, m['df_p_nd'], m['p_future_nd'], 'Non-Dublin'),
        ]:
            recent_p = df_p[df_p['ds'] >= cutoff_p]
            ax.plot(recent_p['ds'], recent_p['y'], color='#1F3864', linewidth=2.5, marker='o', markersize=4, label='Historical')
            ax.plot(future_pred['ds'], future_pred['yhat'], color='#1A7A4A', linewidth=2.5, linestyle='--', marker='s', markersize=5, label='Prophet Forecast')
            ax.fill_between(future_pred['ds'], future_pred['yhat_lower'], future_pred['yhat_upper'], alpha=0.25, color='#1A7A4A', label='95% CI')
            ax.annotate(f'€{future_pred["yhat"].iloc[-1]:,.0f}', xy=(future_pred['ds'].iloc[-1], future_pred['yhat'].iloc[-1]), xytext=(10,10), textcoords='offset points', fontsize=10, fontweight='bold', color='#1A7A4A')
            ax.set_title(f'Prophet — {name} Forecast', fontsize=12, fontweight='bold')
            ax.set_ylabel('Monthly Rent (€)'); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'€{int(x):,}'))
            ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── SARIMAX ──────────────────────────────────────────────────────────────
    elif page == "📡 SARIMAX":
        st.header("SARIMAX — Exogenous Variables Model")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        for ax, col, pred, ci, name, mae, mape in [
            (ax1,'Dublin',    m['pred_d_sx'],  m['pred_d_sx_ci'],  'Dublin',     m['mae_d_sx'],  m['mape_d_sx']),
            (ax2,'Non_Dublin',m['pred_nd_sx'], m['pred_nd_sx_ci'], 'Non-Dublin', m['mae_nd_sx'], m['mape_nd_sx']),
        ]:
            ax.plot(m['train_sx'].index, m['train_sx'][col], color='#1F3864', linewidth=2, label='Historical (Train)')
            ax.plot(m['test_sx'].index,  m['test_sx'][col],  color='#C0392B', linewidth=2, label='Actual (Test)')
            ax.plot(pred.index, pred.values, color='#1A7A4A', linewidth=2, linestyle='--', label=f'SARIMAX (MAPE={mape:.1f}%)')
            ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2, color='#1A7A4A', label='95% CI')
            ax.set_title(f'SARIMAX — {name}  |  MAE=€{mae:.0f}/mo, MAPE={mape:.1f}%', fontsize=12, fontweight='bold')
            ax.set_ylabel('Monthly Rent (€)'); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'€{int(x):,}'))
            ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.header("SARIMAX Future Forecast")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        cutoff_sx = pd.Timestamp('2022-01-01')
        recent_sx = m['df_sx'][m['df_sx'].index >= cutoff_sx]
        for ax, col, fc_mean, fc_ci, name in [
            (ax1,'Dublin',    m['sx_d_fc_mean'],  m['sx_d_fc_ci'],  'Dublin'),
            (ax2,'Non_Dublin',m['sx_nd_fc_mean'], m['sx_nd_fc_ci'], 'Non-Dublin'),
        ]:
            ax.plot(recent_sx.index, recent_sx[col], color='#1F3864', linewidth=2.5, marker='o', markersize=4, label='Historical')
            ax.plot(fc_mean.index, fc_mean.values, color='#1A7A4A', linewidth=2.5, linestyle='--', marker='s', markersize=5, label='SARIMAX Forecast')
            ax.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.25, color='#1A7A4A', label='95% CI')
            ax.annotate(f'€{fc_mean.iloc[-1]:,.0f}', xy=(fc_mean.index[-1], fc_mean.iloc[-1]), xytext=(10,10), textcoords='offset points', fontsize=10, fontweight='bold', color='#1A7A4A')
            ax.set_title(f'SARIMAX — {name} Forecast', fontsize=12, fontweight='bold')
            ax.set_ylabel('Monthly Rent (€)'); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'€{int(x):,}'))
            ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Model Comparison ─────────────────────────────────────────────────────
    elif page == "🏆 Model Comparison":
        st.header("Final Model Comparison")
        comp = m['comp_df'].copy()
        comp_display = comp[['Model','Dublin_MAE','Dublin_RMSE','Dublin_MAPE','NonDublin_MAE','NonDublin_RMSE','NonDublin_MAPE','Avg_MAPE','Avg_MAE']].copy()
        for c in ['Dublin_MAE','Dublin_RMSE','NonDublin_MAE','NonDublin_MAE','NonDublin_RMSE','Avg_MAE']:
            comp_display[c] = comp_display[c].apply(lambda x: f'€{x:.0f}')
        for c in ['Dublin_MAPE','NonDublin_MAPE','Avg_MAPE']:
            comp_display[c] = comp_display[c].apply(lambda x: f'{x:.2f}%')
        st.dataframe(comp_display, use_container_width=True)

        st.success(f"🥇 Best Model: **{comp.iloc[0]['Model']}** — Avg MAPE: {comp.iloc[0]['Avg_MAPE']:.2f}%")

        model_colors = ['#1A7A4A','#2E5FA3','#C0392B','#E67E22','#95A5A6']
        models_list  = comp['Model'].tolist()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        for ax, col, title in [(ax1,'Dublin_MAPE','Dublin'),(ax2,'NonDublin_MAPE','Non-Dublin')]:
            vals = comp[col].tolist()
            bars = ax.barh(models_list, vals, color=model_colors[:len(vals)], edgecolor='white', height=0.6)
            for bar, val in zip(bars, vals):
                ax.text(val+0.1, bar.get_y()+bar.get_height()/2, f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')
            min_idx = vals.index(min(vals)); bars[min_idx].set_edgecolor('#FFD700'); bars[min_idx].set_linewidth(3)
            ax.text(vals[min_idx]/2, min_idx, ' BEST', va='center', fontsize=10, color='white', fontweight='bold')
            ax.set_title(f'{title} — MAPE (Lower is Better)', fontsize=13, fontweight='bold')
            ax.set_xlabel('MAPE (%)'); ax.axvline(5, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(axis='x', alpha=0.3); ax.set_xlim(0, max(vals)*1.2)
        plt.suptitle('Model Comparison — MAPE', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════
# PAGE — Policy Conclusions
# ════════════════════════════════════════════════════════════════════
elif page == "📋 Policy Conclusions":
    st.header("Research Findings & Policy Recommendations")

    with st.expander("SQ1: Which Economic Factors Drive Irish Rental Prices Most?", expanded=True):
        st.markdown("""
**Dublin:**
- Rent momentum (lagged values) is the **primary driver** (~18% importance)
- Yearly seasonal pattern is significant
- Unemployment (lagged) has macro impact (~11%)
- Property price index significant in SARIMAX

**Non-Dublin:**
- Previous quarter rent dominates (~37%)
- Unemployment is a **stronger** driver outside Dublin (~14%)
- RPZ dummy — Rent Pressure Zones had structural impact

**Conclusion:** Rent momentum is the PRIMARY driver in both regions. Unemployment is significant, especially outside Dublin.
        """)

    with st.expander("SQ2: Is Ireland Heading Towards a Systemic Rental Crisis?", expanded=True):
        st.markdown(f"""
**Current Situation (Q3 2025):**
- Dublin: **€{latest['Dublin']:,.0f}/month** (up {((latest['Dublin']/first['Dublin'])-1)*100:.0f}% since 2007)
- Non-Dublin: **€{latest['Non_Dublin']:,.0f}/month** (up {((latest['Non_Dublin']/first['Non_Dublin'])-1)*100:.0f}% since 2007)

**Crisis Indicators:**
- ❌ Rents have grown 100%+ since 2007 — far exceeding wage growth
- ❌ All 5 models forecast continued upward trajectory to 2027
- ❌ RPZ (2016) slowed but did NOT reverse the trend
- ❌ Non-Dublin rents accelerating faster — crisis is spreading

**Conclusion: YES — Ireland's rental market shows systemic crisis characteristics.**
        """)

    with st.expander("SQ3: Can Predictive Analytics Give Policymakers Enough Time?", expanded=True):
        st.markdown("""
**Forecast Window:** 8 quarters (2 years) ahead — Q4 2025 to Q3 2027

**Model Accuracy (Best Models):** < 5% MAPE → 6–8 quarter warning window for policymakers

**Conclusion: YES — models forecast 2 years ahead with <5% error. Housing policy requires 12–18 months to implement, so this is sufficient lead time.**
        """)

    st.subheader("🏛️ Policy Recommendations")
    cols = st.columns(2)
    with cols[0]:
        st.info("**1. Supply-Side Intervention (Most Critical)**\nAccelerate social & affordable housing delivery. Fast-track planning in high-demand Dublin corridors.")
        st.info("**2. Strengthen RPZ Legislation**\nLower RPZ cap from 2% to inflation-linked rate. Expand RPZ coverage to Non-Dublin commuter belts.")
        st.info("**3. Unemployment–Rent Linkage**\nIncentivise remote-work hubs to distribute demand geographically.")
    with cols[1]:
        st.success("**4. Quarterly Forecasting Dashboard**\nDeploy SARIMAX/Prophet as a live policy tool. Auto-alert when forecast exceeds 30% affordability threshold.")
        st.success("**5. Data Infrastructure**\nAdd mortgage rates, construction costs, migration data as exogenous variables.")

    st.divider()
    st.subheader("Final Conclusion — MS5131 Major Project")
    st.markdown("""
> **"Is Ireland's rental market heading towards a systemic crisis — and can predictive analytics give policymakers enough time to stop it?"**

### ✅ YES to both parts.

The best-performing model captures both time-series structure AND macroeconomic drivers.
**Immediate supply-side intervention is required to break rent inertia.**
Strengthen and expand RPZ legislation beyond Dublin.
    """)
