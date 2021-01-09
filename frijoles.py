import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import squarify
from matplotlib import pyplot as plt
from beancount.loader import load_file
from beancount.query.query import run_query
from beancount.query.numberify import numberify_results

st.set_page_config(layout="wide", page_title="Frijoles", page_icon="frijoles.png")


@st.cache
def load_beancount_file(file_name):
    entries, _, opts = load_file(file_name)
    currency = opts["operating_currency"][0]
    cols, rows = run_query(entries, opts,
                           "SELECT   account,   YEAR(date) AS year,\
                                     MONTH(date) as month,\
                                     SUM(convert(position, '{}', date)) AS amount\
                            WHERE    account ~ 'Expenses'\
                            OR       account ~ 'Income'\
                            GROUP BY account, year, month\
                            ORDER BY account, year, month".format(currency)
                           )
    cols, rows = numberify_results(cols, rows)
    df = pd.DataFrame(rows, columns=[k[0] for k in cols])
    df.rename(columns={"account": "Account", "year": "Year", "month": "Month",
                       "amount ({})".format(currency): "Amount ({})".format(currency)}, inplace=True)
    df = df.astype({"Account": str, "Year": int, "Month": int, "Amount ({})".format(currency): np.float})
    df = df[["Account", "Year", "Month", "Amount ({})".format(currency)]].fillna(0)
    df["YearMonth"] = df.apply(lambda x: "{}-{:0>2d}".format(x["Year"], x["Month"]), axis=1)
    df = df[["Account", "YearMonth", "Amount ({})".format(currency)]]
    return df


def time_interval_aggregation(df_orig, time_period):
    if time_period == "Month":
        df = df_orig.rename(columns={"YearMonth": "Month"})
        df = df.pivot_table(index="Account", columns=['Month']).fillna(0).reset_index()
    elif time_period == "Quarter":
        df = df_orig.copy()
        df.index = pd.to_datetime(df["YearMonth"], format='%Y-%m') + pd.offsets.MonthEnd(0)
        df.drop(columns="YearMonth", inplace=True)
        # Aggregate at quarter level
        df = df.groupby(["Account", pd.Grouper(freq='Q')]).sum()
        df.index.rename(names=["Account", "Quarter"], inplace=True)
        df = df.pivot_table(index="Account", columns=df.index.get_level_values(1).map(lambda s: s.strftime('%Y-%m'))
                            .astype('period[Q]').strftime('%F-Q%q')).fillna(0).reset_index()
    elif time_period == "Year":
        df = df_orig.copy()
        df.index = pd.to_datetime(df["YearMonth"], format='%Y-%m') + pd.offsets.MonthEnd(0)
        df.drop(columns="YearMonth", inplace=True)
        # Aggregate at year level
        df = df.groupby(["Account", pd.Grouper(freq='Y')]).sum()
        df.index.rename(names=["Account", "Year"], inplace=True)
        df = df.pivot_table(index="Account", columns=df.index.get_level_values(1).map(lambda s: s.strftime('%Y-%m'))
                            .astype('period[Y]').strftime('%F')).fillna(0).reset_index()
    n_levels = df["Account"].str.count(":").max() + 1
    cols = ["Account_L{}".format(k) for k in range(n_levels)]
    df[cols] = df["Account"].str.split(':', n_levels - 1, expand=True)
    df = df.fillna('').drop(columns="Account", level=0).set_index(cols)
    return df.copy(), n_levels


def multiperiod_table(df_orig):
    # Time interval aggregation level
    time_interval = st.sidebar.radio(
        "Time interval:", ("Month", "Quarter", "Year"), index=1)
    df, n_levels = time_interval_aggregation(df_orig, time_interval)

    agg_lvl = st.sidebar.selectbox(
        'Account aggregation level', list(range(n_levels)), index=int(n_levels/2), key="acnt_agg_lvl")
    gb = ["Account_L{}".format(k) for k in range(agg_lvl + 1)]

    dfn = df.copy()
    if st.sidebar.checkbox('Invert sign of "Income"', value=True):
        dfn.loc["Income", :] = -dfn.loc["Income", :].values

    st.subheader('Multiperiod report')
    if st.checkbox('Time is on X-axis', value=True):
        st.dataframe(dfn.groupby(gb).sum())
    else:
        st.dataframe(dfn.groupby(gb).sum().transpose())
    return


def income_expenses_over_time(df_orig):
    # Time interval aggregation level
    time_interval = st.sidebar.radio(
        "Time interval:", ("Month", "Quarter", "Year"), index=1)
    dfn, n_levels = time_interval_aggregation(df_orig, time_interval)
    if st.sidebar.checkbox('Invert sign of "Income"', value=True):
        dfn.loc["Income", :] = -dfn.loc["Income", :].values
    st.subheader('Income and Expenses over Time')
    plot_type = st.sidebar.selectbox('Plot type', ["pyplot", "altair"], key="plot_type")
    df_L0 = dfn.groupby(["Account_L0"]) \
        .sum() \
        .transpose() \
        .reset_index()

    df_L0.columns.name = "Account"
    if plot_type == "pyplot":
        fig = plt.figure(figsize=(15, 5))
        ax = plt.axes()
        df_L0.plot.bar(ax=ax, x=time_interval, y=["Income", "Expenses"],
                       xlabel=time_interval, ylabel=df_L0["level_0"][0], rot=90)
        ax.locator_params(axis="x", tight=True, nbins=40)
        st.pyplot(fig)
    elif plot_type == "altair":
        n_intervals = df_L0.shape[0]
        df_new = df_L0.drop(columns="level_0") \
            .set_index(time_interval) \
            .stack() \
            .reset_index() \
            .rename(columns={0: dfn.columns.levels[0][0]})
        custom_spacing = 2
        chart = alt.Chart(df_new).mark_bar().encode(
            column=alt.Column(time_interval, spacing=custom_spacing, header=alt.Header(title="Income and Expenses",
                                                                                       labelOrient='bottom',
                                                                                       labelAlign='right',
                                                                                       labelAngle=-90)),
            x=alt.X('Account:O', axis=alt.Axis(title=None, labels=False, ticks=False)),
            y=alt.Y('{}:Q'.format(dfn.columns.levels[0][0]), title=dfn.columns.levels[0][0], axis=alt.Axis(grid=False)),
            color=alt.Color('Account', scale=alt.Scale(range=['#EA98D2', '#659CCA'])),
            tooltip=[alt.Tooltip('Account:O', title='Account'),
                     alt.Tooltip('{}:Q'.format(dfn.columns.levels[0][0]), title=dfn.columns.levels[0][0]),
                     alt.Tooltip('{}:N'.format(time_interval), title=time_interval)]
        ).properties(width=(1000 - n_intervals * custom_spacing) / n_intervals)
        st.altair_chart(chart, use_container_width=False)
    return


def treemap_analysis(df):
    df, n_levels = time_interval_aggregation(df, "Year")
    account = st.sidebar.radio('Account:', ("Income", "Expenses"), index=1)

    df_L1 = df.groupby(["Account_L0", "Account_L1"]).sum()
    tot = df_L1.sum(axis=1).to_frame()

    # Treemap Plot of account
    data = tot.loc[account].sort_values(by=0, ascending=False)
    idx = [k[0] != 0 for k in data.values]
    values = data.values[idx]
    labels = data.index[idx]

    width = 1
    height = 0.5
    values_norm = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(values_norm, 0, 0, width, height)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(account, x=0.5, y=0.55, fontsize=16)
    axes = [fig.add_axes([rect['x'], rect['y'], rect['dx'], rect['dy'], ]) for rect in rects]

    # If there are many subcategories of an account, repeat the list of colors many times
    # to make the number of colors sufficient.
    for ax, txt, color in zip(axes, labels, plt.cm.Pastel1.colors*5):
        ax.text(0.5, 0.5, txt, horizontalalignment='center', verticalalignment='center')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_facecolor(color)
    st.pyplot(fig)
    return


def main():
    st.title('Frijoles: a web-interface for Beancount accounting system')
    st.sidebar.header('Multiperiod report')
    file_name = st.sidebar.file_uploader("Upload *.beancount file:")

    if file_name is not None:
        try:
            with st.spinner("Uploading your *.beancount file..."):
                print(file_name.name)
                df_orig = load_beancount_file(file_name.name)
        except Exception as e:
            st.error(
                f"Sorry, there was a problem processing your *.beancount file.\n {e}"
            )

        analysis_type = st.sidebar.radio("Analysis type:",
                                         ("Multiperiod table", "Income & Expenses over Time", "Treemap"))
        if analysis_type == "Multiperiod table":
            multiperiod_table(df_orig)
        elif analysis_type == "Income & Expenses over Time":
            income_expenses_over_time(df_orig)
        elif analysis_type == "Treemap":
            treemap_analysis(df_orig)
    return


if __name__ == "__main__":
    main()
