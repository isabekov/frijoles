import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
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


@st.cache
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
    return df, n_levels


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

        # Time interval aggregation level
        time_interval = st.sidebar.radio(
            "Time interval:", ("Month", "Quarter", "Year")
        )
        df, n_levels = time_interval_aggregation(df_orig, time_interval)

        agg_lvl = st.sidebar.selectbox(
            'Account aggregation level', list(range(n_levels)), key="acnt_agg_lvl")
        gb = ["Account_L{}".format(k) for k in range(agg_lvl + 1)]

        dfn = df.copy()
        if st.sidebar.checkbox('Invert sign of "Income"', value=True):
            dfn.loc["Income", :] = -dfn.loc["Income", :].values

        st.subheader('Multiperiod report')

        if st.checkbox('Time is on X-axis', value=True):
            st.dataframe(dfn.groupby(gb).sum())
        else:
            st.dataframe(dfn.groupby(gb).sum().transpose())

        st.subheader('Income and Expenses over Time')
        plot_type = st.selectbox('Plot type', ["pyplot", "altair"], key="plot_type")
        df_L0 = dfn.groupby(["Account_L0"])\
                   .sum()\
                   .transpose()\
                   .reset_index()
        df_L0.columns.name = "Account"
        if plot_type == "pyplot":
            fig = plt.figure(figsize=(10, 4))
            ax = plt.axes()
            df_L0.plot.bar(ax=ax, x=time_interval, y=["Income", "Expenses"],
                           xlabel=time_interval, ylabel=df_L0["level_0"][0], rot=90)
            st.pyplot(fig)
        elif plot_type == "altair":
            n_intervals = df_L0.shape[0]
            df_new = df_L0.drop(columns="level_0")\
                          .set_index(time_interval)\
                          .stack()\
                          .reset_index()\
                          .rename(columns={0: "Amount"})
            custom_spacing = 2
            chart = alt.Chart(df_new).mark_bar().encode(
                column=alt.Column(time_interval, spacing=custom_spacing, header=alt.Header(title="Income and Expenses",
                                  labelOrient='bottom', labelAlign='right', labelAngle=-90)),
                x=alt.X('Account:O', axis=alt.Axis(title=None, labels=False, ticks=False)),
                y=alt.Y('Amount:Q', title="Amount", axis=alt.Axis(grid=False)),
                color=alt.Color('Account', scale=alt.Scale(range=['#EA98D2', '#659CCA'])),
                tooltip=[alt.Tooltip('Account:O', title='Account'),
                         alt.Tooltip('Amount:Q', title="Amount"),
                         alt.Tooltip('{}:N'.format(time_interval), title=time_interval)]
                                ).properties(width=(700 - n_intervals*custom_spacing)/n_intervals)
            st.altair_chart(chart, use_container_width=False)
    return


if __name__ == "__main__":
    main()
