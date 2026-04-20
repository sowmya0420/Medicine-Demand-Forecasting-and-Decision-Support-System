import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def get_forecast_result(df):

    # STEP 1
    monthly_sales = df.groupby(
        ['product_id', 'product_name', 'sale_month']
    )['units_sold'].sum().reset_index()

    summary = monthly_sales.groupby(
        ['product_id', 'product_name']
    )['units_sold'].agg(
        min_month_sales='min',
        max_month_sales='max',
        last_month_sales='last',
        avg_sales='mean'
    ).reset_index()

    # STEP 2
    def classify(avg):
        if avg >= 80:
            return "Fast"
        elif avg >= 40:
            return "Moderate"
        else:
            return "Slow"

    summary['movement_class'] = summary['avg_sales'].apply(classify)

    # STEP 3
    def fuzzy_predict(row):
        last_sale = row['last_month_sales']
        movement = row['movement_class']

        if movement == "Fast":
            return round(last_sale * 1.15)
        elif movement == "Moderate":
            return round(last_sale * 1.05)
        else:
            return round(last_sale * 0.90)

    summary['predicted_sales'] = summary.apply(
        fuzzy_predict,
        axis=1
    )

    # STEP 4
    stock_df = df.groupby(
        ['product_id', 'product_name']
    )['units_in_stock'].last().reset_index()

    result = pd.merge(
        summary,
        stock_df,
        on=['product_id', 'product_name']
    )

    # STEP 5
    def stock_status(row):
        stock = row['units_in_stock']
        pred = row['predicted_sales']

        lower_limit = pred * 0.8
        upper_limit = pred * 1.5

        if stock == 0:
            return "Out of Stock"
        elif lower_limit <= stock <= upper_limit:
            return "Sufficient"
        elif stock < lower_limit:
            return "Understocked"
        else:
            return "Overstock"

    result['status'] = result.apply(stock_status, axis=1)

    return result
def run_obj1_forecasting(df):
    result = get_forecast_result(df)

    st.title("Module 2 - Real-Time Demand Forecasting")

    # =========================================
    # STEP 1: Monthly sales summary
    # =========================================
    monthly_sales = df.groupby(
        ['product_id', 'product_name', 'sale_month']
    )['units_sold'].sum().reset_index()

    summary = monthly_sales.groupby(
        ['product_id', 'product_name']
    )['units_sold'].agg(
        min_month_sales='min',
        max_month_sales='max',
        last_month_sales='last',
        avg_sales='mean'
    ).reset_index()

    # =========================================
    # STEP 2: Movement classification
    # =========================================
    def classify(avg):
        if avg >= 80:
            return "Fast"
        elif avg >= 40:
            return "Moderate"
        else:
            return "Slow"

    summary['movement_class'] = summary['avg_sales'].apply(classify)

    # =========================================
    # STEP 3: Soft Computing Forecasting
    # =========================================
    def fuzzy_predict(row):
        last_sale = row['last_month_sales']
        movement = row['movement_class']

        if movement == "Fast":
            return round(last_sale * 1.15)
        elif movement == "Moderate":
            return round(last_sale * 1.05)
        else:
            return round(last_sale * 0.90)

    summary['predicted_sales'] = summary.apply(
        fuzzy_predict,
        axis=1
    )

    # =========================================
    # STEP 4: Stock merge
    # =========================================
    stock_df = df.groupby(
        ['product_id', 'product_name']
    )['units_in_stock'].last().reset_index()

    result = pd.merge(
        summary,
        stock_df,
        on=['product_id', 'product_name']
    )

    # =========================================
    # STEP 5: Fuzzy stock status
    # =========================================
    def stock_status(row):
        stock = row['units_in_stock']
        pred = row['predicted_sales']

        lower_limit = pred * 0.8
        upper_limit = pred * 1.5

        if stock == 0:
            return "Out of Stock"
        elif lower_limit <= stock <= upper_limit:
            return "Sufficient"
        elif stock < lower_limit:
            return "Understocked"
        else:
            return "Overstock"

    result['status'] = result.apply(stock_status, axis=1)

    # =========================================
    # STATUS OVERVIEW
    # =========================================
    st.subheader("Current Status Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Understocked", (result['status'] == "Understocked").sum())
    col2.metric("Overstock", (result['status'] == "Overstock").sum())
    col3.metric("Sufficient", (result['status'] == "Sufficient").sum())
    col4.metric("Out of Stock", (result['status'] == "Out of Stock").sum())

    # =========================================
    # TAB STYLE
    # =========================================
    st.markdown("""
        <style>
        button[data-baseweb="tab"] {
            color: black !important;
            font-weight: bold;
            font-size: 16px;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: red !important;
            border-bottom: 3px solid red !important;
        }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(
        ["1. Demand Forecast", "2. Medicine Details"]
    )

    # =========================================
    # TAB 1
    # =========================================
    with tab1:
        st.subheader("Demand Forecast Dashboard")

        under_df = result[result['status'] == "Understocked"].head(4)
        over_df = result[result['status'] == "Overstock"].head(4)
        suff_df = result[result['status'] == "Sufficient"].head(4)
        out_df = result[result['status'] == "Out of Stock"].head(4)

        display_df = pd.concat(
            [under_df, over_df, suff_df, out_df],
            ignore_index=True
        )

        display_df.insert(
            0,
            "S.No",
            range(1, len(display_df) + 1)
        )

        display_df = display_df[
            [
                'S.No',
                'product_id',
                'product_name',
                'movement_class',
                'min_month_sales',
                'max_month_sales',
                'last_month_sales',
                'predicted_sales',
                'units_in_stock',
                'status'
            ]
        ]

        def color_status(val):
            if val == "Understocked":
                return "background-color: yellow"
            elif val == "Overstock":
                return "background-color: red"
            elif val == "Out of Stock":
                return "background-color: red"
            else:
                return "background-color: lightgreen"

        st.dataframe(
            display_df.style.map(
                color_status,
                subset=['status']
            ),
            hide_index=True
        )

        st.download_button(
            "Download Full Forecast Data",
            data=result.to_csv(index=False),
            file_name="forecast_report_all_records.csv",
            mime="text/csv"
        )

    # =========================================
    # TAB 2
    # =========================================
    with tab2:
        st.markdown(
            "<h2 style='color:green;'>Medicine Details Dashboard</h2>",
            unsafe_allow_html=True
        )

        medicine_names = df['product_name'].dropna().unique()
        product_ids = df['product_id'].dropna().unique()

        search_type = st.radio(
            "Search by",
            ["Medicine Name", "Product ID"],
            horizontal=True
        )

        if search_type == "Medicine Name":
            search = st.selectbox(
                "Select Medicine",
                options=medicine_names
            )
            med_df = df[df['product_name'] == search]
        else:
            search = st.selectbox(
                "Select Product ID",
                options=product_ids
            )
            med_df = df[df['product_id'] == search]

        latest = med_df.iloc[-1]

        st.markdown("---")

        st.markdown(
            "<h4 style='color:green;'>Medicine Information</h4>",
            unsafe_allow_html=True
        )

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.info(f"**Product ID**\n\n{latest['product_id']}")
            st.info(f"**Medicine Name**\n\n{latest['product_name']}")

        with info_col2:
            st.success(f"**Current Stock**\n\n{latest['units_in_stock']}")
            st.success(f"**Category**\n\n{latest['category']}")

        with info_col3:
            st.warning(f"**Restocked On**\n\n{latest['restock_date']}")
            st.warning(f"**Days in Stock**\n\n{latest['days_since_restock']}")

        st.markdown("---")

        st.markdown(
            "<h4 style='color:green;'>Batch-wise Expiry Dates</h4>",
            unsafe_allow_html=True
        )

        expiry_batches = med_df[
            ['batch_number', 'expiration_date']
        ].drop_duplicates()

        expiry_batches = expiry_batches.sort_values(
            by='expiration_date'
        )

        st.dataframe(expiry_batches, hide_index=True)

        st.markdown("---")

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.markdown(
                "<h4 style='color:green;'>Selected Medicine Trend</h4>",
                unsafe_allow_html=True
            )

            med_sales_trend = med_df.groupby(
                'sale_month'
            )['units_sold'].sum().sort_index()

            fig, ax = plt.subplots(figsize=(4.8, 3))

            ax.plot(
                med_sales_trend.index,
                med_sales_trend.values,
                marker='o',
                color='green',
                linewidth=2
            )

            ax.set_xlabel("Month")
            ax.set_ylabel("Units Sold")
            ax.set_title("Medicine Sales")
            ax.grid(True)

            st.pyplot(fig)

        with graph_col2:
            st.markdown(
                "<h4 style='color:green;'>Company Avg Sales</h4>",
                unsafe_allow_html=True
            )

            company_avg_sales = df.groupby(
                'sale_month'
            )['units_sold'].mean().sort_index()

            fig2, ax2 = plt.subplots(figsize=(4.8, 3))

            ax2.bar(
                company_avg_sales.index,
                company_avg_sales.values
            )

            ax2.set_xlabel("Month")
            ax2.set_ylabel("Avg Units")
            ax2.set_title("Company Average")
            ax2.grid(axis='y')

            st.pyplot(fig2)

        st.markdown("---")

        st.markdown(
            "<h4 style='color:green;'>Sales Comparison Dashboard</h4>",
            unsafe_allow_html=True
        )

        comparison_df = pd.DataFrame({
            'Selected Medicine': med_sales_trend,
            'Company Average': company_avg_sales
        }).fillna(0)

        fig3, ax3 = plt.subplots(figsize=(6.5, 3.2))

        ax3.plot(
            comparison_df.index,
            comparison_df['Selected Medicine'],
            marker='o',
            linewidth=2,
            label='Selected Medicine'
        )

        ax3.plot(
            comparison_df.index,
            comparison_df['Company Average'],
            marker='s',
            linewidth=2,
            linestyle='--',
            label='Company Average'
        )

        ax3.set_xlabel("Month")
        ax3.set_ylabel("Units Sold")
        ax3.set_title("Comparison Trend")
        ax3.legend()
        ax3.grid(True)

        st.pyplot(fig3)
    return result
