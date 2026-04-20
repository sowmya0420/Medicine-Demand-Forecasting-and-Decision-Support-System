import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def run_obj2_alerts(df, result):
    st.title("Module 3 - Intelligent Alert Generation System")

    # =========================================
    # Merge store details
    # =========================================
    store_cols = [
        "product_name",
        "store_id",
        "store_name",
        "store_street_address",
        "store_city",
        "store_state",
        "units_in_stock"
    ]

    store_info = df[store_cols].drop_duplicates()

    result = pd.merge(
        result,
        store_info,
        on=["product_name", "units_in_stock"],
        how="left"
    )

    tab1, tab2 = st.tabs(
        ["1. Active Alerts", "2. Analytics Dashboard"]
    )

    # =========================================
    # TAB 1 - ACTIVE ALERTS
    # =========================================
    with tab1:
        st.subheader("Real-Time Intelligent Alerts")

        alerts = []

        for _, row in result.iterrows():
            product = row["product_name"]
            stock = row["units_in_stock"]
            pred = row["predicted_sales"]
            status = row["status"]

            current_store = row["store_name"]
            current_city = row["store_city"]

            # =====================================
            # LOW STOCK ALERT
            # =====================================
            if status == "Understocked":
                required_qty = max(pred - stock, 0)

                other_stores = df[
                    (df["product_name"] == product) &
                    (df["store_name"] != current_store)
                ]

                transfer_suggestion = None

                if not other_stores.empty:
                    donor = other_stores.loc[
                        other_stores["units_in_stock"].idxmax()
                    ]

                    if donor["units_in_stock"] > required_qty:
                        transfer_suggestion = (
                            f"Transfer {required_qty} units from "
                            f"{donor['store_name']} "
                            f"({donor['store_city']})"
                        )

                if transfer_suggestion:
                    action = transfer_suggestion
                else:
                    action = (
                        f"Place order for {required_qty} units "
                        f"for next 7 days demand"
                    )

                alerts.append([
                    current_store,
                    current_city,
                    product,
                    "Low Stock",
                    action,
                    "High"
                ])

            # =====================================
            # HIGH STOCK ALERT
            # =====================================
            elif status == "Overstock":
                alerts.append([
                    current_store,
                    current_city,
                    product,
                    "High Stock",
                    "Transfer to low-demand store / reduce reorder",
                    "Medium"
                ])

            # =====================================
            # DEMAND SPIKE
            # =====================================
            elif pred > row["avg_sales"] * 1.2:
                alerts.append([
                    current_store,
                    current_city,
                    product,
                    "Demand Spike",
                    "Increase reorder frequency and monitor trend",
                    "Medium"
                ])

        # =====================================
        # EXPIRY RISK ALERT
        # =====================================
        if "expiration_date" in df.columns:
            temp = df.copy()

            temp["expiration_date"] = pd.to_datetime(
                temp["expiration_date"]
            )

            today = pd.Timestamp.today()

            temp["days_left"] = (
                temp["expiration_date"] - today
            ).dt.days

            risky = temp[temp["days_left"] <= 30]

            for _, row in risky.iterrows():
                med = row["product_name"]
                current_store = row["store_name"]
                current_city = row["store_city"]
                current_stock = row["units_in_stock"]

                other_stores = df[
                    (df["product_name"] == med) &
                    (df["store_name"] != current_store)
                ]

                transfer_store = None

                if not other_stores.empty:
                    needy_store = other_stores.loc[
                        other_stores["units_in_stock"].idxmin()
                    ]

                    if needy_store["units_in_stock"] < current_stock * 0.5:
                        transfer_store = needy_store["store_name"]

                if transfer_store:
                    action = (
                        f"Transfer near-expiry stock to "
                        f"{transfer_store} (low stock)"
                    )
                else:
                    action = (
                        "Offer 15% discount / combo sale"
                    )

                alerts.append([
                    current_store,
                    current_city,
                    med,
                    "Expiry Risk",
                    action,
                    "High"
                ])

        alert_df = pd.DataFrame(
            alerts,
            columns=[
                "Store",
                "City",
                "Medicine",
                "Alert Type",
                "Suggested Action",
                "Risk Level"
            ]
        )

        # =====================================
        # SEARCH BAR
        # =====================================
        st.markdown("### Search Medicine Alerts")

        medicine_search = st.text_input(
            "Search by medicine name"
        )

        if medicine_search:
            alert_df = alert_df[
                alert_df["Medicine"]
                .str.contains(
                    medicine_search,
                    case=False,
                    na=False
                )
            ]

        # =====================================
        # METRICS
        # =====================================
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Alerts", len(alert_df))
        col2.metric(
            "High Risk",
            (alert_df["Risk Level"] == "High").sum()
        )
        col3.metric(
            "Medium Risk",
            (alert_df["Risk Level"] == "Medium").sum()
        )

        st.markdown("---")

        # =====================================
        # ALERT TABLE
        # =====================================
        st.markdown("### Active Alerts Table")
        st.dataframe(alert_df, hide_index=True)

    # =========================================
    # TAB 2 - ANALYTICS
    # =========================================
    with tab2:
        st.subheader("Analytics Dashboard")

        col1, col2 = st.columns(2)

        # =====================================
        # PROFIT VS EXPENSES
        # =====================================
        with col1:
            monthly_sales = df.groupby(
                "sale_month"
            )["units_sold"].sum().sort_index()

            profit = monthly_sales * 20
            expenses = monthly_sales * 12

            fig1, ax1 = plt.subplots(figsize=(5, 3))

            ax1.plot(
                monthly_sales.index,
                profit.values,
                marker="o",
                linewidth=2,
                label="Profit"
            )

            ax1.plot(
                monthly_sales.index,
                expenses.values,
                marker="s",
                linewidth=2,
                linestyle="--",
                label="Expenses"
            )

            ax1.set_title("Profit vs Expenses")
            ax1.set_xlabel("Month")
            ax1.set_ylabel("Amount")
            ax1.legend()
            ax1.grid(True)

            st.pyplot(fig1)

            return alert_df
       
