import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# FUZZY SETUP  (universes + membership functions)
# ============================================================

expiry_range = np.arange(-500, 1100, 1)   # days_to_expiry
demand_range = np.arange(0,    2010, 1)    # avg units sold
stock_range  = np.arange(0,    5100, 1)    # units in stock
risk_range   = np.arange(0,    10.1, 0.1)  # output risk score

# -- Expiry Urgency --
exp_critical = fuzz.trapmf(expiry_range, [-500, -500,   0,  30])
exp_warning  = fuzz.trapmf(expiry_range, [   0,   30,  60,  90])
exp_safe     = fuzz.trapmf(expiry_range, [  60,   90, 1099, 1099])

# -- Demand Level --
dem_low    = fuzz.trapmf(demand_range, [0,  0,   3,   6])
dem_medium = fuzz.trapmf(demand_range, [3,  6,   9,  18])
dem_high   = fuzz.trapmf(demand_range, [9, 18, 2009, 2009])

# -- Stock Level --
stk_low    = fuzz.trapmf(stock_range, [0,   0,   57,  83])
stk_medium = fuzz.trapmf(stock_range, [57,  83, 142, 200])
stk_high   = fuzz.trapmf(stock_range, [142, 200, 5009, 5009])

# -- Output: Risk Score (0–10) --
risk_low    = fuzz.trimf(risk_range, [0, 1,  3])
risk_medium = fuzz.trimf(risk_range, [2, 5,  7])
risk_high   = fuzz.trimf(risk_range, [6, 8, 10])


# ============================================================
# FUZZY INFERENCE ENGINE
# ============================================================

def compute_risk_score(days_to_exp, avg_sold, in_stock):
    exp = float(np.clip(days_to_exp, -500, 1099))
    dem = float(np.clip(avg_sold,    0,    2009))
    stk = float(np.clip(in_stock,    0,    5009))

    e_crit = fuzz.interp_membership(expiry_range, exp_critical, exp)
    e_warn = fuzz.interp_membership(expiry_range, exp_warning,  exp)
    e_safe = fuzz.interp_membership(expiry_range, exp_safe,     exp)

    d_low  = fuzz.interp_membership(demand_range, dem_low,    dem)
    d_med  = fuzz.interp_membership(demand_range, dem_medium, dem)
    d_high = fuzz.interp_membership(demand_range, dem_high,   dem)

    s_low  = fuzz.interp_membership(stock_range, stk_low,    stk)
    s_med  = fuzz.interp_membership(stock_range, stk_medium, stk)
    s_high = fuzz.interp_membership(stock_range, stk_high,   stk)

    # Fuzzy Rules
    # HIGH RISK rules
    act_high = max(
        min(e_crit, s_high),            # expired + too much stock
        min(e_crit, s_high, d_low),     # expired + high stock + low demand
        min(e_crit, s_med),             # expired + medium stock
        min(e_warn, s_high, d_low),     # warning + high stock + low demand
    )
    # MEDIUM RISK rules
    act_med = max(
        min(e_warn, s_med),             # warning + medium stock
        min(e_warn, s_high, d_med),     # warning + high stock + medium demand
        min(e_crit, s_low, d_med),      # expired but low stock + medium demand
    )
    # LOW RISK rules
    act_low = max(
        min(e_safe, d_high),            # safe expiry + fast selling
        min(e_safe, s_low),             # safe expiry + low stock (will be used up)
        min(e_warn, s_low, d_high),     # warning but fast selling + low stock
    )

    # Aggregate
    agg = np.fmax(
        np.fmin(act_high, risk_high),
        np.fmax(
            np.fmin(act_med,  risk_medium),
            np.fmin(act_low,  risk_low)
        )
    )

    if agg.sum() == 0:
        return 0.0
    return round(fuzz.defuzz(risk_range, agg, 'centroid'), 2)


def risk_label(score):
    if score >= 6:
        return 'High Risk'
    elif score >= 3:
        return 'Medium Risk'
    return 'Low Risk'


def suggested_action(row):
    score     = row['risk_score']
    days      = row['min_days_to_expiry']
    demand    = row['avg_units_sold']
    stock     = row['units_in_stock']
    prod_name = row['product_name']
    src_store = row['store_name']
    df_full   = row['_df']

    # Find needy stores (same medicine, different store, lower stock)
    other = df_full[
        (df_full['product_name'] == prod_name) &
        (df_full['store_name']   != src_store)
    ]

    if score >= 6:
        if days < 0:
            base_action = f"EXPIRED – Remove {int(stock)} units from shelf immediately"
        else:
            transfer_qty = max(int(stock * 0.6), 1)
            if not other.empty:
                needy = other.loc[other['units_in_stock'].idxmin()]
                if needy['units_in_stock'] < stock * 0.5:
                    return (
                        f"Transfer {transfer_qty} units to "
                        f"{needy['store_name']} ({needy['store_city']}) "
                        f"[lower stock: {int(needy['units_in_stock'])} units]"
                    )
            return f"Apply 20% discount & promote to clear {int(stock)} units (expires in {int(days)} days)"

    elif score >= 3:
        if not other.empty:
            needy = other.loc[other['units_in_stock'].idxmin()]
            if needy['units_in_stock'] < stock * 0.7:
                transfer_qty = max(int((stock - needy['units_in_stock']) // 2), 1)
                return (
                    f"Partial transfer of {transfer_qty} units to "
                    f"{needy['store_name']} ({needy['store_city']})"
                )
        return f"Monitor closely – {int(days)} days to expiry, consider 10% discount"

    else:
        return "No action needed – stock is safe"

    return base_action


# ============================================================
# MAIN MODULE FUNCTION
# ============================================================

def run_obj4_expiry_optimization(df):

    st.title("Module 4 – Expiry-Aware Stock Optimization (Fuzzy Logic)")

    # ---- Aggregate per product ----
    prod = df.groupby(['product_id', 'product_name']).agg(
        avg_units_sold=('units_sold',     'mean'),
        units_in_stock=('units_in_stock', 'last'),
        min_days_to_expiry=('days_to_expiry', 'min'),
        store_name=('store_name', 'first'),
        store_city=('store_city', 'first'),
        category=('category',    'first')
    ).reset_index()

    # ---- Compute fuzzy risk scores ----
    with st.spinner("Running fuzzy inference engine..."):
        prod['risk_score'] = prod.apply(
            lambda r: compute_risk_score(
                r['min_days_to_expiry'],
                r['avg_units_sold'],
                r['units_in_stock']
            ), axis=1
        )
        prod['risk_level'] = prod['risk_score'].apply(risk_label)

        # Pass full df into each row for redistribution logic
        prod['_df'] = [df] * len(prod)
        prod['action'] = prod.apply(suggested_action, axis=1)
        prod.drop(columns=['_df'], inplace=True)

    # ============================================================
    # KPI ROW
    # ============================================================
    high_count   = (prod['risk_level'] == 'High Risk').sum()
    medium_count = (prod['risk_level'] == 'Medium Risk').sum()
    low_count    = (prod['risk_level'] == 'Low Risk').sum()
    expired_count = (prod['min_days_to_expiry'] < 0).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 High Risk",    high_count)
    c2.metric("🟡 Medium Risk",  medium_count)
    c3.metric("🟢 Low Risk",     low_count)
    c4.metric("☠️ Already Expired", expired_count)

    st.markdown("---")

    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3 = st.tabs([
        "1. Fuzzy Logic Setup",
        "2. Expiry Risk Dashboard",
        "3. Redistribution Plan"
    ])

    # ============================================================
    # TAB 1 – FUZZY MEMBERSHIP FUNCTIONS
    # ============================================================
    with tab1:
        st.subheader("Fuzzy Membership Functions")
        st.write(
            "Three input variables feed the Mamdani fuzzy inference engine. "
            "Boundaries are derived from dataset quartiles."
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # -- Expiry Urgency --
        ax = axes[0, 0]
        ax.plot(expiry_range, exp_critical, 'r',  linewidth=2, label='Critical (≤30 days)')
        ax.plot(expiry_range, exp_warning,  'orange', linewidth=2, label='Warning (30–90 days)')
        ax.plot(expiry_range, exp_safe,     'g',  linewidth=2, label='Safe (>90 days)')
        ax.set_xlim([-150, 400])
        ax.set_title('Input 1 – Expiry Urgency (days to expiry)')
        ax.set_xlabel('Days to Expiry')
        ax.set_ylabel('Membership Degree')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(0,  color='red',   linestyle='--', alpha=0.4, linewidth=1)
        ax.axvline(30, color='orange',linestyle='--', alpha=0.4, linewidth=1)
        ax.axvline(90, color='green', linestyle='--', alpha=0.4, linewidth=1)
        ax.annotate('Expired', xy=(-120, 0.85), color='red', fontsize=8)
        ax.annotate('0d', xy=(2, 0.5), color='red', fontsize=7)
        ax.annotate('30d', xy=(32, 0.5), color='orange', fontsize=7)
        ax.annotate('90d', xy=(92, 0.5), color='green', fontsize=7)

        # -- Demand Level --
        ax = axes[0, 1]
        ax.plot(demand_range, dem_low,    'b', linewidth=2, label='Low  (avg ≤ 6 units)')
        ax.plot(demand_range, dem_medium, 'g', linewidth=2, label='Medium (6–18 units)')
        ax.plot(demand_range, dem_high,   'r', linewidth=2, label='High (≥ 18 units)')
        ax.set_xlim([0, 60])
        ax.set_title('Input 2 – Demand Level (avg units sold)')
        ax.set_xlabel('Average Units Sold')
        ax.set_ylabel('Membership Degree')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # -- Stock Level --
        ax = axes[1, 0]
        ax.plot(stock_range, stk_low,    'b', linewidth=2, label='Low  (≤ 83 units)')
        ax.plot(stock_range, stk_medium, 'g', linewidth=2, label='Medium (83–200)')
        ax.plot(stock_range, stk_high,   'r', linewidth=2, label='High (≥ 200)')
        ax.set_xlim([0, 600])
        ax.set_title('Input 3 – Stock Level (units in stock)')
        ax.set_xlabel('Units in Stock')
        ax.set_ylabel('Membership Degree')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # -- Output: Risk Score --
        ax = axes[1, 1]
        ax.plot(risk_range, risk_low,    'g', linewidth=2, label='Low Risk  (0–3)')
        ax.plot(risk_range, risk_medium, 'orange', linewidth=2, label='Medium Risk (3–6)')
        ax.plot(risk_range, risk_high,   'r', linewidth=2, label='High Risk  (6–10)')
        ax.set_title('Output – Expiry Risk Score (0–10)')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Membership Degree')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(3, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(6, color='red',    linestyle='--', alpha=0.5, linewidth=1)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Fuzzy Rules Applied")

        rules_data = {
            "Rule": [
                "R1", "R2", "R3", "R4",
                "R5", "R6", "R7",
                "R8", "R9", "R10"
            ],
            "Expiry Urgency": [
                "Critical", "Critical", "Critical", "Warning",
                "Warning",  "Warning",  "Critical",
                "Safe",     "Safe",     "Warning"
            ],
            "Stock Level": [
                "High", "High", "Medium", "High",
                "Medium", "High", "Low",
                "Any", "Low", "Low"
            ],
            "Demand Level": [
                "Any", "Low", "Any", "Low",
                "Any", "Medium", "Medium",
                "High", "Any", "High"
            ],
            "→ Risk Output": [
                "High", "High", "High", "High",
                "Medium", "Medium", "Medium",
                "Low", "Low", "Low"
            ]
        }

        rules_df = pd.DataFrame(rules_data)

        def color_risk(val):
            if val == "High":
                return "background-color: #ffcccc"
            elif val == "Medium":
                return "background-color: #fff3cd"
            elif val == "Low":
                return "background-color: #d4edda"
            return ""

        st.dataframe(
            rules_df.style.map(color_risk, subset=["→ Risk Output"]),
            hide_index=True,
            use_container_width=True
        )

    # ============================================================
    # TAB 2 – RISK DASHBOARD
    # ============================================================
    with tab2:
        st.subheader("Expiry Risk Dashboard")

        # Search / filter
        col_search, col_filter = st.columns([2, 1])
        with col_search:
            search_term = st.text_input("Search by medicine name")
        with col_filter:
            risk_filter = st.selectbox(
                "Filter by risk level",
                ["All", "High Risk", "Medium Risk", "Low Risk"]
            )

        display = prod.copy()

        if search_term:
            display = display[
                display['product_name'].str.contains(search_term, case=False, na=False)
            ]
        if risk_filter != "All":
            display = display[display['risk_level'] == risk_filter]

        display_table = display[[
            'product_name', 'category', 'store_name', 'store_city',
            'min_days_to_expiry', 'avg_units_sold', 'units_in_stock',
            'risk_score', 'risk_level'
        ]].copy()

        display_table.columns = [
            'Medicine', 'Category', 'Store', 'City',
            'Days to Expiry', 'Avg Demand', 'Stock',
            'Risk Score', 'Risk Level'
        ]

        display_table.insert(0, 'S.No', range(1, len(display_table) + 1))
        display_table = display_table.sort_values('Risk Score', ascending=False).reset_index(drop=True)
        display_table['S.No'] = range(1, len(display_table) + 1)

        def color_risk_level(val):
            if val == 'High Risk':
                return 'background-color: #ffcccc; color: #7b0000'
            elif val == 'Medium Risk':
                return 'background-color: #fff3cd; color: #7b4f00'
            elif val == 'Low Risk':
                return 'background-color: #d4edda; color: #155724'
            return ''

        def color_days(val):
            try:
                v = float(val)
                if v < 0:
                    return 'background-color: #ffcccc'
                elif v <= 30:
                    return 'background-color: #ffe5b4'
            except:
                pass
            return ''

        st.dataframe(
            display_table.style
                .map(color_risk_level, subset=['Risk Level'])
                .map(color_days,       subset=['Days to Expiry'])
                .format({'Risk Score': '{:.2f}', 'Avg Demand': '{:.1f}'}),
            hide_index=True,
            use_container_width=True
        )

        st.markdown("---")

        # Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### Risk Level Distribution")
            risk_counts = prod['risk_level'].value_counts()
            colors_pie = {
                'High Risk':   '#ff6b6b',
                'Medium Risk': '#ffd93d',
                'Low Risk':    '#6bcb77'
            }
            fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
            pie_colors = [colors_pie.get(l, '#ccc') for l in risk_counts.index]
            ax_pie.pie(
                risk_counts.values,
                labels=risk_counts.index,
                autopct='%1.1f%%',
                colors=pie_colors,
                startangle=140
            )
            ax_pie.set_title("Products by Expiry Risk Level")
            st.pyplot(fig_pie)

        with chart_col2:
            st.markdown("#### Risk Score Distribution")
            fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
            ax_hist.hist(prod['risk_score'], bins=20, color='#4a90d9', edgecolor='white')
            ax_hist.axvline(3, color='orange', linestyle='--', linewidth=1.5, label='Medium threshold (3)')
            ax_hist.axvline(6, color='red',    linestyle='--', linewidth=1.5, label='High threshold (6)')
            ax_hist.set_xlabel('Fuzzy Risk Score')
            ax_hist.set_ylabel('Number of Products')
            ax_hist.set_title('Distribution of Risk Scores')
            ax_hist.legend(fontsize=8)
            ax_hist.grid(True, alpha=0.3)
            st.pyplot(fig_hist)

        # Category breakdown
        st.markdown("#### Risk by Category")
        cat_risk = prod.groupby(['category', 'risk_level']).size().unstack(fill_value=0)
        for col in ['High Risk', 'Medium Risk', 'Low Risk']:
            if col not in cat_risk.columns:
                cat_risk[col] = 0
        cat_risk = cat_risk[['High Risk', 'Medium Risk', 'Low Risk']]

        fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
        cat_risk.plot(
            kind='bar',
            ax=ax_bar,
            color=['#ff6b6b', '#ffd93d', '#6bcb77'],
            edgecolor='white'
        )
        ax_bar.set_xlabel('Category')
        ax_bar.set_ylabel('Number of Products')
        ax_bar.set_title('Expiry Risk Breakdown by Medicine Category')
        ax_bar.legend(title='Risk Level', fontsize=8)
        ax_bar.tick_params(axis='x', rotation=30)
        ax_bar.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_bar)

        st.download_button(
            "Download Risk Report CSV",
            data=display_table.to_csv(index=False),
            file_name="expiry_risk_report.csv",
            mime="text/csv"
        )

    # ============================================================
    # TAB 3 – REDISTRIBUTION PLAN
    # ============================================================
    with tab3:
        st.subheader("Redistribution & Action Plan")

        st.markdown("""
        This plan is generated by the fuzzy inference engine. Actions are prioritized by
        risk score. Redistribution transfers are suggested only when a needy store
        (lower stock of the same medicine) is found in the dataset.
        """)

        action_df = prod[prod['risk_level'].isin(['High Risk', 'Medium Risk'])].copy()
        action_df = action_df.sort_values('risk_score', ascending=False).reset_index(drop=True)

        action_display = action_df[[
            'product_name', 'store_name', 'store_city',
            'min_days_to_expiry', 'units_in_stock',
            'avg_units_sold', 'risk_score', 'risk_level', 'action'
        ]].copy()

        action_display.columns = [
            'Medicine', 'Store', 'City',
            'Days to Expiry', 'Stock',
            'Avg Demand', 'Risk Score', 'Risk Level', 'Suggested Action'
        ]
        action_display.insert(0, 'Priority', range(1, len(action_display) + 1))

        # Medicine selector
        selected_med = st.selectbox(
            "View action detail for a specific medicine:",
            options=["All"] + list(action_display['Medicine'].unique())
        )

        if selected_med != "All":
            action_display = action_display[action_display['Medicine'] == selected_med]

        def color_row(val):
            if val == 'High Risk':
                return 'background-color: #ffcccc; color: #7b0000'
            elif val == 'Medium Risk':
                return 'background-color: #fff3cd; color: #7b4f00'
            return ''

        st.dataframe(
            action_display.style
                .map(color_row, subset=['Risk Level'])
                .format({'Risk Score': '{:.2f}', 'Avg Demand': '{:.1f}'}),
            hide_index=True,
            use_container_width=True
        )

        st.markdown("---")

        # Summary metrics for redistribution
        transfer_rows   = action_display[action_display['Suggested Action'].str.startswith('Transfer')]
        discount_rows   = action_display[action_display['Suggested Action'].str.contains('discount|Discount')]
        remove_rows     = action_display[action_display['Suggested Action'].str.startswith('EXPIRED')]

        m1, m2, m3 = st.columns(3)
        m1.metric("🔄 Transfer Suggestions",  len(transfer_rows))
        m2.metric("🏷️ Discount Suggestions",  len(discount_rows))
        m3.metric("🗑️ Remove from Shelf",     len(remove_rows))

        if not transfer_rows.empty:
            st.markdown("#### Transfer Summary")
            st.dataframe(
                transfer_rows[['Medicine', 'Store', 'City', 'Stock', 'Suggested Action']],
                hide_index=True,
                use_container_width=True
            )

        if not discount_rows.empty:
            st.markdown("#### Discount / Promotion Summary")
            st.dataframe(
                discount_rows[['Medicine', 'Store', 'City', 'Days to Expiry', 'Stock', 'Suggested Action']],
                hide_index=True,
                use_container_width=True
            )

        if not remove_rows.empty:
            st.markdown("#### Expired – Remove from Shelf")
            st.dataframe(
                remove_rows[['Medicine', 'Store', 'City', 'Days to Expiry', 'Stock', 'Suggested Action']],
                hide_index=True,
                use_container_width=True
            )

        st.markdown("---")

        st.download_button(
            "Download Full Redistribution Plan CSV",
            data=action_display.to_csv(index=False),
            file_name="redistribution_plan.csv",
            mime="text/csv"
        )
