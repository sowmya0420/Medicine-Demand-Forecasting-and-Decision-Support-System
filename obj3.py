import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')


def run_obj3_classification(df):

    st.title("Module 1: Fuzzy Logic-Based Medicine Movement Classification")

    # Create 3 internal tabs
    tab1, tab2, tab3 = st.tabs([
        "Data Analysis",
        "Fuzzy Logic Setup",
        "Classification Results"
    ])

    # ============================================================
    # Aggregate Data
    # ============================================================
    product_summary = df.groupby(['product_id']).agg(
        product_name=('product_name', 'first'),
        category=('category', 'first'),
        total_units_sold=('units_sold', 'sum'),
        transaction_count=('sale_id', 'count'),
        avg_units_sold=('units_sold', 'mean')
    ).reset_index()

    # ============================================================
    # TAB 1 — DATA ANALYSIS
    # ============================================================
    with tab1:
        st.subheader("Step 1: Data Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Records", len(df))

        with col2:
            st.metric("Total Unique Medicines", df['product_id'].nunique())
        st.subheader("Step 2: Input Variable Distributions")

        fig1, axes = plt.subplots(1, 3, figsize=(16, 4))

        vars_info = [
            ('total_units_sold', 'Total Units Sold', 'steelblue'),
            ('transaction_count', 'Transaction Count', 'teal'),
            ('avg_units_sold', 'Avg Units Per Sale', 'coral'),
        ]

        for ax, (col, label, color) in zip(axes, vars_info):
            ax.hist(product_summary[col], bins=25, color=color, edgecolor='white', alpha=0.85)
            ax.axvline(product_summary[col].quantile(0.25), color='green', linestyle='--', label="Q1")
            ax.axvline(product_summary[col].quantile(0.50), color='orange', linestyle='--', label="Median")
            ax.axvline(product_summary[col].quantile(0.75), color='red', linestyle='--', label="Q3")
            ax.set_title(label)
            ax.set_xlabel(label)
            ax.set_ylabel('Number of Products')
            ax.legend()

        st.pyplot(fig1)

        st.write("Quartiles used to define fuzzy membership boundaries.")

    # ============================================================
    # STEP 2 — FUZZY MEMBERSHIP FUNCTIONS
    # ============================================================
    
    # --- Universe of discourse ---
    total_sold_range  = np.arange(0,     110010, 1)
    trans_count_range = np.arange(30,    100,    0.1)
    avg_sold_range    = np.arange(0,     2001,   0.1)
    output_range      = np.arange(0,     10.1,   0.1)

    # --- Input: total_units_sold ---
    total_low    = fuzz.trapmf(total_sold_range, [0,    0,    165,  292])
    total_medium = fuzz.trapmf(total_sold_range, [165,  292,  446,  968])
    total_high   = fuzz.trapmf(total_sold_range, [446,  968,  110009, 110009])

    # --- Input: transaction_count ---
    trans_low    = fuzz.trapmf(trans_count_range, [30,  30,  41,  50])
    trans_medium = fuzz.trapmf(trans_count_range, [41,  45,  56,  60])
    trans_high   = fuzz.trapmf(trans_count_range, [56,  60,  98,  98])

    # --- Input: avg_units_sold ---
    avg_low    = fuzz.trapmf(avg_sold_range, [0,    0,    3.08, 5.96])
    avg_medium = fuzz.trapmf(avg_sold_range, [3.08, 5.96, 9.0,  18.12])
    avg_high   = fuzz.trapmf(avg_sold_range, [9.0,  18.12, 2001, 2001])

    # --- Output: movement score (0-10) ---
    out_slow     = fuzz.trimf(output_range, [0,  1,  3])
    out_moderate = fuzz.trimf(output_range, [2,  5,  7])
    out_fast     = fuzz.trimf(output_range, [6,  8,  10])


    # ============================================================
    # TAB 2 — FUZZY LOGIC SETUP (STREAMLIT)
    # ============================================================
    with tab2:
        st.subheader("Step 2: Fuzzy Membership Functions")
        st.write("Boundaries experimentally set from quartile analysis")

        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))

        # total_units_sold memberships
        ax = axes2[0, 0]
        ax.plot(total_sold_range, total_low,    'b',  linewidth=2, label='Low (slow)')
        ax.plot(total_sold_range, total_medium, 'g',  linewidth=2, label='Medium (moderate)')
        ax.plot(total_sold_range, total_high,   'r',  linewidth=2, label='High (fast)')
        ax.set_xlim([0, 5000])
        ax.set_title('Total Units Sold — Membership Functions')
        ax.set_xlabel('Total Units Sold')
        ax.set_ylabel('Membership Degree (0 to 1)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.annotate('Q1=165', xy=(165, 0.5), fontsize=7, color='gray')
        ax.annotate('Median=292', xy=(292, 0.8), fontsize=7, color='gray')
        ax.annotate('Q3=446', xy=(446, 0.5), fontsize=7, color='gray')

        # transaction_count memberships
        ax = axes2[0, 1]
        ax.plot(trans_count_range, trans_low,    'b',  linewidth=2, label='Low')
        ax.plot(trans_count_range, trans_medium, 'g',  linewidth=2, label='Medium')
        ax.plot(trans_count_range, trans_high,   'r',  linewidth=2, label='High')
        ax.set_title('Transaction Count — Membership Functions')
        ax.set_xlabel('Transaction Count')
        ax.set_ylabel('Membership Degree (0 to 1)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # avg_units_sold memberships
        ax = axes2[1, 0]
        ax.plot(avg_sold_range, avg_low,    'b',  linewidth=2, label='Low')
        ax.plot(avg_sold_range, avg_medium, 'g',  linewidth=2, label='Medium')
        ax.plot(avg_sold_range, avg_high,   'r',  linewidth=2, label='High')
        ax.set_xlim([0, 100])
        ax.set_title('Avg Units Per Sale — Membership Functions')
        ax.set_xlabel('Avg Units Sold')
        ax.set_ylabel('Membership Degree (0 to 1)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # output memberships
        ax = axes2[1, 1]
        ax.plot(output_range, out_slow,     'b',  linewidth=2, label='Slow Moving')
        ax.plot(output_range, out_moderate, 'g',  linewidth=2, label='Moderate')
        ax.plot(output_range, out_fast,     'r',  linewidth=2, label='Fast Moving')
        ax.set_title('Output — Movement Score (0 to 10)')
        ax.set_xlabel('Score')
        ax.set_ylabel('Membership Degree (0 to 1)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig2)

    # ============================================================
    # STEP 3 — FUZZY CLASSIFICATION
    # ============================================================
    scores = []
    labels = []

    for _, row in product_summary.iterrows():
        total = row['total_units_sold']
        trans = row['transaction_count']
        avg   = row['avg_units_sold']

        t_low  = fuzz.interp_membership(total_sold_range, total_low, min(total, 110009))
        t_med  = fuzz.interp_membership(total_sold_range, total_medium, min(total, 110009))
        t_high = fuzz.interp_membership(total_sold_range, total_high, min(total, 110009))

        tr_low  = fuzz.interp_membership(trans_count_range, trans_low, trans)
        tr_med  = fuzz.interp_membership(trans_count_range, trans_medium, trans)
        tr_high = fuzz.interp_membership(trans_count_range, trans_high, trans)

        a_low  = fuzz.interp_membership(avg_sold_range, avg_low, min(avg, 2001))
        a_med  = fuzz.interp_membership(avg_sold_range, avg_medium, min(avg, 2001))
        a_high = fuzz.interp_membership(avg_sold_range, avg_high, min(avg, 2001))

        fast = max(min(t_high, a_high), min(t_high, a_med), tr_high)
        mod  = max(min(t_med, a_med), min(t_med, a_low), tr_med)
        slow = max(min(t_low, a_low), min(t_low, a_med), tr_low)

        score = (fast*8 + mod*5 + slow*2) / (fast + mod + slow + 1e-6)

        if score >= 6.5:
            label = 'Fast Moving'
        elif score >= 3.5:
            label = 'Moderate'
        else:
            label = 'Slow Moving'

        scores.append(score)
        labels.append(label)

    product_summary['fuzzy_score'] = scores
    product_summary['movement_class'] = labels

    # ============================================================
    # TAB 3 — RESULTS
    # ============================================================
    with tab3:
        st.subheader("Step 3: Classification Results")

        # -------------------------------
        # Charts
        # -------------------------------
        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))

        # Pie Chart
        product_summary['movement_class'].value_counts().plot(
            kind='pie', autopct='%1.1f%%', ax=axes3[0]
        )
        axes3[0].set_title("Products by Movement Class")

        # Score Histogram
        axes3[1].hist(product_summary['fuzzy_score'], bins=20)
        axes3[1].axvline(3.5, color='orange', linestyle='--')
        axes3[1].axvline(6.5, color='green', linestyle='--')
        axes3[1].set_title("Fuzzy Score Distribution")

        # Scatter Plot
        colors = product_summary['movement_class'].map({
            'Fast Moving': 'green',
            'Moderate': 'orange',
            'Slow Moving': 'red'
        })

        axes3[2].scatter(product_summary['total_units_sold'],
                        product_summary['avg_units_sold'],
                        c=colors)
        axes3[2].set_title("Total vs Avg Units Sold")
        axes3[2].set_xlabel("Total Units Sold")
        axes3[2].set_ylabel("Avg Units Sold")

        st.pyplot(fig3)

        st.markdown("---")
        st.subheader("Medicine Classification Tables")

        # -------------------------------
        # Separate Tables
        # -------------------------------
        fast_df = product_summary[product_summary['movement_class'] == 'Fast Moving']
        mod_df  = product_summary[product_summary['movement_class'] == 'Moderate']
        slow_df = product_summary[product_summary['movement_class'] == 'Slow Moving']

        fast_count = len(fast_df)
        mod_count  = len(mod_df)
        slow_count = len(slow_df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"### Fast Moving Medicines — {fast_count}")
            st.dataframe(
                fast_df[['product_name', 'total_units_sold', 'fuzzy_score']],
                use_container_width=True
            )

        with col2:
            st.markdown(f"### Moderate Medicines — {mod_count}")
            st.dataframe(
                mod_df[['product_name', 'total_units_sold', 'fuzzy_score']],
                use_container_width=True
            )

        with col3:
            st.markdown(f"### Slow Moving Medicines — {slow_count}")
            st.dataframe(
                slow_df[['product_name', 'total_units_sold', 'fuzzy_score']],
                use_container_width=True
            )

        # Download button 
        st.markdown("---")
        st.download_button(
            label="Download Classification Results CSV",
            data=product_summary.to_csv(index=False),
            file_name="classified_medicines.csv",
            mime="text/csv"
        )