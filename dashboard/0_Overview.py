"""Uber NYC Hot-Zones — Streamlit Dashboard."""
import streamlit as st

st.set_page_config(
    page_title="Uber NYC Hot-Zones",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Uber NYC Hot-Zones Analysis")

st.markdown(
    "This project analyzes Uber pickup patterns across New York City "
    "using two complementary datasets: GPS coordinates (Apr-Sep 2014) "
    "and zone-level records (Jan-Jun 2015). "
    "The goal is to identify spatial hot-zones via clustering and "
    "measure year-over-year growth dynamics."
)

st.image(
    "dashboard/static/chloropeth_pickups-by-zone-2014.png",
    caption="NYC taxi zones colored by 2014 pickup count (log scale)",
)

st.markdown("---")

st.subheader("Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        "**Spatial Analysis (2014 GPS)**\n"
        "- Manhattan dominates pickup activity, "
        "concentrating the vast majority of rides\n"
        "- k=11 clusters identified via the elbow method "
        "validated by geometric elbow detection\n"
        "- DBSCAN was evaluated but rejected -- "
        "uniform density in core areas yields no natural clusters"
    )

with col2:
    st.markdown(
        "**Year-over-Year Dynamics (Apr-Jun 2014 vs 2015)**\n"
        "- 321.7% growth in pickups between April-June 2014 and 2015\n"
        "- Peak hour shifted from 17:00 (2014) to 19:00 (2015), "
        "suggesting increased evening/leisure usage\n"
        "- Weekend adoption grew -- weekday share dropped "
        "from 76.3% to 69.5%"
    )

st.markdown("---")

st.subheader("Driver Recommendations")

st.markdown(
    "Based on the clustering and temporal analysis, "
    "here are actionable positioning strategies by time of day:"
)

st.markdown(
    "1. **Morning rush (7-9 AM):** Position near business districts "
    "and transit hubs (Midtown, Financial District, Penn Station area)\n"
    "2. **Evening rush (5-7 PM):** High demand in office areas "
    "-- Midtown East/West, Chelsea, Union Square\n"
    "3. **Late night (10 PM-2 AM):** Focus on nightlife and "
    "entertainment districts (Union Sq, TriBeCa, Williamsburg) "
    "and residential areas like Park Slope\n"
    "4. **Weekends:** Demand shifts toward entertainment and "
    "residential areas; airports maintain steady volume\n"
    "5. **Growth zones:** Expanding outer-borough areas "
    "(Jackson Heights +1432%, Astoria +939%) offer lower "
    "competition with growing demand"
)

st.markdown("---")

st.subheader("Pages")

st.markdown(
    "- **Hot Zone Explorer:** interactive map with hot-zone clusters, "
    "temporal charts, and top zones ranking\n"
    "- **Year Comparison:** year-over-year comparison of pickups "
    "between April-June 2014 and 2015\n"
    "- **Methodology:** clustering methodology, algorithm comparison, "
    "and limitations\n"
    "- **Glossary:** definitions of key terms, metrics, "
    "and acronyms used throughout the dashboard"
)
