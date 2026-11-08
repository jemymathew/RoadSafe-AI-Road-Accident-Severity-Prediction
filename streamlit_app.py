
import streamlit as st
import requests
import os
# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RoadSafe AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ==========================================
# 2. CATEGORY LABELS
# ==========================================

LABELS = {
    "1st_Road_Class": {
        -1: "Unknown",
        1: "Motorway",
        2: "A(M) road",
        3: "A road",
        4: "B road",
        5: "C road",
        6: "Unclassified road"
    },

    "2nd_Road_Class": {
        -1: "Unknown",
        1: "Motorway",
        2: "A(M) road",
        3: "A road",
        4: "B road",
        5: "C road",
        6: "Unclassified road"
    },

    "Road_Type": {
        1: "Roundabout",
        2: "One-way street",
        3: "Dual carriageway",
        6: "Single carriageway",
        7: "Slip road",
        9: "Unknown"
    },

    "Junction_Detail": {
        -1: "Unknown",
        0: "Not at junction / within 20m",
        1: "Roundabout",
        2: "Mini-roundabout",
        3: "T or staggered junction",
        5: "Slip road",
        6: "Crossroads",
        7: "More than 4 arms",
        8: "Private drive or access",
        9: "Other junction"
    },

    "Junction_Control": {
        -1: "Unknown",
        0: "Not at junction / within 20m",
        1: "Authorised person",
        2: "Automatic traffic signal",
        3: "Stop sign",
        4: "Give way or uncontrolled"
    },

    "Light_Conditions": {
        -1: "Unknown",
        1: "Daylight",
        4: "Darkness - lights lit",
        5: "Darkness - lights unlit",
        6: "Darkness - no lighting",
        7: "Darkness - lighting unknown"
    },

    "Weather_Conditions": {
        1: "Fine, no high winds",
        2: "Raining, no high winds",
        3: "Snowing, no high winds",
        4: "Fine, high winds",
        5: "Raining, high winds",
        6: "Snowing, high winds",
        7: "Fog or mist",
        8: "Other",
        9: "Unknown"
    },

    "Road_Surface_Conditions": {
        -1: "Unknown",
        1: "Dry",
        2: "Wet or damp",
        3: "Snow",
        4: "Frost or ice",
        5: "Flood over 3cm"
    },

    "Urban_or_Rural_Area": {
        1: "Urban",
        2: "Rural",
        3: "Unallocated"
    },

    "Day_of_Week": {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    },

    "Pedestrian_Human": {
        0: "None within 50 metres",
        1: "School crossing patrol",
        2: "Other authorised person"
    },

    "Pedestrian_Physical": {
        0: "No physical crossing",
        1: "Zebra crossing",
        4: "Pelican, puffin or toucan crossing",
        5: "Pedestrian phase at traffic signal",
        7: "Footbridge or subway",
        8: "Central refuge"
    }
}


def display_label(feature, value):
    mapping = LABELS.get(feature, {})
    description = mapping.get(value)

    if description:
        return f"{description} (Code {value})"

    return f"Code {value}"


def select_code(label, feature, options, default_index=0):
    return st.selectbox(
        label,
        options=options,
        index=default_index,
        format_func=lambda value: display_label(feature, value)
    )


# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F3F6FA;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #102B46, #1D5279);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 22px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 5px;
}

.hero p {
    color: #DCEAF5;
    font-size: 15px;
}

.section-heading {
    color: #173B5E;
    font-size: 21px;
    font-weight: 700;
    margin: 15px 0;
}

div[data-testid="stForm"] {
    background: white;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #E1E8F0;
}

.stButton button,
.stFormSubmitButton button {
    background: #173B5E;
    color: white;
    border-radius: 8px;
    border: none;
    min-height: 45px;
    font-weight: 600;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #2563EB;
    color: white;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E1E8F0;
}

.footer {
    text-align: center;
    color: #64748B;
    font-size: 12px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. APPLICATION HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🚦 RoadSafe AI</h1>
    <h3 style="color:white;">
        Road Accident Severity Prediction
    </h3>
    <p>
        Machine-learning based classification of accident
        severity using historical accident conditions.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 5. API STATUS
# ==========================================

col_status, col_model = st.columns(2)

try:
    api_response = requests.get(f"{API_URL}/", timeout=5)
    api_online = api_response.status_code == 200
except requests.exceptions.RequestException:
    api_online = False

with col_status:
    if api_online:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Disconnected")

with col_model:
    st.info("Model: Random Forest")

st.warning(
    "This application classifies recorded accident conditions. "
    "It does not predict whether an accident will occur or "
    "establish causal relationships. It is not intended for "
    "operational road-safety decisions."
)

st.divider()


# ==========================================
# 6. INPUT FORM
# ==========================================

st.markdown(
    '<div class="section-heading">📍 Road and Location Conditions</div>',
    unsafe_allow_html=True
)

with st.form("accident_form"):

    col1, col2 = st.columns(2)

    with col1:

        police_force = st.number_input(
            "Police Force Code",
            min_value=1,
            max_value=100,
            value=1,
            help="Enter the original STATS19 police force code."
        )

        road_class_1 = select_code(
            "First Road Class",
            "1st_Road_Class",
            [-1, 1, 2, 3, 4, 5, 6]
        )

        road_class_2 = select_code(
            "Second Road Class",
            "2nd_Road_Class",
            [-1, 1, 2, 3, 4, 5, 6]
        )

        road_type = select_code(
            "Road Type",
            "Road_Type",
            [1, 2, 3, 6, 7, 9],
            3
        )

        junction_detail = select_code(
            "Junction Detail",
            "Junction_Detail",
            [-1, 0, 1, 2, 3, 5, 6, 7, 8, 9]
        )

        junction_control = select_code(
            "Junction Control",
            "Junction_Control",
            [-1, 0, 1, 2, 3, 4]
        )

        urban_rural = select_code(
            "Urban or Rural Area",
            "Urban_or_Rural_Area",
            [1, 2, 3]
        )

    with col2:

        pedestrian_human = select_code(
            "Pedestrian Crossing - Human Control",
            "Pedestrian_Human",
            [0, 1, 2]
        )

        pedestrian_physical = select_code(
            "Pedestrian Crossing - Physical Facilities",
            "Pedestrian_Physical",
            [0, 1, 2, 3, 4, 5, 6, 7, 8]
        )

        special_conditions = st.selectbox(
            "Special Conditions at Site",
            [-1, 0, 1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: (
                "Unknown (Code -1)" if x == -1 else f"Code {x}"
            )
        )

        carriageway_hazards = st.selectbox(
            "Carriageway Hazards",
            [-1, 0, 1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: (
                "Unknown (Code -1)" if x == -1 else f"Code {x}"
            )
        )

        speed_limit = st.selectbox(
            "Speed Limit (mph)",
            [20, 30, 40, 50, 60, 70],
            index=1
        )

    st.divider()

    st.markdown(
        '<div class="section-heading">🌦️ Environmental Conditions</div>',
        unsafe_allow_html=True
    )

    col3, col4, col5 = st.columns(3)

    with col3:
        weather = select_code(
            "Weather Conditions",
            "Weather_Conditions",
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )

    with col4:
        light = select_code(
            "Light Conditions",
            "Light_Conditions",
            [-1, 1, 4, 5, 6, 7]
        )

    with col5:
        surface = select_code(
            "Road Surface Conditions",
            "Road_Surface_Conditions",
            [-1, 1, 2, 3, 4, 5]
        )

    st.divider()

    st.markdown(
        '<div class="section-heading">📅 Date and Time Information</div>',
        unsafe_allow_html=True
    )

    col6, col7, col8 = st.columns(3)

    with col6:
        month = st.selectbox(
            "Month",
            list(range(1, 13))
        )

    with col7:
        day = st.selectbox(
            "Day of Month",
            list(range(1, 32))
        )

    with col8:
        hour = st.selectbox(
            "Hour of Day",
            list(range(24)),
            index=14
        )

    day_of_week = select_code(
        "Day of Week",
        "Day_of_Week",
        list(range(7))
    )

    st.divider()

    submitted = st.form_submit_button(
        "🚀 Predict Accident Severity",
        use_container_width=True
    )


# ==========================================
# 7. PREDICTION
# ==========================================

if submitted:

    input_data = {
        "Police_Force": int(police_force),
        "1st_Road_Class": road_class_1,
        "2nd_Road_Class": road_class_2,
        "Road_Type": road_type,
        "Junction_Detail": junction_detail,
        "Junction_Control": junction_control,

        "Pedestrian_Crossing-Human_Control":
            pedestrian_human,

        "Pedestrian_Crossing-Physical_Facilities":
            pedestrian_physical,

        "Light_Conditions": light,
        "Weather_Conditions": weather,
        "Road_Surface_Conditions": surface,

        "Special_Conditions_at_Site": special_conditions,
        "Carriageway_Hazards": carriageway_hazards,
        "Urban_or_Rural_Area": urban_rural,
        "Day_of_Week": day_of_week,

        "Speed_limit": speed_limit,
        "Month": month,
        "Day": day,
        "Hour": hour
    }

    try:

        with st.spinner("Analysing accident conditions..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=input_data,
                timeout=15
            )

        if response.status_code == 200:

            result = response.json()

            severity = result["predicted_severity"]
            severity_code = result["predicted_severity_code"]

            colors = {
                "Fatal": "#DC2626",
                "Serious": "#D97706",
                "Slight": "#15803D"
            }

            color = colors.get(severity, "#2563EB")

            st.divider()

            st.markdown(
                '<div class="section-heading">📊 Prediction Result</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div style="
                background:white;
                padding:28px;
                border-radius:14px;
                border-left:7px solid {color};
                box-shadow:0 3px 12px rgba(0,0,0,0.06);
            ">
                <p style="color:#64748B; margin-bottom:5px;">
                    Predicted Accident Severity
                </p>
                <h1 style="color:{color}; margin:0;">
                    {severity.upper()}
                </h1>
                <p style="color:#64748B;">
                    Severity Code: {severity_code}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                "Prediction generated using the trained Random Forest model."
            )

        elif response.status_code == 422:

            st.error(f"Invalid input: {response.text}")

        else:

            st.error(
                f"API Error {response.status_code}: {response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI. "
            "Check whether the backend is running."
        )

    except requests.exceptions.RequestException as error:

        st.error(f"Request failed: {error}")

    except (KeyError, ValueError) as error:

        st.error(f"Unexpected API response: {error}")


# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.markdown("""
<div class="footer">
    <b>ROADSAFE AI</b> | Road Accident Severity Prediction
    <br><br>
    Machine Learning and MLOps Audit Course Project
    <br>
    Predictions represent learned associations, not causal factors.
</div>
""", unsafe_allow_html=True)
