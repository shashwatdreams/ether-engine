import streamlit as st
import plotly.express as px
import pandas as pd

state_data = pd.DataFrame({
    'state': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
              'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
              'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
              'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina',
              'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
              'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia',
              'Wisconsin', 'Wyoming'],
    'abbrev': ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY',
               'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',
               'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
})

st.title("Interactive U.S. Map with Clickable States")

fig = px.choropleth(
    state_data,
    locations="abbrev",
    locationmode="USA-states",
    scope="usa",
    title="Click on a state to navigate",
    hover_name="state",
    color_discrete_sequence=["#636EFA"],  # Customize color if needed
)

st.plotly_chart(fig)

st.write("### Select a State by Abbreviation")
clicked_state = st.selectbox("Select a state:", state_data['state'])

if clicked_state:
    abbrev = state_data.loc[state_data['state'] == clicked_state, 'abbrev'].values[0].lower()
    st.write(f"Click [here](/{abbrev}) to go to the {clicked_state} page.")
