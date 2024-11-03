import streamlit as st
import plotly.express as px
import pandas as pd

# Sample Data with State Abbreviations and Links
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
               'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'],
    # Placeholder links for each state
    'link': [f'/state/{abbr.lower()}' for abbr in ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 
             'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 
             'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 
             'VA', 'WA', 'WV', 'WI', 'WY']]
})

st.title("Interactive U.S. Map with Clickable States")

# Map visualization using Plotly
fig = px.choropleth(
    state_data,
    locations="abbrev",
    locationmode="USA-states",
    scope="usa",
    title="Click on a state to navigate",
    hover_name="state",
    color_discrete_sequence=["#636EFA"],  # Customize color if needed
)

# Display the map in Streamlit
st.plotly_chart(fig)

# User Input for State Selection
st.write("### Select a State by Abbreviation")
clicked_state = st.selectbox("Select a state:", state_data['state'])

# Check and Navigate
if clicked_state:
    # Get the abbreviation for the selected state
    abbrev = state_data.loc[state_data['state'] == clicked_state, 'abbrev'].values[0].lower()
    st.experimental_set_query_params(state=abbrev)

    # Write a link to navigate
    st.write(f"Click [here](/{abbrev}) to go to {clicked_state}'s page.")

    # Redirect to the state page (in a Streamlit multipage setup)
    st.write(f"Redirecting to the {clicked_state} page...")
    st.experimental_rerun()