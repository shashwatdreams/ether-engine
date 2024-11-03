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

# Title for the app
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
clicked_state = st.text_input("Enter state abbreviation (e.g., AL for Alabama):")

# Check and Navigate
if clicked_state:
    link = state_data.loc[state_data['abbrev'] == clicked_state.upper(), 'link'].values
    if link:
        st.write(f"Click [here to navigate to {clicked_state.upper()}](#)")
        # Placeholder link. In a full app, replace with navigation functionality as Streamlit supports multipage apps.
    else:
        st.write("State not found. Please enter a valid abbreviation.")

