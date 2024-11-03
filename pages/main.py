import streamlit as st
import plotly.express as px
import pandas as pd

senate_race_states = ['Arizona', 'California', 'Connecticut', 'Delaware', 'Florida', 
                      'Hawaii', 'Indiana', 'Maine', 'Maryland', 'Massachusetts', 
                      'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 
                      'Montana', 'Nebraska', 'Nevada', 'New Jersey', 'New Mexico', 
                      'New York', 'North Dakota', 'Ohio', 'Pennsylvania', 
                      'Rhode Island', 'Tennessee', 'Texas', 'Utah', 'Vermont', 
                      'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

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

state_data['senate_race'] = state_data['state'].apply(lambda x: '2024 Race' if x in senate_race_states else 'No Race')

st.title("Interactive U.S. Map with 2024 Senate Races Highlighted")

fig = px.choropleth(
    state_data,
    locations="abbrev",
    locationmode="USA-states",
    scope="usa",
    color="senate_race",  # Highlight states with Senate races
    color_discrete_map={'2024 Race': '#FF6347', 'No Race': '#B0C4DE'},  # Colors: red for races, light blue otherwise
    hover_name="state",
    title="2024 Senate Races by State"
)

st.plotly_chart(fig)

st.write("### Select a State by Abbreviation")
clicked_state = st.selectbox("Select a state:", state_data['state'])

if clicked_state:
    abbrev = state_data.loc[state_data['state'] == clicked_state, 'abbrev'].values[0].lower()
    st.write(f"Click [here](/{abbrev}) to go to the {clicked_state} page.")
