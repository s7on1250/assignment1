import streamlit as st
import requests

st.title("Airline Passenger Satisfaction Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
age = st.number_input("Age", min_value=0, max_value=120, step=1)
type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
flight_distance = st.number_input("Flight Distance", min_value=0, step=1)

inflight_wifi_service = st.slider("Inflight Wifi Service", 0, 5)
departure_arrival_time_convenient = st.slider("Departure/Arrival Time Convenient", 0, 5)
ease_of_online_booking = st.slider("Ease of Online Booking", 0, 5)
gate_location = st.slider("Gate Location", 1, 5)
food_and_drink = st.slider("Food and Drink", 0, 5)
online_boarding = st.slider("Online Boarding", 0, 5)
seat_comfort = st.slider("Seat Comfort", 0, 5)
inflight_entertainment = st.slider("Inflight Entertainment", 0, 5)
on_board_service = st.slider("On-board Service", 0, 5)
leg_room_service = st.slider("Leg Room Service", 0, 5)
baggage_handling = st.slider("Baggage Handling", 0, 5)
checkin_service = st.slider("Checkin Service", 0, 5)
inflight_service = st.slider("Inflight Service", 0, 5)
cleanliness = st.slider("Cleanliness", 0, 5)

departure_delay_in_minutes = st.number_input("Departure Delay in Minutes", min_value=0, step=1)
arrival_delay_in_minutes = st.number_input("Arrival Delay in Minutes (Optional)", min_value=0, step=1, value=0)


url = "http://127.0.0.1:8000/predict_satisfaction/"

# Collect inputs into a dictionary
passenger_data = {
    "gender": gender,
    "customer_type": customer_type,
    "age": age,
    "type_of_travel": type_of_travel,
    "travel_class": travel_class,
    "flight_distance": flight_distance,
    "inflight_wifi_service": inflight_wifi_service,
    "departure_arrival_time_convenient": departure_arrival_time_convenient,
    "ease_of_online_booking": ease_of_online_booking,
    "gate_location": gate_location,
    "food_and_drink": food_and_drink,
    "online_boarding": online_boarding,
    "seat_comfort": seat_comfort,
    "inflight_entertainment": inflight_entertainment,
    "on_board_service": on_board_service,
    "leg_room_service": leg_room_service,
    "baggage_handling": baggage_handling,
    "checkin_service": checkin_service,
    "inflight_service": inflight_service,
    "cleanliness": cleanliness,
    "departure_delay_in_minutes": departure_delay_in_minutes,
    "arrival_delay_in_minutes": arrival_delay_in_minutes,
}

if st.button("Predict Satisfaction"):

    response = requests.post(url, json=passenger_data)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
    else:
        st.error(f"Error: {response.text}")
