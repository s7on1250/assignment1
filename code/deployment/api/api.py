from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint
from typing import Optional
from code.models.inference import inference

app = FastAPI()


class PassengerData(BaseModel):
    gender: str
    customer_type: str
    age: conint(ge=0)
    type_of_travel: str
    travel_class: str
    flight_distance: conint(ge=0)
    inflight_wifi_service: conint(ge=0, le=5)
    departure_arrival_time_convenient: conint(ge=0, le=5)
    ease_of_online_booking: conint(ge=0, le=5)
    gate_location: conint(ge=1, le=5)
    food_and_drink: conint(ge=0, le=5)
    online_boarding: conint(ge=0, le=5)
    seat_comfort: conint(ge=0, le=5)
    inflight_entertainment: conint(ge=0, le=5)
    on_board_service: conint(ge=0, le=5)
    leg_room_service: conint(ge=0, le=5)
    baggage_handling: conint(ge=0, le=5)
    checkin_service: conint(ge=0, le=5)
    inflight_service: conint(ge=0, le=5)
    cleanliness: conint(ge=0, le=5)
    departure_delay_in_minutes: conint(ge=0)
    arrival_delay_in_minutes: Optional[conint(ge=0)]


@app.post("/predict_satisfaction/")
def predict_satisfaction(data: PassengerData):
    passenger_info = data.dict()
    print(passenger_info)
    satisfaction = inference(passenger_info)
    return {"prediction": satisfaction}
