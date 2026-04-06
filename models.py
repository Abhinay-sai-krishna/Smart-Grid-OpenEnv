from pydantic import BaseModel
from typing import Literal, Dict, Any

class ActionSchema(BaseModel):
    action_type: Literal["charge", "discharge", "hold"]
    amount: float

class StateSchema(BaseModel):
    step_id: int
    grid_demand: float
    solar_generation: float
    wind_generation: float
    electricity_price: float
    battery_charge: float
    battery_capacity: float
    total_profit: float

class StepResponse(BaseModel):
    obs: StateSchema
    reward: float
    done: bool
    info: Dict[str, Any]
