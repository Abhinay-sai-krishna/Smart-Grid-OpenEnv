import random

class SmartGridEnv:
    def __init__(self):
        self.max_steps = 24
        self.battery_capacity = 50.0  # MWh
        self.max_charge_rate = 10.0  # MWh per step
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.battery_charge = 0.0
        self.total_profit = 0.0
        self._generate_state()
        return self.get_state()
        
    def _generate_state(self):
        # Time-based pricing and demand
        base_demand = 50.0 + 10 * random.random()
        base_price = 20.0
        
        # Peaks usually around hours 17-21
        if 17 <= self.current_step <= 21:
            base_demand += 40.0
            base_price += 100.0
            
        self.grid_demand = base_demand
        
        # Solar generation peaks at midday (hour 12)
        if 6 <= self.current_step <= 18:
            self.solar_generation = max(0.0, 30.0 * (1.0 - abs(self.current_step - 12) / 6.0))
        else:
            self.solar_generation = 0.0
            
        self.wind_generation = 10.0 + 15 * random.random()
        self.electricity_price = base_price + 5 * random.random()
        
    def get_state(self):
        return {
            "step_id": self.current_step,
            "grid_demand": round(self.grid_demand, 2),
            "solar_generation": round(self.solar_generation, 2),
            "wind_generation": round(self.wind_generation, 2),
            "electricity_price": round(self.electricity_price, 2),
            "battery_charge": round(self.battery_charge, 2),
            "battery_capacity": round(self.battery_capacity, 2),
            "total_profit": round(self.total_profit, 2)
        }
        
    def step(self, action_type: str, amount: float):
        reward = 0.0
        
        # Determine actual action amounts based on capacity 
        amount = max(0.0, min(amount, self.max_charge_rate))
        
        # Grid net demand before battery (positive means we need more external power)
        net_demand = self.grid_demand - (self.solar_generation + self.wind_generation)
        
        actual_action_taken = 0.0
        step_profit = 0.0
        if action_type == "charge":
            # Can't charge more than available capacity
            actual_action_taken = min(amount, self.battery_capacity - self.battery_charge)
            self.battery_charge += actual_action_taken
            # Cost of charging
            cost = actual_action_taken * self.electricity_price
            self.total_profit -= cost
            step_profit = -cost
            
            # Charging takes energy FROM the grid, increasing demand
            net_demand += actual_action_taken 
            
        elif action_type == "discharge":
            # Can't discharge more than battery holds
            actual_action_taken = min(amount, self.battery_charge)
            self.battery_charge -= actual_action_taken
            # Revenue from discharging
            revenue = actual_action_taken * self.electricity_price
            self.total_profit += revenue
            step_profit = revenue
            
            # Discharging provides energy TO the grid, reducing demand
            net_demand -= actual_action_taken 
            
        # Reward design:
        # Base reward: profit from trading this step
        trade_reward = (step_profit / 100.0)
        
        # Grid stability penalty: Heavily penalize high net demand from external grid
        stability_penalty = 0.0
        if net_demand > 0:
            stability_penalty = (net_demand * 0.5)
            
        reward = trade_reward - stability_penalty
            
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        self._generate_state()
            
        return self.get_state(), round(reward, 2), done, {
            "net_demand": round(net_demand, 2),
            "actual_action_taken": round(actual_action_taken, 2)
        }
