# SmartGridOptimizer - OpenEnv

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kondapalliabhinaysaikrishna/Smart-Grid-OpenEnv)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Abhinay-sai-krishna/Smart-Grid-OpenEnv)
A reinforcement learning environment strictly compliant with the OpenEnv specification. Simulate a real-world smart grid network running a 24-hour battery management cycle.

## Environment Description
The environment challenges an AI agent to handle the charging and discharging lifecycle of a 50 MWh commercial battery interacting with a volatile power grid.
The agent must intelligently balance external grid demand parameters, spot market electricity pricing, and variable clean generation inputs (Solar/Wind) to maximize profitability while maintaining net grid stability.

## Action Space
The agent responds with a JSON payload defining its step action. 
* `action_type` [string]: Specifies the behavior mode. Permitted operations are `charge`, `discharge`, and `hold`.
* `amount` [float]: The rate/amount of energy in MWh to transfer. The hardware constraint limit per hour is bounds mapped physically up to `10.0 MWh`. 

## Observation Space 
At each step, the standard `/step` endpoint yields a continuous state snapshot representing contextual parameters and battery capacities:

* `step_id` [int]: Current hour metric (0-24 bounds).
* `grid_demand` [float]: Represents dynamic demand patterns.
* `solar_generation` [float]: Base energy generated exclusively via daytime bounds. 
* `wind_generation` [float]: Stochastic continuous stream of auxiliary energy.
* `electricity_price` [float]: Market rate pricing volatility mapping. 
* `battery_charge` [float]: The preserved available power inside the system limits (MWh).
* `battery_capacity` [float]: Absolute top-line physical holding capacity limit metrics (50 MWh default).
* `total_profit` [float]: Running marginal PnL tracking score. 

## Setup Instructions

**1. Clone the repository and install requirements**
```bash
pip install -r requirements.txt
```

**2. Start the Server Container locally**
```bash
uvicorn main:app --port 8000
```
*(Alternatively classically containerized internally via Docker through standard HF Space ports configured via the included Dockerfile).*

**3. Run the Inference LLM script**
Ensure the baseline environment variables are applied:
```bash
export API_BASE_URL="<open_url>"
export MODEL_NAME="<llm_name>"
export HF_TOKEN="<token>"
python inference.py
```
