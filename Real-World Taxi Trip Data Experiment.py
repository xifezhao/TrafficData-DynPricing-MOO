import pandas as pd
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

# Load the datasets
try:
    df_original = pd.read_csv("original_cleaned_nyc_taxi_data_2018.csv")
    df_trip = pd.read_csv("taxi_trip_data.csv")
    df_zones = pd.read_csv("taxi_zone_geo.csv")
except FileNotFoundError:
    print("Error: One or more CSV files not found. Please ensure they are in the same directory.")
    exit()

# --- 1. System Architecture Components ---

# --- Data Valuation ---
def calculate_temporal_value(base_value, demand_factor, urgency_factor, alpha=0.5, beta=0.3):
    """Calculates the temporal value of the data."""
    return base_value * (1 + alpha * demand_factor + beta * urgency_factor)

def calculate_spatial_value(congestion_level, incident_rate, gamma=0.7, delta=0.5):
    """Calculates the spatial value of the data."""
    return 1 + gamma * congestion_level + delta * incident_rate

def calculate_quality_factor(completeness=1.0, accuracy=1.0, reliability=1.0, w_completeness=0.3, w_accuracy=0.4, w_reliability=0.3):
    """Calculates the quality factor of the data."""
    return (w_completeness * completeness + w_accuracy * accuracy + w_reliability * reliability)

def evaluate_data_value(base_value, demand_factor, urgency_factor, congestion_level, incident_rate, quality_factor):
    """Evaluates the overall value of the data based on spatiotemporal attributes and quality."""
    Vt = calculate_temporal_value(base_value, demand_factor, urgency_factor)
    Vs = calculate_spatial_value(congestion_level, incident_rate)
    return Vt * Vs * quality_factor

# --- Demand Prediction ---
# In a real-world scenario, this would involve training a deep learning model (like LSTM).
# For simplicity, we use a placeholder function here.
def predict_demand(time_of_day, location_id, historical_demand_patterns=None):
    """Predicts the demand for taxi data based on time and location."""
    # Placeholder logic: Higher demand during peak hours and in busy locations
    peak_hours = [8, 9, 17, 18]  # Example peak hours
    busy_locations = df_zones['zone_id'].sample(frac=0.2).tolist() # Example busy locations

    demand = 1.0
    if time_of_day in peak_hours:
        demand += 0.5
    if location_id in busy_locations:
        demand += 0.7

    # Incorporate historical patterns (if available) - This is a very basic example
    if historical_demand_patterns and (time_of_day, location_id) in historical_demand_patterns:
        demand += historical_demand_patterns[(time_of_day, location_id)]

    return demand

# --- Multi-Objective Optimization ---
class DynamicPricingProblem(Problem):
    def __init__(self, demand_prediction_func, data_quality_func, **kwargs):
        super().__init__(n_var=1, n_obj=4, n_constr=0, xl=0.1, xu=100.0, vectorized=True, **kwargs) # Price range 0.1 to 100
        self.demand_prediction_func = demand_prediction_func
        self.data_quality_func = data_quality_func

    def _evaluate(self, x, out, *args, **kwargs):
        prices = x[:, 0]
        n_pop = len(prices)
        # Assume a specific time and location for this optimization step
        time_of_day = 10  # Example hour
        location_id = 138 # Example location ID

        demands = np.array([self.demand_prediction_func(time_of_day, location_id) for _ in range(n_pop)])
        revenues = prices * demands
        user_costs = prices
        data_quality = np.full(n_pop, self.data_quality_func()) # Same quality for all individuals
        market_fairness = -np.var(prices) # Minimize price variance

        out["F"] = np.column_stack([-revenues, user_costs, -data_quality, np.full(n_pop, market_fairness)])

# --- Dynamic Pricing ---
def determine_price(optimization_results):
    """Determines the price based on the Pareto front from optimization."""
    # Simple strategy: Choose a solution that balances revenue and user cost
    # This can be more sophisticated based on business goals
    if optimization_results.F is not None:
        pareto_front = pd.DataFrame(optimization_results.F, columns=["-revenue", "user_cost", "-data_quality", "-market_fairness"])
        pareto_front["revenue"] = -pareto_front["-revenue"]
        # Sort by revenue (descending) and user cost (ascending)
        pareto_front_sorted = pareto_front.sort_values(by=["revenue", "user_cost"], ascending=[False, True])
        best_solution = pareto_front_sorted.iloc[0]
        return best_solution["user_cost"]
    return None

# --- Real-time Feedback Mechanism ---
def collect_feedback(price, actual_demand, user_satisfaction):
    """Collects feedback from users and the market."""
    # In a real system, this would involve storing and processing feedback data
    print(f"Collected feedback: Price={price}, Actual Demand={actual_demand}, Satisfaction={user_satisfaction}")
    return {"price": price, "actual_demand": actual_demand, "user_satisfaction": user_satisfaction}

def update_models(feedback_data):
    """Updates the demand prediction and valuation models based on feedback."""
    # This is a placeholder for model retraining or parameter adjustments
    print("Models updated based on feedback.")
    # In a real system, you would use the feedback data to:
    # 1. Retrain the demand prediction model with new data points.
    # 2. Adjust weights in the valuation model based on user behavior.
    pass

# --- 2. Data Characteristics and Value Assessment ---
def assess_data_value_for_trip(trip_data_row):
    """Assess the data value for a specific taxi trip."""
    base_value = trip_data_row['calculated_total_amount'] if 'calculated_total_amount' in trip_data_row else trip_data_row['fare_amount']
    demand_factor = predict_demand(trip_data_row['hour_of_day'], trip_data_row.get('pickup_location_id', 1)) # Using get to handle potential missing column
    urgency_factor = 0.8 if trip_data_row['hour_of_day'] in [7, 8, 17, 18] else 0.2 # Example
    congestion_level = 0.7 # Placeholder, needs actual calculation
    incident_rate = 0.1 # Placeholder, needs actual calculation
    quality_factor = calculate_quality_factor() # Placeholder, needs actual metrics

    return evaluate_data_value(base_value, demand_factor, urgency_factor, congestion_level, incident_rate, quality_factor)

# --- 3. Demand Prediction Model ---
# (The predict_demand function above serves as the basic demand prediction model)

# --- 4. Dynamic Price Optimization ---
def optimize_price():
    """Optimizes the price using the NSGA-II algorithm."""
    problem = DynamicPricingProblem(predict_demand, calculate_quality_factor)
    algorithm = NSGA2(pop_size=100,
                      sampling=FloatRandomSampling(),
                      crossover=SBX(prob=0.9, eta=15),
                      mutation=PM(eta=20),
                      eliminate_duplicates=True)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 10),
                   seed=42,
                   verbose=False)
    return res

# --- 5. Real-time Feedback Mechanism ---
def simulate_real_time_pricing():
    """Simulates the dynamic pricing process with feedback."""
    # Initial optimization
    optimization_results = optimize_price()
    suggested_price = determine_price(optimization_results)

    if suggested_price is not None:
        print(f"Suggested Price: ${suggested_price:.2f}")

        # Simulate user interaction and collect feedback
        actual_demand = predict_demand(10, 138) * (1 - (suggested_price - 5)/20 if suggested_price > 5 else 1) # Example demand change based on price
        user_satisfaction = 0.7 if suggested_price < 15 else 0.4 # Example satisfaction based on price
        feedback = collect_feedback(suggested_price, actual_demand, user_satisfaction)

        # Update models based on feedback
        update_models(feedback)
    else:
        print("Could not determine a price.")

# --- Main Execution ---
if __name__ == "__main__":
    print("Assessing data value for a sample trip:")
    sample_trip = df_original.iloc[0]
    data_value = assess_data_value_for_trip(sample_trip)
    print(f"Data value for the sample trip: {data_value:.2f}")

    print("\nSimulating dynamic real-time pricing:")
    simulate_real_time_pricing()