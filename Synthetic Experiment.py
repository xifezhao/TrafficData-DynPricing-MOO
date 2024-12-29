import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from deap import base, creator, tools, algorithms
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 如果在非交互式环境中不想弹出窗口，可以使用以下后端（可选）
# matplotlib.use('Agg')


def run_dynamic_pricing_experiment():
    """
    运行动态交通数据定价实验，并生成可视化图表。

    Returns:
        pandas.DataFrame: 包含动态定价结果的 traffic_data DataFrame。
    """

    # 1. Data Acquisition (Using a Synthetic Dataset)
    np.random.seed(42)
    num_points = 1000
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=num_points, freq='H'),
        'latitude': np.random.uniform(34.0, 34.1, num_points),
        'longitude': np.random.uniform(-118.3, -118.2, num_points),
        'speed': np.random.uniform(20, 60, num_points),
        'volume': np.random.randint(50, 500, num_points),
        'incidents': np.random.randint(0, 3, num_points)
    })
    traffic_data = data.copy()

    # 2. Data Valuation (Implementing the Spatiotemporal Value Model)
    def temporal_value(timestamp, base_value=1, alpha=0.5, demand_factor=0, beta=0.3, urgency_factor=0):
        """Calculates the temporal value based on the formula."""
        return base_value * (1 + alpha * demand_factor + beta * urgency_factor)

    def spatial_value(latitude, longitude, congestion_level=0, incident_rate=0, gamma=0.7, delta=0.5):
        """Calculates the spatial value based on the formula."""
        return 1 + gamma * congestion_level + delta * incident_rate

    def quality_factor_calculation(speed, volume):
        """A simplified quality factor based on data validity."""
        if speed > 0 and volume >= 0:
            return 0.7 + 0.3 * min(1, volume / 500)
        else:
            return 0.5

    def calculate_data_value(row, demand_factor):
        """Calculates the overall data value."""
        congestion_level = 1 - row['speed'] / 60
        incident_rate = row['incidents'] / 5

        vt = temporal_value(row['timestamp'], demand_factor=demand_factor, 
                            urgency_factor=(row['timestamp'].hour in [8, 17]))
        vs = spatial_value(row['latitude'], row['longitude'],
                           congestion_level=congestion_level, incident_rate=incident_rate)
        quality = quality_factor_calculation(row['speed'], row['volume'])
        return vt * vs * quality

    # 3. Demand Prediction (Simple Linear Regression Model)
    traffic_data['hour'] = traffic_data['timestamp'].dt.hour
    traffic_data['dayofweek'] = traffic_data['timestamp'].dt.dayofweek

    demand_feature = 'volume'
    model_features = ['hour', 'dayofweek']

    X = traffic_data[model_features]
    y = traffic_data[demand_feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    demand_model = LinearRegression()
    demand_model.fit(X_train, y_train)

    def predict_demand(timestamp):
        """Predicts demand based on the trained model."""
        features = pd.DataFrame({'hour': [timestamp.hour], 'dayofweek': [timestamp.dayofweek]})
        return max(0, demand_model.predict(features)[0])

    # 4. Multi-Objective Optimization (Using DEAP)
    # 注意：在重复执行脚本时，如果已创建过 creator 类，需要先重置；在实际项目中最好写在全局只创建一次
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0, -1.0)) # (revenue, -cost, quality, -price_variance)
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_price_factor", random.uniform, 0.5, 2.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_price_factor, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual, data_point):
        """Evaluates an individual (price factor) based on the objectives."""
        price_factor = individual[0]
        base_value = 1

        predicted_demand_value = predict_demand(data_point['timestamp'])
        demand_factor = (
            predicted_demand_value / traffic_data[demand_feature].max() 
            if traffic_data[demand_feature].max() > 0 else 0
        )

        data_value = calculate_data_value(data_point, demand_factor)
        price = base_value * price_factor * data_value

        revenue = price * predicted_demand_value
        user_cost = price
        quality = quality_factor_calculation(data_point['speed'], data_point['volume'])
        fairness_penalty = abs(price_factor - 1.0)

        return revenue, user_cost, quality, fairness_penalty

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    def run_optimization(data_point):
        """Runs the optimization algorithm for a given data point."""
        # 这里每次都注册 evaluate，主要是为了传入 data_point，不建议在大规模场景反复注册
        toolbox.register("evaluate", evaluate, data_point=data_point) 
        population = toolbox.population(n=50)
        HallOfFame = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10,
                            stats=stats, halloffame=HallOfFame, verbose=False)
        return HallOfFame[0][0]

    # 5. Dynamic Pricing (Applying the Optimized Price)
    optimized_prices = []
    for index, row in traffic_data.iterrows():
        best_price_factor = run_optimization(row)
        base_value = 1
        demand_factor = (
            predict_demand(row['timestamp']) / traffic_data[demand_feature].max()
            if traffic_data[demand_feature].max() > 0 else 0
        )
        data_value = calculate_data_value(row, demand_factor)
        optimized_price = base_value * best_price_factor * data_value
        optimized_prices.append(optimized_price)

    traffic_data['dynamic_price'] = optimized_prices

    return traffic_data


def visualize_dynamic_pricing(traffic_data):
    """
    生成动态定价结果的可视化图表。

    Args:
        traffic_data (pd.DataFrame): 包含时间戳、速度、流量和动态价格的 DataFrame。
    """

    # 确保 timestamp 列是 datetime 类型
    traffic_data['timestamp'] = pd.to_datetime(traffic_data['timestamp'])

    # 提取日期和小时
    traffic_data['date'] = traffic_data['timestamp'].dt.date
    traffic_data['hour'] = traffic_data['timestamp'].dt.hour

    # 按日期和小时聚合数据
    daily_hourly_data = traffic_data.groupby(['date', 'hour']).agg(
        mean_price=('dynamic_price', 'mean'),
        mean_speed=('speed', 'mean'),
        mean_volume=('volume', 'mean')
    ).reset_index()

    # 获取所有唯一的日期
    unique_dates = daily_hourly_data['date'].unique()

    # 设置子图的数量和排列，以适应不同数量的日期
    num_dates = len(unique_dates)
    rows = int(num_dates ** 0.5) + (1 if num_dates % int(num_dates ** 0.5) != 0 else 0)
    cols = int(num_dates ** 0.5) + (1 if num_dates % int(num_dates ** 0.5) != 0 else 0)

    # 创建子图并绘制每个日期的数据
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)
    axes = axes.flatten()  # 将 axes 数组展平以便于索引

    for i, current_date in enumerate(unique_dates):
        date_data = daily_hourly_data[daily_hourly_data['date'] == current_date]
        ax = axes[i]

        # 绘制动态价格、归一化速度和归一化流量
        ax.plot(date_data['hour'], date_data['mean_price'], label='Dynamic Price', color='blue')
        ax.plot(
            date_data['hour'], 
            date_data['mean_speed'] / traffic_data['speed'].max(), 
            label='Normalized Speed', color='green', linestyle='--'
        )
        ax.plot(
            date_data['hour'], 
            date_data['mean_volume'] / traffic_data['volume'].max(), 
            label='Normalized Volume', color='red', linestyle=':'
        )

        ax.set_title(f'{current_date}')
        #ax.set_xlabel('Hour of the Day')
        #ax.set_ylabel('Value (Normalized)')
        # 去掉图例
        # ax.legend()
        ax.grid(True)

    # 移除多余的空白子图
    if num_dates < rows * cols:
        for j in range(num_dates, rows * cols):
            fig.delaxes(axes[j])

    plt.tight_layout()

    # 不显示图表，直接保存为 PDF
    plt.savefig('dynamic_pricing.pdf', format='pdf')
    plt.close(fig)


if __name__ == "__main__":
    traffic_data = run_dynamic_pricing_experiment()
    print(traffic_data[['timestamp', 'speed', 'volume', 'dynamic_price']].head())
    visualize_dynamic_pricing(traffic_data)