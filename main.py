import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score
from create_dataset import get_processed_datasets
import numpy as np
# https://www.sciencedirect.com/science/article/abs/pii/S0191261507001300
# 
# Function to predict traffic volume categories for each hour of the day for a specific day
def predict_traffic_for_day(day_of_week, test_data, inference):
    hours = list(range(24))  # Hours from 0 to 23 (24-hour format)
    predicted_traffic = []
    
    # Predict traffic volume for each hour of the day
    for hour in hours:
        evidence = {'DayOfWeek': day_of_week, 'Hour': hour}
        try:
            predict = inference.map_query(variables=['TrafficVolumeCategory'], evidence=evidence)
            predicted_traffic.append(predict['TrafficVolumeCategory'])
        except Exception as e:
            print(f"Error predicting for {day_of_week} at {hour}: {e}")
            predicted_traffic.append(np.nan)

    return hours, predicted_traffic

if __name__ == "__main__":
    # Get the processed datasets
    train_data, test_data = get_processed_datasets()

    # Define  Bayesian Network
    model = BayesianNetwork([('DayOfWeek', 'TrafficVolumeCategory'), 
                             ('Hour', 'TrafficVolumeCategory')])

    # Fit the Bayesian Network using Bayesian Estimator on training data
    model.fit(train_data, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

    # Print the learned Conditional Probability Distribution
    print("\nLearned CPDs:")
    for cpd in model.get_cpds():
        print(cpd)

    # Create an inference object
    inference = VariableElimination(model)

    # Days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Create subplots for all seven days in a 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    fig.tight_layout(pad=5.0)

    axes = axes.flatten()

    # Loop through each day and plot predicted vs actual traffic
    for idx, day_of_week in enumerate(days_of_week):
        # Predict traffic for day in the test dataset
        hours, predicted_traffic = predict_traffic_for_day(day_of_week, test_data, inference)

        # Extract actual traffic volumes for comparison (for the specified day)
        actual_data = test_data[test_data['DayOfWeek'] == day_of_week].groupby('Hour')['TrafficVolumeCategory'].agg(lambda x: x.mode()[0]).reset_index()
        actual_traffic = actual_data['TrafficVolumeCategory'].tolist()

        # Check if actual traffic data is available for the day
        if len(actual_traffic) == 0:
            print(f"No actual traffic data available for {day_of_week} in the test dataset.")
            continue

        # Compare 
        accuracy = accuracy_score(actual_traffic[:len(predicted_traffic)], predicted_traffic[:len(actual_traffic)])
        print(f'\nModel Accuracy for {day_of_week} (2019-2020 data): {accuracy * 100:.2f}%')

        # Plot 
        axes[idx].plot(hours[:len(actual_traffic)], actual_traffic, marker='o', label='Actual Traffic Volume')
        axes[idx].plot(hours[:len(predicted_traffic)], predicted_traffic[:len(actual_traffic)], linestyle='dashed', marker='x', label='Predicted Traffic Volume')
        axes[idx].set_xticks(hours)  # Set x-ticks as hours (0-23)
        axes[idx].set_xlabel('Time (Hour of Day)')
        axes[idx].set_ylabel('Traffic Volume Category')
        axes[idx].set_title(f'{day_of_week}: Accuracy: {accuracy * 100:.2f}%')

    axes[0].legend()

    # Hide the 8th subplot (extra subplot)
    if len(axes) > 7:
        fig.delaxes(axes[7])

    # Show the plot with all seven days in a 4x2 grid
    plt.show()
