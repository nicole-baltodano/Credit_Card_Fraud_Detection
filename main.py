from data_loader import load_data, split_data
from class_balancer import rebalance_classes
from model import init_model, train_model
from evaluation import plot_metrics, plot_confusion_matrix, evaluate_model

# Load and split the data
file_path = "creditcard.csv"
X, y = load_data(file_path)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Rebalance the classes
X_train_resampled, y_train_resampled = rebalance_classes(X_train, y_train)

# Initialize and train the model
model = init_model(X_train_resampled.shape[1:],X_train_resampled)
history = train_model(model, X_train_resampled, y_train_resampled, X_val, y_val)

# Plot metrics
plot_metrics(history)

# Evaluate the model
y_pred_binary = evaluate_model(model, X_test, y_test)

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred_binary)
