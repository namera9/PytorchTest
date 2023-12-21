import torch
import torch.nn as nn
import numpy as np

# Define the BinaryClassificationModel class
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassificationModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Load the saved model
loaded_model = BinaryClassificationModel(input_size=2)
loaded_model.load_state_dict(torch.load('binaryclassification.pt'))
loaded_model.eval()

# Function to score (make predictions) on new data
def score_model(new_data):
    new_data_tensor = torch.tensor(new_data, dtype=torch.float32)
    with torch.no_grad():
        predictions = loaded_model(new_data_tensor)
        predicted_labels = (predictions >= 0.5).float().view(-1).numpy()
    return predicted_labels

# Generate random new data
new_data = np.random.rand(10, 2)

# Score the model on the new data
predictions = score_model(new_data)

# Print the predictions
print("Predictions:", predictions)
