import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch.nn as nn

# Change directory to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Read the dataset
dataset_path = os.path.join(script_dir, 'dataset')
subpaths = os.listdir(dataset_path)
df = pd.read_csv(os.path.join(dataset_path, subpaths[0]))

# preprocess
# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode the diseases using BERT tokenizer
diseases = df['Disease'].tolist()
encoded_diseases = [tokenizer.encode(disease, add_special_tokens=True) for disease in diseases]

# Convert discrete features to binary values
discrete_features = ['Cough', 'Fatigue', 'Fever', 'Difficulty Breathing', 'Outcome Variable', 'Gender', 'Blood Pressure', 'Cholesterol Level']
for feature in discrete_features:
    if feature == 'Blood Pressure':
        df[feature] = df[feature].map({'Low': 0, 'Normal': 0.5, 'High': 1})
    elif feature == 'Cholesterol Level':
        df[feature] = df[feature].map({'Low': 0, 'Normal': 0.5, 'High': 1})
    elif feature == 'Outcome Variable':
        df[feature] = df[feature].map({'Positive': 1, 'Negative': 0})
    else:
        df[feature] = df[feature].map({'Yes': 1, 'No': 0}) if feature != 'Gender' else df[feature].map({'Male': 1, 'Female': 0})

# Normalize the 'Age' using MinMaxScaler
scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1))

# Find the disease vector with the most dimensions
max_dimensions = max(len(disease) for disease in encoded_diseases)
# Pad 0s at the right of the vectors to match the max dimensions
df['Disease'] = [disease + [0] * (max_dimensions - len(disease)) for disease in encoded_diseases]
normalized_diseases = torch.tensor(df['Disease'].tolist()).float()
df['Disease'] = normalized_diseases.div(torch.norm(normalized_diseases, dim=1, keepdim=True)).tolist()

# Construct the model
class DiseaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DiseaseModel, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(hidden_size + len(discrete_features), output_size)

    def forward(self, x, other_features):
        x = self.linear(x)
        x = self.relu(x)
        x = torch.cat((x, other_features), dim=1)
        x = self.mlp(x)
        return x

input_size = max_dimensions
hidden_size = 64
output_size = 2

model = DiseaseModel(input_size, hidden_size, output_size)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 8
for epoch in range(num_epochs):
    # Split the dataset into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Prepare the input tensors
    train_diseases = torch.tensor(train_df['Disease'].tolist()).float()
    train_other_features = torch.tensor(train_df[discrete_features].values).float()
    train_labels = torch.tensor(train_df['Outcome Variable'].values).long()

    test_diseases = torch.tensor(test_df['Disease'].tolist()).float()
    test_other_features = torch.tensor(test_df[discrete_features].values).float()
    test_labels = torch.tensor(test_df['Outcome Variable'].values).long()

    # Forward pass
    train_outputs = model(train_diseases, train_other_features)
    test_outputs = model(test_diseases, test_other_features)

    # Compute loss
    train_loss = criterion(train_outputs, train_labels)
    test_loss = criterion(test_outputs, test_labels)

    # Backward and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Print the loss for each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    train_outputs = model(train_diseases, train_other_features)
    train_predicted_labels = torch.argmax(train_outputs, dim=1)
    train_accuracy = accuracy_score(train_labels, train_predicted_labels)

    test_outputs = model(test_diseases, test_other_features)
    test_predicted_labels = torch.argmax(test_outputs, dim=1)
    test_accuracy = accuracy_score(test_labels, test_predicted_labels)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')