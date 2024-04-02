import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader


# ##### PREPARING THE DATA ##### #
# merge data
df1 = pd.read_csv('product_descriptions.csv')
df2 = pd.read_csv('train.csv')
merged_df = pd.merge(df2, df1, on='product_uid')
merged_df.to_csv('merged_train_desc.csv', index=False)

# Load in the dataset, combining the train dataset and the product descriptions
file = 'merged_train_desc.csv'
df_full = pd.read_csv(file)
print("Dataset loaded")

# TF-IDF vectorization
df_full['combined_string'] = (df_full['product_uid'].astype(str)
                              + " " + df_full['product_description'].astype(str)
                              + " " + df_full['search_term'].astype(str))
vectorizer = TfidfVectorizer(max_features=1000)  # lower max_features if you encounter memory issues
tfidf_matrix = vectorizer.fit_transform(df_full['combined_string'])
print("TF-IDF matrix built")

# Converting to pytorch tensors
tfidf_array = tfidf_matrix.toarray()
relev_array = np.array(df_full['relevance'])
tfidf_tensor = torch.FloatTensor(tfidf_array)
relev_tensor = torch.FloatTensor(relev_array)
print("Initial tensors set up")

tensorset = TensorDataset(tfidf_tensor, relev_tensor)
dataloader = DataLoader(tensorset, batch_size=32, shuffle=True)
print("Tensors built")


# ##### BUILD NEURAL NETWORK ##### #
collabModel = nn.Sequential(
    nn.Linear(1000, 512),  # input to hidden layer
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
)
print("Model set up")

# ##### TRAIN FEATURE EXTRACTOR ##### #
# build loss model and pytorch optimizer, split data
epochs = 25
learning_rate = 0.05
test_size = 0.2
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(collabModel.parameters(), lr=learning_rate)

train_data, test_data, train_goal, test_goal = train_test_split(tfidf_tensor, relev_tensor, test_size=test_size)
train_tensor = TensorDataset(train_data, train_goal)
test_tensor = TensorDataset(test_data, test_goal)
train_load = DataLoader(train_tensor, batch_size=16, shuffle=True)
test_load = DataLoader(test_tensor, batch_size=16)
print("Beginning training and testing...")

last_loss_trn = 0
last_loss_tst = 0

# run training
for epoch in range(epochs):
    collabModel.train()
    total_loss = 0

    for data, goal in train_load:
        predicted = collabModel(data)
        loss = loss_func(predicted.squeeze(), goal)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    av_loss = total_loss / len(train_load)
    print(f"Epoch {epoch+1} loss: {av_loss}")
    print(f"Delta loss: {last_loss_trn - av_loss}")
    last_loss_trn = av_loss

    with torch.no_grad():
        collabModel.eval()
        test_loss = 0

        for data, goal in test_load:
            predicted = collabModel(data)
            loss = loss_func(predicted.squeeze(), goal)
            test_loss += loss.item()

        av_test_loss = test_loss / len(test_load)
        print(f"Test loss: {av_test_loss}")
        print(f"Delta loss: {last_loss_tst - av_loss}\n")
        last_loss_tst = av_loss

eval_preds = []
eval_goals = []

with torch.no_grad():
    for data, goal in test_load:
        predicted = collabModel(data)
        eval_preds.extend(predicted.squeeze().tolist())
        eval_goals.extend(goal.tolist())

rmse = root_mean_squared_error(eval_goals, eval_preds)
mae = mean_absolute_error(eval_goals, eval_preds)
r2 = r2_score(eval_goals, eval_preds)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")
