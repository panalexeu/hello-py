import pandas as pd
from sklearn.model_selection import train_test_split

# Read the training data
train_df = pd.read_csv('train.csv')

# Split into new_train (85%) and test (15%)
new_train, test = train_test_split(
    train_df,
    test_size=0.15,
    random_state=42,
    stratify=train_df['Transported']
)

# Save new_train
new_train.to_csv('new_train.csv', index=False)

# Create test file WITH Transported column (for evaluation)
test.to_csv('test_with_labels.csv', index=False)

# Create test file WITHOUT Transported column (for predictions)
test_features = test.drop('Transported', axis=1)
test_features.to_csv('test.csv', index=False)

# Print info about the split
print(f"Original dataset size: {len(train_df)}")
print(f"New train set size: {len(new_train)} ({len(new_train)/len(train_df)*100:.1f}%)")
print(f"Test set size: {len(test)} ({len(test)/len(train_df)*100:.1f}%)")
print(f"\nClass distribution in original: \n{train_df['Transported'].value_counts(normalize=True)}")
print(f"\nClass distribution in new_train: \n{new_train['Transported'].value_counts(normalize=True)}")
print(f"\nClass distribution in test: \n{test['Transported'].value_counts(normalize=True)}")
print(f"\nFiles created:")
print(f"- new_train.csv (with Transported column)")
print(f"- test.csv (without Transported column - for making predictions)")
print(f"- test_with_labels.csv (with Transported column - for evaluation)")