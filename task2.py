import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Save plots to files instead of displaying
import matplotlib.pyplot as plt

# Load your local Titanic dataset from Kaggle
titanic = pd.read_csv(r'A:\New folder (3)\train.csv')

# Step 1: Data Cleaning
titanic_cleaned = titanic.copy()

# Drop high-missing or irrelevant columns if they exist
columns_to_drop = [col for col in ['Cabin', 'Ticket', 'Name'] if col in titanic_cleaned.columns]
titanic_cleaned.drop(columns=columns_to_drop, inplace=True)

# Fill missing values
titanic_cleaned['Age'] = titanic_cleaned['Age'].fillna(titanic_cleaned['Age'].median())
titanic_cleaned['Embarked'] = titanic_cleaned['Embarked'].fillna(titanic_cleaned['Embarked'].mode()[0])

# Convert appropriate columns to categorical
categorical_columns = ['Sex', 'Pclass', 'Embarked']
for col in categorical_columns:
    titanic_cleaned[col] = titanic_cleaned[col].astype('category')

# Plot 1: Survival Rate by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=titanic_cleaned)
plt.title('Survival Rate by Gender')
plt.tight_layout()
plt.savefig("survival_by_gender.png")

# Plot 2: Age Distribution by Survival
plt.figure(figsize=(8, 5))
sns.histplot(data=titanic_cleaned, x='Age', hue='Survived', bins=30, kde=True, palette='Set2')
plt.title('Age Distribution by Survival')
plt.tight_layout()
plt.savefig("age_distribution_by_survival.png")

# Plot 3: Survival Rate by Class
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=titanic_cleaned)
plt.title('Survival Rate by Passenger Class')
plt.tight_layout()
plt.savefig("survival_by_class.png")

# Plot 4: Fare Distribution by Survival
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Fare', data=titanic_cleaned)
plt.title('Fare Distribution by Survival')
plt.xticks([0, 1], ['Did not survive', 'Survived'])
plt.tight_layout()
plt.savefig("fare_distribution_by_survival.png")

# Plot 5: Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = titanic_cleaned.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
