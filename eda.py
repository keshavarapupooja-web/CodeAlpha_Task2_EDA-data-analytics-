import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("sales_data.csv")

print("=" * 50)
print("FIRST FIVE RECORDS")
print("=" * 50)
print(df.head())

print("\nDATASET INFORMATION")
print(df.info())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

# Create Total Sales column
df["TotalSales"] = df["Price"] * df["Quantity"]

print("\nTOTAL SALES COLUMN ADDED")
print(df.head())

print("\nCATEGORY WISE SALES")
print(df.groupby("Category")["TotalSales"].sum())

print("\nCITY WISE SALES")
print(df.groupby("City")["TotalSales"].sum())

print("\nTOP SELLING PRODUCT")
print(df.groupby("Product")["Quantity"].sum().sort_values(ascending=False))

print("\nCORRELATION MATRIX")
print(df[["Price", "Quantity", "TotalSales"]].corr())

plt.figure(figsize=(8,5))
sns.barplot(x="Category", y="TotalSales", data=df)
plt.title("Category Wise Sales")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Price"], bins=6)
plt.title("Price Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="Price", y="Quantity", hue="Category", data=df)
plt.title("Price vs Quantity")
plt.show()

plt.figure(figsize=(6,5))
corr = df[["Price","Quantity","TotalSales"]].corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

print("EDA Completed Successfully!")