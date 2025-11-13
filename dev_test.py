import urllib.request
import pandas as pd

url = "https://raw.githubusercontent.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs/refs/heads/main/data_complete.csv"
csv_filename = "data_complete.csv"

# Download the CSV from the internet and save it locally
with urllib.request.urlopen(url) as response:
    content = response.read()

with open(csv_filename, "wb") as f:
    f.write(content)

# Load the CSV into a DataFrame
df = pd.read_csv(csv_filename)

# Define total_implementation_time as initial + post-review
df["total_implementation_time"] = (
    df["initial_implementation_time"] + df["post_review_implementation_time"]
)

# Create direct references to the key columns
dev_id = df["dev_id"]
issue_id = df["issue_id"]
ai_treatment = df["ai_treatment"]
total_implementation_time = df["total_implementation_time"]

print(df[["dev_id", "issue_id", "ai_treatment", "total_implementation_time"]].head())

