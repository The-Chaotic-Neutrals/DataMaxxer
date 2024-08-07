import sys
import json
import polars as pl
from datasets import load_dataset
from pathlib import Path

# Check if a dataset location is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python FilterData.py <Dataset Huggingface ID>")
    sys.exit(1)

dataset_name = sys.argv[1]

# Load the dataset from Hugging Face
dataset = load_dataset(dataset_name)

# Convert the dataset to a Polars DataFrame
data = pl.DataFrame(dataset['train'].to_pandas())
##### Check For Blanks
# Define a function to check if a conversation contains all required roles
def has_required_roles(conversation):
    roles = set(msg['from'] for msg in conversation)
    return 'system' in roles and 'human' in roles and 'gpt' in roles

# Create a boolean column based on the presence of required roles in the "conversations" column
data = data.with_columns(
    pl.col('conversations').map_elements(has_required_roles).alias('has_required_roles')
)

# Filter the data based on the new boolean column
filtered_data = data.filter(pl.col('has_required_roles'))

##### End Check for blanks

# Create the Filtered directory if it doesn't exist
filtered_dir = Path(__file__).parent.absolute() / "Filtered"
filtered_dir.mkdir(exist_ok=True)

# Save the filtered data as JSONL
output_file = filtered_dir / f"{dataset_name.replace('/', '_')}_filtered.jsonl"

with output_file.open('w') as f:
    for row in filtered_data.drop('has_required_roles').to_dicts():
        json.dump(row, f)
        f.write('\n')

print(f"Original dataset size: {len(data)}")
print(f"Filtered dataset size: {len(filtered_data)}")
print(f"Filtered data saved to {output_file}")
