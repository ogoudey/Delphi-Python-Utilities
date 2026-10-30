import pyarrow as pa
import pyarrow.parquet as pq


from huggingface_hub import HfApi

api = HfApi()

repo_id = "LabyrinthAl/household3"
column = "language_persistent"

parquet_files = [
	f for f in api.list_repo_files(
	    repo_id=repo_id,
	    repo_type="dataset",
	)
	if f.startswith("data/chunk-000/")
	and f.endswith(".parquet")
]

print("\n".join(parquet_files))

input("continue?")

for file in parquet_files:
    print(f"Editing {file}...")

    path = f"hf://datasets/{repo_id}/{file}"
    table = pq.read_table(path)

    index = table.schema.get_field_index(column)

    if index == -1:
        print(f"Skipping: column {column!r} not found in {file}")
        continue

    column_type = table.schema.field(index).type

    # Replace each entry with an empty list, preserving the Arrow type.
    empty_lists = pa.array(
        [[] for _ in range(table.num_rows)],
        type=column_type,
    )

    updated_table = table.set_column(index, column, empty_lists)

    # Write the updated Parquet file back to the Hub.
    pq.write_table(
        updated_table,
        path,
        use_content_defined_chunking=True,
        write_page_index=True,
    )

    print(f"Updated {file}")

print("Finished editing selected Parquet files.")



