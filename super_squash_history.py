from huggingface_hub import HfApi

api = HfApi()

repo_id = "LabyrinthAl/household3"

api.super_squash_history(
    repo_id=repo_id,
    repo_type="dataset",
    branch="main",
    commit_message="Public commit",
)
