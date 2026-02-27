from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo, login
import os

repo_id = "maheshnn/PIMA-Diabetes-Prediction" 
repo_type = "dataset"

# Initialize API client
login(os.getenv("HF_TOKEN"))
api = HfApi()

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new space...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

api.upload_folder(
    folder_path="self_paced_courses_1_mlops/data", # Uploading the data 
    repo_id=repo_id,
    repo_type=repo_type,
)
