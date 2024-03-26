def download_from_hub(hf_model_id, hf_token=None):
  """
  Downloads a model from huggingface hub

  Args:
    hf_model_id (str): huggingface model id to be downloaded from
    hf_token (str): huggingface read token

  Returns:
    model_path (str): path to downloaded model
  """
  from huggingface_hub import hf_hub_download, list_repo_files

  repo_files = list_repo_files(repo_id=hf_model_id, repo_type="model", token=hf_token)

  # download config file for triggering download counter
  config_file = "config.json"
  if config_file in repo_files:
    _ = hf_hub_download(
      repo_id=hf_model_id,
      filename=config_file,
      repo_type="model",
      token=hf_token,
    )

  # download model file
  model_file = [f for f in repo_files if f.endswith(".pt")][0]
  file = hf_hub_download(
    repo_id=hf_model_id,
    filename=model_file,
    repo_type="model",
    token=hf_token,
  )
  return file
