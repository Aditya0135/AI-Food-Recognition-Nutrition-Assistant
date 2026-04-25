from pathlib import Path
import shutil

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from AI_Food_Recognition_Nutrition_Assistant import logger
from AI_Food_Recognition_Nutrition_Assistant.entity.config_entity import ModelPusherConfig
from AI_Food_Recognition_Nutrition_Assistant.utils.common import ensure_dir


class ModelPusher:
    def __init__(self, config: ModelPusherConfig):
        self.config = config

    def _prepare_bundle(self) -> Path:
        bundle_dir = ensure_dir(self.config.model_bundle_path)
        model_target = bundle_dir / self.config.model_filename
        class_names_target = bundle_dir / self.config.class_names_filename

        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {self.config.model_path}")

        if not self.config.class_names_path.exists():
            raise FileNotFoundError(f"Class names file not found: {self.config.class_names_path}")

        shutil.copy2(self.config.model_path, model_target)
        shutil.copy2(self.config.class_names_path, class_names_target)
        logger.info(f"Prepared model bundle at {bundle_dir}")
        return bundle_dir

    def push_to_hf(self) -> str:
        if not self.config.hf_repo_id:
            raise ValueError("HF_MODEL_REPO_ID is not set.")

        bundle_dir = self._prepare_bundle()
        api = HfApi(token=self.config.hf_token)
        api.create_repo(repo_id=self.config.hf_repo_id, private=self.config.private_repo, exist_ok=True)
        try:
            api.upload_folder(
                folder_path=str(bundle_dir),
                repo_id=self.config.hf_repo_id,
                repo_type="model",
                commit_message="Upload trained food recognition model",
            )
        except HfHubHTTPError as e:
            raise PermissionError(
                "Hugging Face upload failed. Verify HF_TOKEN has 'Write' access for model repos "
                f"and permission to push to '{self.config.hf_repo_id}'."
            ) from e

        repo_url = f"https://huggingface.co/{self.config.hf_repo_id}"
        logger.info(f"Pushed model bundle to {repo_url}")
        return repo_url
