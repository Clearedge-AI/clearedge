from ultralytics import YOLO as YOLOBase
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
from .hf_utils import download_from_hub

from pathlib import Path

class YOLO(YOLOBase):
  def __init__(self, model="yolov8n.yaml", type="v8", hf_token=None) -> None:
    """
    Initializes the YOLO object.

    Args:
      model (str, Path): model to load or create
      type (str): Type/version of models to use. Defaults to "v8".
      hf_token (str): huggingface token
    """
    self.type = type
    self.ModelClass = None  # model class
    self.TrainerClass = None  # trainer class
    self.ValidatorClass = None  # validator class
    self.PredictorClass = None  # predictor class
    self.predictor = None  # reuse predictor
    self.model = None  # model object
    self.trainer = None  # trainer object
    self.task = None  # task type
    self.ckpt = None  # if loaded from *.pt
    self.cfg = None  # if loaded from *.yaml
    self.ckpt_path = None
    self.overrides = {}  # overrides for trainer object

    # needed so torch can load models
    super().__init__()

    # Load or create new YOLO model
    suffix = Path(model).suffix
    if not suffix and Path(model).stem in GITHUB_ASSETS_STEMS:
      model, suffix = (
          Path(model).with_suffix(".pt"),
          ".pt",
      )  # add suffix, i.e. yolov8n -> yolov8n.pt
    try:
      if Path(model).suffix not in (".pt", ".yaml"):
        self._load_from_hf_hub(model, hf_token=hf_token)
      elif suffix == ".yaml":
        self._new(model)
      else:
        self._load(model)
    except Exception as e:
      raise NotImplementedError(
          f"Unable to load model='{model}'. "
          f"As an example try model='yolov8n.pt' or model='yolov8n.yaml'"
      ) from e

  def _load_from_hf_hub(self, weights: str, hf_token=None):
    """
    Initializes a new model and infers the task type from the model head

    Args:
        weights (str): model checkpoint to be loaded
        hf_token (str): huggingface token
    """
    # try to download from hf hub
    weights = download_from_hub(weights, hf_token=hf_token)

    self.model, self.ckpt = attempt_load_one_weight(weights)
    self.ckpt_path = weights
    self.task = self.model.args["task"]
    self.overrides = self.model.args
    self._reset_ckpt_args(self.overrides)

    # for loading a model with ultralytics <8.0.44
    if hasattr(self, "_assign_ops_from_task"):
      (
          self.ModelClass,
          self.TrainerClass,
          self.ValidatorClass,
          self.PredictorClass,
      ) = self._assign_ops_from_task()

    # for loading a model with ultralytics >=8.0.44
    else:
      if self.task not in self.task_map:
        raise ValueError(
            f"Task '{self.task}' not supported. Supported tasks: {list(self.task_map.keys())}"
        )
      (
          self.ModelClass,
          self.TrainerClass,
          self.ValidatorClass,
          self.PredictorClass,
      ) = self.task_map[self.task]
