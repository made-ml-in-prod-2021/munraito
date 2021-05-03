from dataclasses import dataclass
from marshmallow_dataclass import class_schema


@dataclass()
class EvalPipelineParams:
    input_df_path: str
    preds_path: str
    transformer_path: str
    model_path: str


EvalPipelineParamsSchema = class_schema(EvalPipelineParams)
