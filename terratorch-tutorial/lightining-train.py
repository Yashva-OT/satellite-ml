from terratorch.datasets import HLSBands
from terratorch.tasks import PixelwiseRegressionTask

model_args = dict(
  backbone="prithvi_eo_v2_300",
  backbone_pretrained=False,
  backbone_num_frames=1,
  backbone_bands=[
      HLSBands.BLUE,
      HLSBands.GREEN,
      HLSBands.RED,
      HLSBands.NIR_NARROW,
      HLSBands.SWIR_1,
      HLSBands.SWIR_2,
  ],
  necks=[{"name": "SelectIndices", "indices": [-1]},
               {"name": "ReshapeTokensToImage"}],
  decoder="FCNDecoder",
  decoder_channels=128,
  head_dropout=0.1
)

task = PixelwiseRegressionTask(
    model_args,
    "EncoderDecoderFactory",
    loss="rmse",
    # lr=lr,
    ignore_index=-1,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.05},
)

# Pass this LightningModule to a Lightning Trainer, together with some LightningDataModule
# Or use YAML config file and run via CLI.