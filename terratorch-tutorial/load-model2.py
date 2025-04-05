import terratorch # even though we don't use the import directly, we need it so that the models are available in the timm registry
from terratorch.models import EncoderDecoderFactory
from terratorch.datasets import HLSBands
import ipdb
ipdb.set_trace()

model_factory = EncoderDecoderFactory()

# Let's build a segmentation model
# Parameters prefixed with backbone_ get passed to the backbone
# Parameters prefixed with decoder_ get passed to the decoder
# Parameters prefixed with head_ get passed to the head

model = model_factory.build_model(
    task="segmentation",
    backbone="prithvi_eo_v2_300",
    backbone_pretrained=False,
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
    head_dropout=0.1,
    num_classes=4,
)

# Rest of your PyTorch / PyTorchLightning code