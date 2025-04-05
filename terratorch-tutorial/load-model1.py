from terratorch import BACKBONE_REGISTRY
import ipdb;
ipdb.set_trace()
print(BACKBONE_REGISTRY)

# find available prithvi models
print([model_name for model_name in BACKBONE_REGISTRY if "terratorch_prithvi" in model_name])
# >>> ['terratorch_prithvi_eo_tiny', 'terratorch_prithvi_eo_v1_100', 'terratorch_prithvi_eo_v2_300', 'terratorch_prithvi_eo_v2_600', 'terratorch_prithvi_eo_v2_300_tl', 'terratorch_prithvi_eo_v2_600_tl']

# show all models with list(BACKBONE_REGISTRY)

# check a model is in the registry
"terratorch_prithvi_eo_v2_300" in BACKBONE_REGISTRY
# >>> True

# without the prefix, all internal registries will be searched until the first match is found
"prithvi_eo_v1_100" in BACKBONE_REGISTRY
# >>> True

# instantiate your desired model
# the backbone registry prefix (e.g. `terratorch` or `timm`) is optional
# in this case, the underlying registry is terratorch.
model = BACKBONE_REGISTRY.build("prithvi_eo_v1_100", pretrained=False)

# instantiate your model with more options, for instance, passing weights from your own file
model = BACKBONE_REGISTRY.build(
    "prithvi_eo_v2_300", num_frames=1, ckpt_path='path/to/model.pt'
)
# Rest of your PyTorch / PyTorchLightning code