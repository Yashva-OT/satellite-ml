from deepforest import main
import ipdb

def load_deepforest_model():
    model = main.deepforest()
    model.use_release()  # Load pretrained model
    model.model.eval()
    return model


if __name__ == "__main__":
    ipdb.set_trace()
    load_deepforest_model()