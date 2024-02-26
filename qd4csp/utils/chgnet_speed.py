from chgnet.model import CHGNet

def chgnet_speed():
    model = CHGNet.load()
    print(model.graph_converter.algorithm)
