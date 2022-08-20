__all__ = ["Config"]


class Configs:
    def __init__(self):
        self.content_conv_name = "block5_conv2"
        self.style_conv_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1"
        ]
        self.alpha = 1
        self.beta = 10
        self.weighting_factor = 0.2
