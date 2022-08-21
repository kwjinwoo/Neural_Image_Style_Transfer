__all__ = ["Configs"]


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
        self.alpha = 8
        self.beta = 1e4
        self.weighting_factor = 0.2

        self.iteration = 4000
        self.inter_save = 100


if __name__ == "__main__":
    cfg = Configs()
    print(cfg.alpha / cfg.beta)
