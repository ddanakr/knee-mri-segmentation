from monai.networks.nets import UNet

def build_unet(config: dict):

    model = UNet(
        spatial_dims = 2,
        in_channels = config.get("in_channels", 1),
        out_channels = config.get("out_channels", 6),
        channels = tuple(config.get("channels", [32, 64, 128, 256, 512])),
        strides = tuple(config.get("strides", [2, 2, 2, 2])),
        num_res_units = config.get("num_res_units", 2),
        norm = config.get("norm", "instance"),
        dropout = config.get("dropout", 0.1)
    )

    return model

if __name__ == "__main__":
    import torch

    dummy_config = {"in_channels": 1, "out_channels": 6}
    model = build_unet(dummy_config)

    x = torch.randn(2, 1, 256, 256)  # batch=2, channel=1, H=256, W=256
    y = model(x)
    print("input shape:", x.shape)
    print("output shape:", y.shape) 