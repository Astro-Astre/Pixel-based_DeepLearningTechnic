from efficientnet_pytorch import EfficientNet


def efficientnet_b0(input_channels, num_class):
    model = EfficientNet.from_name('efficientnet-b0')
    model.in_channels = input_channels
    model._fc.out_features = num_class
    return model


def efficientnet_b1(input_channels, num_class):
    model = EfficientNet.from_name('efficientnet-b1')
    model.in_channels = input_channels
    model._fc.out_features = num_class
    return model


def efficientnet_b2(input_channels, num_class):
    model = EfficientNet.from_name('efficientnet-b2')
    model.in_channels = input_channels
    model._fc.out_features = num_class
    return model


def efficientnet_b3(input_channels, num_class):
    model = EfficientNet.from_name('efficientnet-b3')
    model.in_channels = input_channels
    model._fc.out_features = num_class
    return model


def efficientnet_b4(input_channels, num_class):
    model = EfficientNet.from_name('efficientnet-b4')
    model.in_channels = input_channels
    model._fc.out_features = num_class
    return model


def efficientnet_b5(input_channels, num_class):
    model = EfficientNet.from_name('efficientnet-b5')
    model.in_channels = input_channels
    model._fc.out_features = num_class
    return model


def efficientnet_b6(input_channels, num_class):
    model = EfficientNet.from_name('efficientnet-b6')
    model.in_channels = input_channels
    model._fc.out_features = num_class
    return model


def efficientnet_b7(input_channels, num_class):
    model = EfficientNet.from_name('efficientnet-b7')
    model.in_channels = input_channels
    model._fc.out_features = num_class
    return model
