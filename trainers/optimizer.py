from torch.optim import AdamW, SGD


def build_optimizer(model, config, learning_rate: float | None = None):

    name = config.optimizer.name.lower()

    lr = (
        learning_rate
        if learning_rate is not None
        else config.optimizer.learning_rate
    )

    if name == "adamw":

        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config.optimizer.weight_decay,
        )

    if name == "sgd":

        return SGD(
            model.parameters(),
            lr=lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
        )

    raise ValueError(name)
