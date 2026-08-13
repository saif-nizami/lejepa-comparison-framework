from torch.optim import AdamW, SGD


def build_optimizer(model, config, learning_rate: float | None = None,):

    name = config.optimizer.name.lower()

    if name == "adamw":

        return AdamW(
            model.parameters(),
            # lr=config.optimizer.learning_rate,
            lr = (
                learning_rate
                if learning_rate is not None
                else cfg.optimizer.learning_rate
            ),
            weight_decay=config.optimizer.weight_decay,
        )

    if name == "sgd":

        return SGD(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            momentum=0.9,
            weight_decay=config.optimizer.weight_decay,
        )

    raise ValueError(name)