from transformers import AdamW
from transformers import WarmupLinearSchedule


def prepare_optimizer_and_scheduler(args, model, t_total=None):
    """Returns an Adam optimizer configured for optimization of a Transformers model (`model`).

    Args:
        args (ArgumentParser): Object containing objects parsed from the command line.
        model (nn.Module): The Transformers model to be fine-tuned.
        t_total (int): The total number of optimization steps.

    Returns:
        A 2-tuple containing an initialized `AdamW` optimizer and `WarmupLinearSchedule` scheduler
        for the fine-tuning of a Transformers (`model`).
    """
    # These are hardcoded because transformers named them to match to TF implementations
    decay_blacklist = {'LayerNorm.bias', 'LayerNorm.weight'}

    decay, no_decay = [], []

    for name, param in model.named_parameters():
        # Frozen weights
        if not param.requires_grad:
            continue
        # A shape of len 1 indicates a normalization layer
        # See: https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
        if len(param.shape) == 1 or name.endswith('.bias') or name in decay_blacklist:
            no_decay.append(param)
        else:
            decay.append(param)

    grouped_parameters = [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': args.weight_decay}
    ]

    optimizer = AdamW(grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon,
                      correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=args.warmup * t_total,
                                     t_total=t_total)

    return optimizer, scheduler
