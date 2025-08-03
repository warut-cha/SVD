import torch

from torch.profiler import profile, record_function, ProfilerActivity


def estimate_model_complexity(model: torch.nn.Module) -> float:
    _ = model.eval()

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 64, 64, 2, device=device)  # Dummy input

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True
    ) as prof:
        with record_function("model_inference"):
            model(dummy_input)

    total_flops = 0
    for event in prof.key_averages():
        if event.flops is not None:
            total_flops += event.flops

    total_macs = total_flops / 2
    mega_macs = total_macs / 1e6

    return mega_macs