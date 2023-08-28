# third party
import torch
from transformers import AutoModelForCausalLM

# torch.multiprocessing.set_start_method('spawn')
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m", load_in_4bit=True, device_map="auto"
)
# model = None


def list_gpus():
    """List the available cuda devices

    Returns:
        List[str]: list of device names
    """
    num_gpus = torch.cuda.device_count()
    return [torch.cuda.get_device_name(i) for i in range(num_gpus)]


def list_models():
    return [model]
