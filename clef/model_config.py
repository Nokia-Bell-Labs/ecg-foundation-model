import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from omegaconf import OmegaConf


def register_ecgfounder_model(config):
    """Create an ECGFounder model"""
    from clef.baselines.models.ECGFounder import (
        ft_1lead_ECGFounder,
        ft_12lead_ECGFounder,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = config.get("num_classes", 4)
    linear_prob = config.get("linear_prob", False)

    lead_config = config.get("lead_config", "1lead")
    if lead_config == "12lead":
        pth_folder = config.get("checkpoint_path", "models/ecgfounder")
        pth = os.path.join(pth_folder, "12_lead_ECGFounder.pth")
        model = ft_12lead_ECGFounder(device, pth, n_classes, linear_prob=linear_prob)
    else:
        pth_folder = config.get("checkpoint_path", "models/ecgfounder")
        pth = os.path.join(pth_folder, "1_lead_ECGFounder.pth")
        model = ft_1lead_ECGFounder(device, pth, n_classes, linear_prob=linear_prob)

    model.name = "ECGFounder"
    return model


def register_ked_model(config):
    """Create a KED xresnet1d model"""
    from clef.baselines.models.KED import xresnet1d101

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = config.get("num_classes", 4)
    lead_config = config.get("lead_config", "1lead")
    input_channels = 12 if lead_config == "12lead" else 1
    linear_prob = config.get("linear_prob", False)

    kernel_size = config.get("kernel_size", 5)
    ps_head = config.get("ps_head", 0.5)

    model = xresnet1d101(
        num_classes=num_classes,
        input_channels=12,
        kernel_size=kernel_size,
        ps_head=ps_head,
    ).to(device=device)

    checkpoint_folder = config.get("checkpoint_path", "models/ecgfm-ked")
    checkpoint_path = (
        os.path.join(
            checkpoint_folder, "best_valid_all_increase_with_augment_epoch_3.pt"
        )
        if checkpoint_folder
        else None
    )
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "ecg_model" in checkpoint:
                model.load_state_dict(checkpoint["ecg_model"])
            else:
                model.load_state_dict(checkpoint)

        except Exception as e:
            print(f"Error loading KED model checkpoint: {e}")

    if lead_config == "1lead" and input_channels == 1:
        old_conv = model[0][0]
        model[0][0] = nn.Conv1d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

    if linear_prob:
        # freeze the backbone
        for param in model.parameters():
            param.requires_grad = False

    model.name = "KED" # xresnet1d101
    return model


def register_stmem_model(config):
    """Create an STMEM model"""
    from clef.baselines.models.STMEM import ST_MEM_ViT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = config.get("num_classes", 4)
    lead_config = config.get("lead_config", "1lead")
    input_channels = 12 if lead_config == "12lead" else 1
    input_len = config.get("input_len", 5000) 
    patch_size = config.get("patch_size", 75) 
    input_len = (input_len // patch_size) * patch_size

    linear_prob = config.get("linear_prob", False)

    model = ST_MEM_ViT(
        seq_len=input_len,
        patch_size=patch_size,
        num_leads=input_channels,
        num_classes=n_classes,
    ).to(device=device)

    checkpoint_folder = config.get("checkpoint_path", "models/st-mem")
    checkpoint_path = (
        os.path.join(checkpoint_folder, "st_mem_vit_base_full.pth")
        if checkpoint_folder
        else None
    )
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Remove key {k} from pre-trained checkpoint")
                    del checkpoint_model[k]
            model.load_state_dict(checkpoint_model, strict=False)
        except Exception as e:
            print(f"Error loading STMEM model checkpoint: {e}")

    if linear_prob:
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

        for param in model.head.parameters():
            param.requires_grad = True

    model.name = "STMEM"
    model.input_len = input_len
    return model


def register_moirai_model(config):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    pred_len = config.get("pred_len", 100)
    input_len = config.get("input_len", 5000)
    patch_size = config.get("patch_size", 100)
    linear_prob = config.get("linear_prob", False)

    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base"), # small, base, large
        prediction_length=pred_len,
        context_length=input_len,
        patch_size=patch_size,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=1,
    )

    if linear_prob:
        # freeze the backbone
        for param in model.parameters():
            param.requires_grad = False
            
    model.name = "Moirai"
    return model


def register_moment_model(config):
    """Create a MOMENT model"""
    from clef.baselines.models.moment import MOMENTPipeline, TASKS

    # Extract parameters from config
    model_name = config.get(
        "model_name", "AutonLab/MOMENT-1-base"
    )
    lead_config = config.get("lead_config", "1lead")
    n_channels = 12 if lead_config == "12lead" else 1
    num_class = config.get("num_classes", 4)
    linear_prob = config.get("linear_prob", False)
    if linear_prob:
        freeze_encoder = True
        freeze_embedder = True
    else:
        freeze_encoder = config.get(
            "freeze_encoder", False
        ) 
        freeze_embedder = config.get("freeze_embedder", False)
    enable_gradient_checkpointing = config.get(
        "enable_gradient_checkpointing", True
    )  
    reduction = config.get("reduction", "concat") 

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": TASKS.EMBED,
            "n_channels": n_channels,
            "num_class": num_class,
            "freeze_encoder": freeze_encoder,
            "freeze_embedder": freeze_embedder,
            "enable_gradient_checkpointing": enable_gradient_checkpointing,
            "reduction": reduction, 
        },
    )
    model.init()

    model.name = "MOMENT"
    return model


def register_resnet_model(config):
    """Create a resnet1d model with configurable size"""
    from clef.baselines.models.CLEF import create_net1d_by_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = config.get("num_classes", 4)
    linear_prob = config.get("linear_prob", False)
    model_size = config.get("model_size", "medium") 
    
    lead_config = config.get("lead_config", "1lead")
    in_channels = 12 if lead_config == "12lead" else 1
    
    model = create_net1d_by_size(
        device=device, 
        model_size=model_size,
        n_classes=n_classes, 
        linear_prob=linear_prob,
        in_channels=in_channels
    )

    model.name = f"resnet1d-{model_size}"
    return model


def register_pretrained_resnet_model(config):
    """Create a resnet1d model and load pretrained weights with configurable size"""
    from clef.baselines.models.CLEF import create_net1d_by_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = config.get("num_classes", 4)
    linear_prob = config.get("linear_prob", False)
    model_size = config.get("model_size", "medium")  # New parameter

    lead_config = config.get("lead_config", "1lead")
    in_channels = 12 if lead_config == "12lead" else 1
    
    # Construct checkpoint path based on model size and lead config
    pth_folder = config.get("checkpoint_path", "models/pretrained")
    pth_name = config.get("statekey_file", "test.ckpt") 
    assert model_size in pth_name

    pth = os.path.join(pth_folder, pth_name)

    model = create_net1d_by_size(
        device=device,
        model_size=model_size,
        n_classes=n_classes,
        linear_prob=linear_prob,
        pth=pth,
        in_channels=in_channels
    )

    if linear_prob:
        # freeze the backbone
        for param in model.parameters():
            param.requires_grad = False

    model.name = f"{pth_name.split("_")[0]}-{pth_name.split("_")[2]}"
    return model


model_mapping = {
    "ecgfounder": register_ecgfounder_model,
    "ked": register_ked_model,
    "stmem": register_stmem_model,
    "moirai": register_moirai_model,
    "moment": register_moment_model,
    "resnet": register_resnet_model,
    "pretrained": register_pretrained_resnet_model,
}


@dataclass
class ModelConfig:
    name: str

    def __post_init__(self):
        if self.name not in model_mapping:
            raise ValueError(
                f"Unknown model: {self.name}. Available models: {list(model_mapping.keys())}"
            )
        self._register_fn = model_mapping[self.name]

    def model_class(self, config, **kwargs):
        """
        Create a model instance while handling additional parameters appropriately.

        Args:
            config: Configuration dictionary/object for the model
            **kwargs: Additional parameters like task_name, datamodule, etc.

        Returns:
            Created model instance
        """
        lead_config = config.get("lead_config", "1lead")

        # Update config with lead configuration
        if isinstance(config, dict):
            config["lead_config"] = lead_config
        else:
            config = OmegaConf.merge(config, {"lead_config": lead_config})

        # Create the model using the updated config
        model = self._register_fn(config)

        return model
