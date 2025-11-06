import os
from typing import Optional, Union, List, Type

import torch
from lycoris.kohya import LycorisNetwork, LoConModule
try:
    from lycoris.kohya import LycorisNetworkKohya  # type: ignore
except ImportError:  # pragma: no cover - older LyCORIS versions
    LycorisNetworkKohya = None
from lycoris.modules.glora import GLoRAModule
from torch import nn
from transformers import CLIPTextModel
from toolkit.network_mixins import ToolkitNetworkMixin, ToolkitModuleMixin, ExtractableModuleMixin

# diffusers specific stuff
LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear'
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv'
]

class LoConSpecialModule(ToolkitModuleMixin, LoConModule, ExtractableModuleMixin):
    def __init__(
            self,
            lora_name: str,
            org_module: nn.Module,
            multiplier: float = 1.0,
            lora_dim: int = 4,
            alpha: float = 1,
            dropout: float = 0.,
            rank_dropout: float = 0.,
            module_dropout: float = 0.,
            use_cp: bool = False,
            network: 'LycorisSpecialNetwork' = None,
            use_bias: bool = False,
            **kwargs,
    ):
        """Initialise a LyCORIS LoCon module with toolkit-specific behaviour.

        Args:
            lora_name: Unique identifier for the adapter.
            org_module: Source module (Linear / Conv2d) to wrap.
            multiplier: Runtime multiplier applied to the adapter output.
            lora_dim: Rank for the low-rank decomposition.
            alpha: Alpha scaling value used by LyCORIS.
            dropout: Module-level dropout probability.
            rank_dropout: Rank dropout probability.
            module_dropout: Module dropout probability.
            use_cp: Request CP (Tucker) decomposition for spatial convolutions.
            network: Owning `LycorisSpecialNetwork` instance.
            use_bias: If ``True`` add a trainable bias to the up projection.
            **kwargs: Additional toolkit / LyCORIS parameters (e.g. ``dora_wd``).
        """
        ToolkitModuleMixin.__init__(self, network=network)

        # Remove toolkit-only hints that the base module does not expect
        kwargs.pop('parent', None)
        kwargs.pop('network', None)

        # Map commonly used toolkit flags to LoCon parameters
        weight_decompose = kwargs.pop('weight_decompose', kwargs.pop('dora_wd', False))
        wd_on_out = kwargs.pop('wd_on_out', True)
        bypass_mode = kwargs.pop('bypass_mode', None)
        rank_dropout_scale = kwargs.pop('rank_dropout_scale', False)
        use_scalar = kwargs.pop('use_scalar', False)
        use_tucker = kwargs.pop('use_tucker', False)
        rs_lora = kwargs.pop('rs_lora', False)

        for name, value in (
                ('dropout', dropout),
                ('rank_dropout', rank_dropout),
                ('module_dropout', module_dropout),
        ):
            if value is not None and not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(value).__name__}")

        dropout = 0.0 if dropout is None else float(dropout)
        rank_dropout = 0.0 if rank_dropout is None else float(rank_dropout)
        module_dropout = 0.0 if module_dropout is None else float(module_dropout)

        if org_module.bias is None:
            use_bias = False

        is_conv2d = isinstance(org_module, nn.Conv2d)
        has_spatial_kernel = is_conv2d and getattr(org_module, 'kernel_size', (1, 1)) != (1, 1)
        effective_use_tucker = use_tucker or (use_cp and has_spatial_kernel)

        LoConModule.__init__(
            self,
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            use_tucker=effective_use_tucker,
            use_scalar=use_scalar,
            rank_dropout_scale=rank_dropout_scale,
            weight_decompose=weight_decompose,
            wd_on_out=wd_on_out,
            bypass_mode=bypass_mode,
            rs_lora=rs_lora,
            **kwargs,
        )

        # Track when we requested CP-style decomposition for reference/debugging
        self.cp = effective_use_tucker and getattr(self, 'tucker', False)

        if use_bias:
            if not hasattr(self, 'lora_up'):
                raise RuntimeError(f"Cannot enable bias for {lora_name}: lora_up not initialised")

            try:
                if isinstance(self.lora_up, nn.Linear):
                    new_up = nn.Linear(self.lora_up.in_features, self.lora_up.out_features, bias=True)
                    new_up.weight.data.copy_(self.lora_up.weight.data)
                    nn.init.zeros_(new_up.bias)
                    self.lora_up = new_up
                elif isinstance(self.lora_up, nn.Conv2d):
                    new_up = nn.Conv2d(
                        self.lora_up.in_channels,
                        self.lora_up.out_channels,
                        self.lora_up.kernel_size,
                        stride=self.lora_up.stride,
                        padding=self.lora_up.padding,
                        dilation=self.lora_up.dilation,
                        groups=self.lora_up.groups,
                        bias=True,
                    )
                    new_up.weight.data.copy_(self.lora_up.weight.data)
                    nn.init.zeros_(new_up.bias)
                    self.lora_up = new_up
                else:
                    print(f"⚠️  Warning: use_bias requested for unsupported layer type {type(self.lora_up).__name__}; skipping")
            except Exception as exc:
                raise RuntimeError(f"Failed to add bias parameter for {lora_name}: {exc}") from exc


class LycorisSpecialNetwork(ToolkitNetworkMixin, LycorisNetwork):
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel",
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
        # 'UNet2DConditionModel',
        # 'Conv2d',
        # 'Timesteps',
        # 'TimestepEmbedding',
        # 'Linear',
        # 'SiLU',
        # 'ModuleList',
        # 'DownBlock2D',
        # 'ResnetBlock2D',  # need
        # 'GroupNorm',
        # 'LoRACompatibleConv',
        # 'LoRACompatibleLinear',
        # 'Dropout',
        # 'CrossAttnDownBlock2D', # needed
        # 'Transformer2DModel',  # maybe not, has duplicates
        # 'BasicTransformerBlock', # duplicates
        # 'LayerNorm',
        # 'Attention',
        # 'FeedForward',
        # 'GEGLU',
        # 'UpBlock2D',
        # 'UNetMidBlock2DCrossAttn'
    ]
    UNET_TARGET_REPLACE_NAME = [
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"

    def __init__(
            self,
            text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
            unet,
            multiplier: float = 1.0,
            lora_dim: int = 4,
            alpha: float = 1,
            dropout: Optional[float] = None,
            rank_dropout: Optional[float] = None,
            module_dropout: Optional[float] = None,
            conv_lora_dim: Optional[int] = None,
            conv_alpha: Optional[float] = None,
            use_cp: Optional[bool] = False,
            network_module: Type[object] = LoConSpecialModule,
            train_unet: bool = True,
            train_text_encoder: bool = True,
            use_text_encoder_1: bool = True,
            use_text_encoder_2: bool = True,
            use_bias: bool = False,
            is_lorm: bool = False,
            **kwargs,
    ) -> None:
        # call ToolkitNetworkMixin super
        ToolkitNetworkMixin.__init__(
            self,
            train_text_encoder=train_text_encoder,
            train_unet=train_unet,
            is_lorm=is_lorm,
            **kwargs
        )
        # call the parent of the parent LycorisNetwork
        torch.nn.Module.__init__(self)

        # LyCORIS unique stuff
        if dropout is None:
            dropout = 0
        if rank_dropout is None:
            rank_dropout = 0
        if module_dropout is None:
            module_dropout = 0
        self.train_unet = train_unet
        self.train_text_encoder = train_text_encoder

        self.torch_multiplier = None
        # triggers a tensor update
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.network_type = 'lokr'
        self.base_model_ref = None
        self.peft_format = False
        self.is_pixart = False

        if not self.ENABLE_CONV or conv_lora_dim is None:
            conv_lora_dim = 0
            conv_alpha = 0

        self.conv_lora_dim = int(conv_lora_dim)
        if self.conv_lora_dim and self.conv_lora_dim != self.lora_dim:
            print('Apply different lora dim for conv layer')
            print(f'Conv Dim: {conv_lora_dim}, Linear Dim: {lora_dim}')
        elif self.conv_lora_dim == 0:
            print('Disable conv layer')

        self.alpha = alpha
        self.conv_alpha = float(conv_alpha)
        if self.conv_lora_dim and self.alpha != self.conv_alpha:
            print('Apply different alpha value for conv layer')
            print(f'Conv alpha: {conv_alpha}, Linear alpha: {alpha}')

        if 1 >= dropout >= 0:
            print(f'Use Dropout value: {dropout}')
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        # LoRA+ learning-rate ratios (populated when using LycorisNetworkKohya helpers)
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        # create module instances
        def create_modules(
                prefix,
                root_module: torch.nn.Module,
                target_replace_modules,
                target_replace_names=[]
        ) -> List[network_module]:
            print('Create LyCORIS Module')
            loras = []
            created_names = set()  # Track already created LoRA names to avoid duplicates
            skipped_duplicates = 0  # Count how many duplicates were skipped
            # remove this
            named_modules = root_module.named_modules()
            # add a few to tthe generator

            for name, module in named_modules:
                module_name = module.__class__.__name__
                if module_name in target_replace_modules:
                    if module_name in self.MODULE_ALGO_MAP:
                        algo = self.MODULE_ALGO_MAP[module_name]
                    else:
                        algo = network_module
                    for child_name, child_module in module.named_modules():
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        if lora_name.startswith('lora_unet_input_blocks_1_0_emb_layers_1'):
                            print(f"{lora_name}")

                        if child_module.__class__.__name__ in LINEAR_MODULES and lora_dim > 0:
                            # Skip if we've already created a LoRA for this name
                            if lora_name not in created_names:
                                lora = algo(
                                    lora_name, child_module, self.multiplier,
                                    self.lora_dim, self.alpha,
                                    self.dropout, self.rank_dropout, self.module_dropout,
                                    use_cp,
                                    network=self,
                                    parent=module,
                                    use_bias=use_bias,
                                    **kwargs
                                )
                                created_names.add(lora_name)
                            else:
                                lora = None
                                skipped_duplicates += 1
                        elif child_module.__class__.__name__ in CONV_MODULES:
                            # Skip if we've already created a LoRA for this name
                            if lora_name not in created_names:
                                k_size, *_ = child_module.kernel_size
                                if k_size == 1 and lora_dim > 0:
                                    lora = algo(
                                        lora_name, child_module, self.multiplier,
                                        self.lora_dim, self.alpha,
                                        self.dropout, self.rank_dropout, self.module_dropout,
                                        use_cp,
                                        network=self,
                                        parent=module,
                                    use_bias=use_bias,
                                        **kwargs
                                    )
                                    created_names.add(lora_name)
                                elif conv_lora_dim > 0:
                                    lora = algo(
                                        lora_name, child_module, self.multiplier,
                                        self.conv_lora_dim, self.conv_alpha,
                                        self.dropout, self.rank_dropout, self.module_dropout,
                                        use_cp,
                                        network=self,
                                        parent=module,
                                        use_bias=use_bias,
                                        **kwargs
                                    )
                                    created_names.add(lora_name)
                                else:
                                    continue
                            else:
                                lora = None
                        else:
                            continue
                        if lora is not None:
                            loras.append(lora)
                elif name in target_replace_names:
                    if name in self.NAME_ALGO_MAP:
                        algo = self.NAME_ALGO_MAP[name]
                    else:
                        algo = network_module
                    lora_name = prefix + '.' + name
                    lora_name = lora_name.replace('.', '_')
                    if module.__class__.__name__ == 'Linear' and lora_dim > 0:
                        lora = algo(
                            lora_name, module, self.multiplier,
                            self.lora_dim, self.alpha,
                            self.dropout, self.rank_dropout, self.module_dropout,
                            use_cp,
                            parent=module,
                            network=self,
                            use_bias=use_bias,
                            **kwargs
                        )
                    elif module.__class__.__name__ == 'Conv2d':
                        k_size, *_ = module.kernel_size
                        if k_size == 1 and lora_dim > 0:
                            lora = algo(
                                lora_name, module, self.multiplier,
                                self.lora_dim, self.alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                network=self,
                                parent=module,
                                use_bias=use_bias,
                                **kwargs
                            )
                        elif conv_lora_dim > 0:
                            lora = algo(
                                lora_name, module, self.multiplier,
                                self.conv_lora_dim, self.conv_alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                network=self,
                                parent=module,
                                use_bias=use_bias,
                                **kwargs
                            )
                        else:
                            continue
                    else:
                        continue
                    loras.append(lora)

            # Report if any duplicates were skipped
            if skipped_duplicates > 0:
                print(f"  Skipped {skipped_duplicates} duplicate LoRA module names")

            return loras

        if network_module == GLoRAModule:
            print('GLoRA enabled, only train transformer')
            # only train transformer (for GLoRA)
            LycorisSpecialNetwork.UNET_TARGET_REPLACE_MODULE = [
                "Transformer2DModel",
                "Attention",
            ]
            LycorisSpecialNetwork.UNET_TARGET_REPLACE_NAME = []

        if isinstance(text_encoder, list):
            text_encoders = text_encoder
            use_index = True
        else:
            text_encoders = [text_encoder]
            use_index = False

        self.text_encoder_loras = []
        if self.train_text_encoder:
            for i, te in enumerate(text_encoders):
                if not use_text_encoder_1 and i == 0:
                    continue
                if not use_text_encoder_2 and i == 1:
                    continue
                self.text_encoder_loras.extend(create_modules(
                    LycorisSpecialNetwork.LORA_PREFIX_TEXT_ENCODER + (f'{i + 1}' if use_index else ''),
                    te,
                    LycorisSpecialNetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
                ))
        print(f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules.")
        if self.train_unet:
            # Check if this is a Qwen model and adjust target modules accordingly
            unet_class_name = unet.__class__.__name__
            target_modules = LycorisSpecialNetwork.UNET_TARGET_REPLACE_MODULE

            if unet_class_name == "QwenImageTransformer2DModel":
                # For Qwen Image Edit models, target only the modules that directly contain
                # Linear/Conv2d layers to avoid duplicates from nested module matching
                target_modules = [
                    # Primary targets - these should contain Linear/Conv2d directly
                    "Attention",          # Contains to_q, to_k, to_v, to_out Linear layers
                    "FeedForward",       # Contains Linear layers for MLP
                    # Secondary targets if primary don't work
                    "MLP",
                    "FFN",
                    "GEGLU",
                    "GeGLU",
                    # Fallback to block-level if nothing else works
                    # Note: Only use these if the above don't work, as they may cause duplicates
                    # "QwenImageTransformerBlock",  # Commented out to avoid duplicates
                    # "BasicTransformerBlock",      # Commented out to avoid duplicates
                ]
                print(f"Detected Qwen Image model, using targeted module search to avoid duplicates")

            self.unet_loras = create_modules(LycorisSpecialNetwork.LORA_PREFIX_UNET, unet,
                                             target_modules)

            # If no modules were found with the standard approach, try a fallback
            if len(self.unet_loras) == 0 and unet_class_name == "QwenImageTransformer2DModel":
                print("No modules found with standard targets, attempting automatic discovery...")

                # Collect all module class names that look trainable
                discovered_modules = set()
                for name, module in unet.named_modules():
                    module_class_name = module.__class__.__name__
                    # Look for modules that typically contain parameters
                    if any(keyword in module_class_name.lower() for keyword in
                           ["linear", "conv", "attention", "feedforward", "mlp", "ffn"]):
                        # Check if module has parameters (is trainable)
                        if any(p.requires_grad for p in module.parameters(recurse=False)):
                            discovered_modules.add(module_class_name)

                if discovered_modules:
                    print(f"Discovered potential target modules: {sorted(discovered_modules)}")
                    # Try with the discovered module types
                    self.unet_loras = create_modules(
                        LycorisSpecialNetwork.LORA_PREFIX_UNET,
                        unet,
                        list(discovered_modules)
                    )
                    print(f"Created {len(self.unet_loras)} LyCORIS modules using discovered targets")
        else:
            self.unet_loras = []
        print(f"create LyCORIS for U-Net: {len(self.unet_loras)} modules.")

        # Final warning if no modules were created
        if self.train_unet and len(self.unet_loras) == 0:
            print("\n" + "=" * 60)
            print("⚠️  WARNING: No LyCORIS modules were created for U-Net!")
            print("This usually means the model architecture is incompatible.")
            print("Debugging tips:")
            print("1. Run debug_qwen_modules.py to analyze the model structure")
            print("2. Check that the model is loaded correctly")
            print("3. Verify the model architecture matches your config")
            print("=" * 60 + "\n")

        self.weights_sd = None

        # assertion - now more informative instead of failing
        names = set()
        duplicates = []
        for lora in self.text_encoder_loras + self.unet_loras:
            if lora.lora_name in names:
                duplicates.append(lora.lora_name)
            names.add(lora.lora_name)

        if duplicates:
            print(f"⚠️  Warning: Found {len(duplicates)} duplicate LoRA names (handled gracefully):")
            for dup in duplicates[:5]:  # Show first 5 duplicates
                print(f"   - {dup}")
            if len(duplicates) > 5:
                print(f"   ... and {len(duplicates) - 5} more")

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        """Attach LyCORIS modules to the provided encoders with optional flags."""
        if apply_text_encoder:
            print("enable LyCORIS for text encoder")
        else:
            print("disable LyCORIS for text encoder")
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LyCORIS for U-Net")
        else:
            print("disable LyCORIS for U-Net")
            self.unet_loras = []

        self.loras = self.text_encoder_loras + self.unet_loras
        if len(self.loras) == 0:
            raise ValueError("No LyCORIS modules available to apply; ensure module discovery succeeded.")

        super().apply_to()

    def prepare_optimizer_params(
            self,
            text_encoder_lr: Optional[float] = None,
            unet_lr: Optional[float] = None,
            default_lr: Optional[float] = None,
            optimizer_params: Optional[dict] = None,
            learning_rate: Optional[float] = None,
            **extra,
    ):
        """Bridge toolkit optimiser configuration to the upstream LyCORIS API."""
        # Preserve backwards compatibility with callers that still pass ``learning_rate``
        if learning_rate is None:
            learning_rate = default_lr

        lycoris_impl = LycorisNetworkKohya.prepare_optimizer_params if LycorisNetworkKohya is not None else None

        if lycoris_impl is not None:
            params = lycoris_impl(
                self,
                text_encoder_lr=text_encoder_lr if text_encoder_lr is not None else learning_rate,
                unet_lr=unet_lr if unet_lr is not None else learning_rate,
                learning_rate=learning_rate,
            )
        else:
            # Fallback to base implementation (older LyCORIS exposes only ``lr``)
            base_lr = learning_rate if learning_rate is not None else text_encoder_lr or unet_lr
            params = super().prepare_optimizer_params(base_lr)

        # Upstream returns a tuple of (param_groups, descriptions); only the groups are used here.
        if isinstance(params, tuple) and len(params) >= 1:
            params = params[0]

        # Optionally allow caller-specified overrides (e.g. weight decay) per param group.
        if optimizer_params:
            for group in params:
                group.update({k: v for k, v in optimizer_params.items() if k != 'lr'})

        return params
