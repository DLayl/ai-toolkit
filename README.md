# AI Toolkit - Enhanced Fork

This is an enhanced fork of [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) with additional optimizations and fixes, incorporating improvements from [relaxis/ai-toolkit](https://github.com/relaxis/ai-toolkit).

## 🚀 Key Enhancements

### H200/GH200 GPU Optimizations
- **UVM (Unified Virtual Memory) support** for large model training
- **Memory management improvements** for high-VRAM GPUs
- **GH200 helper utilities** for optimal GPU utilization
- **Enhanced PromptEmbeds loading** with device parameter support

### LyCORIS Network Improvements
- **Qwen Image Edit 2509 compatibility** - Full LyCORIS/LoKR support for Qwen models
- **Automatic module discovery** - Intelligent detection of trainable layers
- **Duplicate prevention** - Robust handling of overlapping module hierarchies
- **Architecture-specific targeting** - Model-aware module selection

### MoE (Mixture of Experts) Training
- **Per-expert metrics tracking** with independent downsampling
- **Optimizer state restoration** - Preserves min_lr, max_lr, lr_bump across restarts
- **lr_mask clamping** - Maintains per-expert learning rate bounds
- **Dynamic boundary detection** - Automatic switchBoundaryEvery configuration
- **Improved UI metrics** - Accurate expert tracking and learning rate display

### WAN 2.2 Video Model Support
- **VAE dtype handling fixes** - Prevents blurry samples in I2V training
- **Temporal jitter** - Enhanced frame sampling for video training
- **Rotary embedding fixes** - Matches Diffusers WAN reference implementation
- **SageAttention support** - Memory-efficient attention for video models

### Upstream Integrations
- **CFG-zero toggle** (ostris) - Optional guidance loss configuration
- **Qwen control image fix** (ostris) - Correct resolution scaling for match_target_res
- **Automagic optimizer enhancements** - Improved state persistence and avg_lr tracking

## 🔧 Technical Improvements

### Memory Management
- Enhanced memory allocation for H200/GH200 GPUs
- Optimized VAE encoding/decoding with proper dtype handling
- Improved gradient accumulation and memory flushing

### Training Stability
- Fixed FP16 hardcoding causing precision loss
- Proper mask_multiplier dtype handling (native vs forced FP16)
- Enhanced error handling with graceful degradation

### Code Quality
- Comprehensive duplicate tracking in module creation
- Robust error handling with informative warnings
- Model-specific logic with automatic fallbacks

## 📦 Installation

```bash
git clone https://github.com/DLayl/ai-toolkit.git
cd ai-toolkit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🎯 Use Cases

This fork is particularly well-suited for:
- **Qwen Image Edit training** with LyCORIS networks
- **WAN 2.2 I2V (Image-to-Video)** LoRA training
- **H200/GH200 GPU** workloads requiring UVM
- **MoE (Mixture of Experts)** multistage training
- **High-resolution video** model fine-tuning

## 📊 Commit History

Latest commit (98defc8): **Sync upstream fixes and apply GH200/Qwen improvements**
- 14 files changed: 982 insertions, 466 deletions
- Integrated relaxis MoE optimizer fixes
- Integrated ostris CFG-zero and Qwen control fixes
- Complete LyCORIS/Qwen compatibility layer
- GH200 UVM optimizations
- Memory management improvements

## 🙏 Acknowledgments

This fork builds upon excellent work from:
- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) - Original AI Toolkit
- [relaxis/ai-toolkit](https://github.com/relaxis/ai-toolkit) - MoE training improvements

## 📝 License

Same as upstream - see [LICENSE](LICENSE) file for details.

## 🔗 Related Repositories

- Upstream: https://github.com/ostris/ai-toolkit
- MoE Fork: https://github.com/relaxis/ai-toolkit
- This Fork: https://github.com/DLayl/ai-toolkit
