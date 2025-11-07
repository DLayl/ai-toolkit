#!/bin/bash
# Launch script for Wan2.2 LoRA training on GH200 with UVM optimization

echo "=================================================="
echo "Wan2.2 LoRA Training on GH200 with UVM Support"
echo "=================================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. Configure UVM for optimal performance
echo -e "${BLUE}Configuring PyTorch UVM...${NC}"
export PYTORCH_CUDA_ALLOC_CONF="use_uvm:True,uvm_oversubscription_ratio:3.0,uvm_stage1_threshold:0.15,uvm_stage1_min_bytes:10737418240,uvm_access_pattern:gpu_first"

# Explanation of UVM settings:
# - use_uvm:True - Enable UVM
# - uvm_oversubscription_ratio:3.0 - Allow up to 3x GPU memory (294GB effective)
# - uvm_stage1_threshold:0.15 - Switch to UVM at 15% free VRAM (~83GB used)
# - uvm_stage1_min_bytes:10GB - Minimum 10GB free before stage 1
# - uvm_access_pattern:gpu_first - Optimize for GPU-centric access

echo "UVM Configuration:"
echo "  - Oversubscription: 3.0x (294GB effective memory)"
echo "  - Stage 1 threshold: 15% free VRAM"
echo "  - Access pattern: GPU-first"
echo ""

# 2. Set CUDA optimizations
echo -e "${BLUE}Setting CUDA optimizations...${NC}"
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_SYNC=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 3. Set Flash Attention 3
export USE_FLASH_ATTENTION=1
export FLASH_ATTENTION_VERSION=3

# 4. Optimize for GH200's NVLink-C2C
export NCCL_P2P_LEVEL=NVL  # NVLink optimized
export NCCL_IB_DISABLE=1    # No InfiniBand on single node

# 5. Create output directories
echo -e "${BLUE}Creating output directories...${NC}"
mkdir -p ~/ai-toolkit/output
mkdir -p ~/ai-toolkit/logs
mkdir -p ~/ai-toolkit/cache

# 6. Fix dataset paths if needed
if [ ! -d ~/DLAY-1024 ]; then
    echo -e "${YELLOW}Warning: ~/DLAY-1024 not found!${NC}"
    echo "Please ensure your dataset is at ~/DLAY-1024/"
    exit 1
fi

# 7. Launch training with monitoring
echo -e "${GREEN}Starting training...${NC}"
echo ""

# Log file with timestamp
LOG_FILE=~/ai-toolkit/logs/wan22_training_$(date +%Y%m%d_%H%M%S).log

# Launch with accelerate
cd ~/ai-toolkit

# Start UVM monitor in background
python3 ~/monitor_uvm_training.py &
MONITOR_PID=$!

# Launch training
accelerate launch --mixed_precision bf16 \
    run.py config/wan22_dlay_gh200_uvm.yaml 2>&1 | tee $LOG_FILE

# Kill monitor when training ends
kill $MONITOR_PID 2>/dev/null

echo ""
echo -e "${GREEN}Training complete!${NC}"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Check your outputs at:"
echo "  - Models: ~/ai-toolkit/output/dlay_wan22_gh200_uvm_alpha/"
echo "  - Samples: ~/ai-toolkit/output/dlay_wan22_gh200_uvm_alpha/samples/"