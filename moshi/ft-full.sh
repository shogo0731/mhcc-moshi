export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,4"
export TORCH_COMPILE_DISABLE=1
# export CUDA_LAUNCH_BLOCKING=1

# フォルダを削除するので気を付けること
rm -rf /users/s1f102201582/projects/mhcc-moshi/moshi/output/v2.2
cd /users/s1f102201582/projects/moshi-finetune && torchrun \
--nproc-per-node=4 \
-m train /users/s1f102201582/projects/mhcc-moshi/moshi/ft_config-full.yaml
