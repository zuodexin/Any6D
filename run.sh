export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=CoordAR/third_party/custom_bop_toolkit:${PYTHONPATH}

# python run_ho3d_anchor.py \
#   --anchor_folder ./anchor_results/dexycb_reference_view_ours \
#   --ycb_model_path ./dataset/ho3d/YCB_Video_Models \
#   --img_to_3d


# python run_ho3d_query.py \
#   --anchor_path anchor_results/dexycb_reference_view_ours \
#   --hot3d_data_root dataset/ho3d \
#   --ycb_model_path dataset/ho3d/YCB_Video_Models


# python run_demo.py

# predict linemod

python predict_result.py \
  --dataset lm \
  --datasets_path data/BOP \
  --split test \
  --split_type none \
  # --img_to_3d 


# predict ycbv

# python predict_result.py \
#   --dataset ycbv \
#   --datasets_path data/BOP \
#   --split test \
#   --split_type none \
#   --img_to_3d 