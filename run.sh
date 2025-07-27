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

# predict linemod, 1, 6, 13, 14

for obj_id in 1 6 13 14; do
  python predict_result.py \
    --dataset lm \
    --datasets_path data/BOP \
    --split test \
    --split_type none \
    --img_to_3d \
    --obj_id $obj_id &> logs/predict/lm_test/${obj_id}.log &
  echo "predicting for object ID: $obj_id"
done

# predict ycbv 7,9,21

export CUDA_VISIBLE_DEVICES=6
for obj_id in 7 9 21; do
  python predict_result.py \
    --dataset ycbv \
    --datasets_path data/BOP \
    --split test \
    --split_type none \
    --img_to_3d \
    --obj_id $obj_id &> logs/predict/ycbv_test/${obj_id}.log &
  echo "predicting for object ID: $obj_id"
done

# clean process
# ps -aux | grep predict_result.py | awk '{print $2}' | xargs kill -9

# wait for all background processes to finish
wait