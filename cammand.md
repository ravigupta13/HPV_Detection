go to CLAM_latest folder
do conda activate clam_latest 

##create features
CUDA_VISIBLE_DEVICES=2,3 python extract_features_fp.py --data_h5_dir /workspace/hpv_project/A_new_pipeline_hpv/h5_files --data_slide_dir /workspace/hpv_project/hpv_svs --csv_path /workspace/hpv_project/hpv_396_mag_label_200.csv --feat_dir /workspace/hpv_project/A_new_pipeline_hpv/feature_regnet --batch_size 1024 --slide_ext .svs --model_name RegNet


CUDA_VISIBLE_DEVICES=2,3 python extract_features_fp.py --data_h5_dir /workspace/CLAM_latest/coords_tum_ntum/h5_res34_old --data_slide_dir /workspace/hpv_project/hpv_svs --csv_path /workspace/hpv_project/hpv_396_mag_label_200.csv --feat_dir /workspace/CLAM_latest/coords_tum_ntum/feature_res34_old/convnext_f --batch_size 1024 --slide_ext .svs --model_name ConvNext_tiny


###main file run

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_sb --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --data_root_dir /workspace/hpv_project/A_new_pipeline_hpv/feature_resnet34 --embed_dim 1024


### Heatmap
python visualize_clam_custom_tcga_heatmap.py --result_dir /workspace/clam1/CLAM/heatmaps/custom_heatmap_result --slides_dir_csv /workspace/clam1/CLAM/Tumor_200_sb_custom_vis_input.csv --ckpt_path /workspace/clam1/CLAM/results_TS_hpv/200_sb_4split/TS_CLAM_200_s1/s_2_checkpoint.pt --h5_dir /workspace/hpv_project/feature_tumor_selected/tumor_vs_normal_resnet_features/h5_files  --model_type clam_sb --slides_dir /workspace/hpv_project/hpv_svs


########################tmh
CUDA_VISIBLE_DEVICES=2,3 python extract_features_fp.py --data_h5_dir /workspace/CLAM_latest/coords_tum_ntum/tmh_feature/h5_tmh_inference_res34_old_cn --data_slide_dir /wsi_dataset/tmh/tmh_hnsc --csv_path /workspace/CLAM_latest/tmh_hpv_84_label_rna_ish.csv --feat_dir /workspace/CLAM_latest/coords_tum_ntum/tmh_feature/feature/#resnet34_old_cn_feat --batch_size 1024 --slide_ext .svs --model_name ResNet34
