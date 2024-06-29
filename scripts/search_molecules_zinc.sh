DEVICES=$1

CUDA_VISIBLE_DEVICES=$DEVICES python search.py \
--task 'graph_level' \
--data 'ZINC' \
--data_clip 1.0 \
--in_dim_V 28 \
--in_dim_E 4 \
--batch 64 \
--epochs 40 \
--node_dim 50 \
--edge_dim 50 \
--nb_layers 12 \
--nb_nodes  3 \
--portion 0.9 \
--dropout 0.1 \
--pos_encode 0 \
--nb_workers 0 \
--report_freq 1 \
--fn_agg sum \
--arch_save 'archs/folder2' \
--search_mode 'train' \
--batchnorm_op