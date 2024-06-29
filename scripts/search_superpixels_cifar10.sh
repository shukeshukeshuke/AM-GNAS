DEVICES=$1

# no batchnorm + dropout 0.2
CUDA_VISIBLE_DEVICES=$DEVICES python search.py \
--task 'graph_level' \
--data 'CIFAR10' \
--nb_classes 10 \
--in_dim_V 5 \
--in_dim_E 1 \
--batch 64 \
--epochs 50 \
--node_dim 50 \
--edge_dim 50 \
--portion 0.2 \
--nb_layers 4 \
--nb_nodes 3 \
--dropout 0.2 \
--nb_workers 0 \
--report_freq 1 \
--fn_agg mean \
--search_mode 'train' \
--arch_save 'archs/folder5'