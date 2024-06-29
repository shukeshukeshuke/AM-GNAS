DEVICES=$1
GENOTYPE=$2

CUDA_VISIBLE_DEVICES=$DEVICES python train.py \
--task 'node_level' \
--data 'SBM_CLUSTER' \
--nb_classes 6 \
--in_dim_V 7 \
--in_dim_E 1 \
--pos_encode 0 \
--nb_layers 4 \
--nb_nodes 3 \
--batch 32 \
--node_dim 50 \
--edge_dim 50 \
--dropout 0.2 \
--batchnorm_op \
--epochs 200 \
--lr 1e-3 \
--fn_agg mean \
--weight_decay 0.0 \
--optimizer 'ADAM' \
--load_genotypes $GENOTYPE