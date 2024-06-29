DEVICES=$1
GENOTYPE=$2

# no batchnorm + dropout 0.2

CUDA_VISIBLE_DEVICES=$DEVICES python train.py \
--task 'graph_level' \
--data 'MNIST' \
--nb_classes 10 \
--in_dim_V 3 \
--in_dim_E 1 \
--batch 64 \
--node_dim 50 \
--edge_dim 50 \
--pos_encode 0 \
--nb_layers 4 \
--nb_nodes 3 \
--dropout 0.2 \
--epochs 200 \
--lr 1e-3 \
--weight_decay 0.0 \
--fn_agg sum \
--optimizer 'ADAM' \
--load_genotypes $GENOTYPE