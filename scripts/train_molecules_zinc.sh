DEVICES=$1
GENOTYPE=$2
CUDA_VISIBLE_DEVICES=$DEVICES python train.py \
--task 'graph_level' \
--data 'ZINC' \
--in_dim_V 28 \
--in_dim_E 4 \
--batch 128 \
--node_dim 50 \
--edge_dim 50 \
--nb_layers 12 \
--nb_nodes  3 \
--dropout 0.1 \
--pos_encode 0 \
--batchnorm_op \
--epochs 200 \
--lr 1e-3 \
--weight_decay 0.0 \
--fn_agg sum \
--optimizer 'ADAM' \
--load_genotypes $GENOTYPE