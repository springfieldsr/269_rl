# CS269 Reinforcement Learning Project
usage: train.py

       [-h]
       [--nhid NHID]
       [--pnhid PNHID]
       [--dropout DROPOUT]
       [--pdropout PDROPOUT]
       [--lr LR]
       [--rllr RLLR]
       [--entcoef ENTCOEF]
       [--frweight FRWEIGHT]
       [--batchsize BATCHSIZE]
       [--budgets BUDGETS]
       [--ntest NTEST]
       [--nval NVAL]
       [--datasets DATASETS]
       [--metric METRIC]
       [--remain_epoch REMAIN_EPOCH]
       [--shaping SHAPING]
       [--logfreq LOGFREQ]
       [--maxepisode MAXEPISODE]
       [--save SAVE]
       [--savename SAVENAME]
       [--policynet POLICYNET]
       [--multigraphindex MULTIGRAPHINDEX]
       [--use_entropy USE_ENTROPY]
       [--use_degree USE_DEGREE]
       [--use_local_diversity USE_LOCAL_DIVERSITY]
       [--use_select USE_SELECT]
       [--use_centrality USE_CENTRALITY]
       [--use_feature_similarity USE_FEATURE_SIMILARITY]
       [--use_embedding_similarity USE_EMBEDDING_SIMILARITY]
       [--pg PG]
       [--ppo_epoch PPO_EPOCH]
       [--gpu GPU]
       [--schedule SCHEDULE]

optional arguments:
  * -h, --help
    show this
    help
    message and
    exit
  * --nhid NHID
  * --pnhid PNHID
  * --dropout DROPOUT
  * --pdropout PDROPOUT
  * --lr LR
  * --rllr RLLR
  * --entcoef ENTCOEF
  * --frweight FRWEIGHT
  * --batchsize BATCHSIZE
  * --budgets BUDGETS
    budget per
    class
  * --ntest NTEST
  * --nval NVAL
  * --datasets DATASETS
  * --metric METRIC
  * --remain_epoch REMAIN_EPOCH
    continues
    training $r
    emain_epoch
    epochs
    after all
    the
    selection
  * --shaping SHAPING
    reward
    shaping
    method, 0
    for no
    shaping;1
    for add
    future
    reward,i.e.
    R=
    r+R*gamma;2
    for use fin
    alreward;3
    for
    subtract ba
    seline(valu
    e of curent
    state)1234
    means all
    the method
    is used,
  * --logfreq LOGFREQ
  * --maxepisode MAXEPISODE
  * --save SAVE
  * --savename SAVENAME
  * --policynet POLICYNET
  * --multigraphindex MULTIGRAPHINDEX
  * --use_entropy USE_ENTROPY
  * --use_degree USE_DEGREE
  * --use_local_diversity USE_LOCAL_DIVERSITY
  * --use_select USE_SELECT
  * --use_centrality USE_CENTRALITY
  * --use_feature_similarity USE_FEATURE_SIMILARITY
  * --use_embedding_similarity USE_EMBEDDING_SIMILARITY
  * --pg PG
  * --ppo_epoch PPO_EPOCH
  * --gpu GPU
  * --schedule SCHEDULE