# MER_with_Transformer
Pytorch implementation for "Multi-label Multimodal Emotion Recognition with Transformer-based Fusion and Emotion-level Representation Learning.". IEEE Access, vol.11, pp. 14742-14751, 2023. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10042438
## Cmd for running
python main.py -lr=5e-5 -mod=tav --img-interval=500 --loss=bce --model=mme2e --num-emotions=6 --trans-dim=256 --trans-nlayers=4 --trans-nheads=4 --text-lr-factor=10 --text-model-size=base --text-max-len=50 --cuda=0 --early-stop=5 -ep=30 -bs=32 --dataset=mosei
