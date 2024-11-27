for %%lr in (0.01, 0.001, 0.0001) do (
    python train1.py --lr %lr% --epochs 100
)