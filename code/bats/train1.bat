for %%lr in (0.02, 0.05, 0.1) do (
    python main.py --lr %lr% --epochs 200 --batch_size 16
)