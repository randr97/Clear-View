# CLEAR VIEW

Lightweight De-Hazification model with 1800 trainable params and 8MiB in size.

# Instruction
1. `driver.py` is the start point of execution
2. It takes in multiple args
```
python3 driver.py \
    --og_img=data/original/ \
    --hazy_img=data/haze/ \
    --lr=0.0004 \
    --weight_decay=0.0001 \
    --grad_clip_norm=0.1 \
    --num_epochs=10 \
    --train_batch=8 \
    --val_batch=8 \
    --num_workers=4 \
    --snapshots_folder=snapshots/ \
    --model=clearview \
    --run_id=clearview \
    --logs=logs/
```
3. Inorder to run clearview model use option `--model=clearview` else `--model=dehaze` for DehazeNet
4. Use `plot.ipynb` to visualize various results

>Note: The training process outputs `Run logs in logs folder` and `model.pth in snapshots` folder
