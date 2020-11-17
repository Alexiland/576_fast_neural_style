train full

```bash
CUDA_VISIBLE_DEVICES=6 python neural_style/neural_style.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --save-model-dir ./logs/full --epochs 2 --cuda 6
```

train quant 0

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 0 --num_bits 0 --num_grad_bits 0 > ./train_logs/quant/style-candy_num_bits-0_num_grad_bits-0.txt 2>&1 & 
```

train quant 4

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 1 --num_bits 4 --num_grad_bits 4 > ./train_logs/quant/style-candy_num_bits-4_num_grad_bits-4.txt 2>&1 & 
```

train quant 8

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 2 --num_bits 8 --num_grad_bits 8 > ./train_logs/quant/style-candy_num_bits-8_num_grad_bits-8.txt 2>&1 & 
```







Stylize

```bash
CUDA_VISIBLE_DEVICES=4 python neural_style/neural_style.py eval --content-image ./images/content-images/amber.jpg --model ./logs/full/epoch_2_Sun_Nov_15_23:50:36_2020_100000.0_10000000000.0.model --output-image ./output_images/full/amber_out.jpg --cuda 4
```

