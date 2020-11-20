train full

```bash
CUDA_VISIBLE_DEVICES=6 python neural_style/neural_style.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --save-model-dir ./logs/full --epochs 2 --cuda 6
```

# candy

train quant 0 

```bash
CUDA_VISIBLE_DEVICES=7 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 9 --num_bits 0 --num_grad_bits 0 > ./train_logs/quant/style-candy_num_bits-0_num_grad_bits-0.txt 2>&1 & 
```

train quant 1 

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 1 --num_bits 1 --num_grad_bits 1 > ./train_logs/quant/style-candy_num_bits-1_num_grad_bits-1.txt 2>&1 & 
```

train quant 2 

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 2 --num_bits 2 --num_grad_bits 1 > ./train_logs/quant/style-candy_num_bits-2_num_grad_bits-2.txt 2>&1 & 
```

train quant 3 

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 3 --num_bits 3 --num_grad_bits 1 > ./train_logs/quant/style-candy_num_bits-3_num_grad_bits-3.txt 2>&1 & 
```

train quant 4 

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 1 --num_bits 4 --num_grad_bits 4 > ./train_logs/quant/style-candy_num_bits-4_num_grad_bits-4.txt 2>&1 & 
```

train quant 6 

```bash
CUDA_VISIBLE_DEVICES=5 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 5 --num_bits 6 --num_grad_bits 6 > ./train_logs/quant/style-candy_num_bits-6_num_grad_bits-6.txt 2>&1 & 
```

train quant 8 

```bash
CUDA_VISIBLE_DEVICES=6 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 8 --num_bits 8 --num_grad_bits 8 > ./train_logs/quant/style-candy_num_bits-8_num_grad_bits-8.txt 2>&1 & 
```

train quant 16 

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 8 --num_bits 16 --num_grad_bits 16 > ./train_logs/quant/style-candy_num_bits-16_num_grad_bits-16.txt 2>&1 & 
```



train quant 4 grad 0

```bash
CUDA_VISIBLE_DEVICES=8 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 8 --num_bits 4 --num_grad_bits 0 > ./train_logs/quant/style-candy_num_bits-4_num_grad_bits-0.txt 2>&1 & 
```

train quant 6 grad 0

```bash
CUDA_VISIBLE_DEVICES=9 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 9 --num_bits 6 --num_grad_bits 0 > ./train_logs/quant/style-candy_num_bits-6_num_grad_bits-0.txt 2>&1 & 
```

train quant 8 grad 0

```bash
CUDA_VISIBLE_DEVICES=7 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 7 --num_bits 8 --num_grad_bits 0 > ./train_logs/quant/style-candy_num_bits-8_num_grad_bits-0.txt 2>&1 & 
```

train quant 16 grad 0

```bash
CUDA_VISIBLE_DEVICES=6 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/candy.jpg --style_name candy --save-model-dir ./logs/quant --epochs 2 --cuda 6 --num_bits 16 --num_grad_bits 0 > ./train_logs/quant/style-candy_num_bits-16_num_grad_bits-0.txt 2>&1 & 
```



# Mosaic

train quant 0 

```bash
CUDA_VISIBLE_DEVICES=8 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/mosaic.jpg --style_name mosaic --save-model-dir ./logs/quant --epochs 2 --cuda 9 --num_bits 0 --num_grad_bits 0 > ./train_logs/quant/style-mosaic_num_bits-0_num_grad_bits-0.txt 2>&1 & 
```

train quant 4 

```bash
CUDA_VISIBLE_DEVICES=6 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/mosaic.jpg --style_name mosaic --save-model-dir ./logs/quant --epochs 2 --cuda 1 --num_bits 4 --num_grad_bits 4 > ./train_logs/quant/style-mosaic_num_bits-4_num_grad_bits-4.txt 2>&1 & 
```

train quant 6 

```bash
CUDA_VISIBLE_DEVICES=7 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/mosaic.jpg --style_name mosaic --save-model-dir ./logs/quant --epochs 2 --cuda 7 --num_bits 6 --num_grad_bits 6 > ./train_logs/quant/style-mosaic_num_bits-6_num_grad_bits-6.txt 2>&1 & 
```

train quant 8 

```bash
CUDA_VISIBLE_DEVICES=8 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/mosaic.jpg --style_name mosaic --save-model-dir ./logs/quant --epochs 2 --cuda 8 --num_bits 8 --num_grad_bits 8 > ./train_logs/quant/style-mosaic_num_bits-8_num_grad_bits-8.txt 2>&1 & 
```

train quant 16 

```bash
CUDA_VISIBLE_DEVICES=7 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/mosaic.jpg --style_name mosaic --save-model-dir ./logs/quant --epochs 2 --cuda 7 --num_bits 16 --num_grad_bits 16 > ./train_logs/quant/style-mosaic_num_bits-16_num_grad_bits-16.txt 2>&1 & 
```



train quant 1 grad 0

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/mosaic.jpg --style_name mosaic --save-model-dir ./logs/quant --epochs 2 --cuda 2 --num_bits 1 --num_grad_bits 0 > ./train_logs/quant/style-mosaic_num_bits-1_num_grad_bits-0.txt 2>&1 & 
```

train quant 4 grad 0

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/mosaic.jpg --style_name mosaic --save-model-dir ./logs/quant --epochs 2 --cuda 1 --num_bits 4 --num_grad_bits 0 > ./train_logs/quant/style-mosaic_num_bits-4_num_grad_bits-0.txt 2>&1 & 
```

train quant 6 grad 0

```bash
CUDA_VISIBLE_DEVICES=5 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/mosaic.jpg --style_name mosaic --save-model-dir ./logs/quant --epochs 2 --cuda 5 --num_bits 6 --num_grad_bits 0 > ./train_logs/quant/style-mosaic_num_bits-6_num_grad_bits-0.txt 2>&1 & 
```

train quant 8 grad 0

```bash
CUDA_VISIBLE_DEVICES=5 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/mosaic.jpg --style_name mosaic --save-model-dir ./logs/quant --epochs 2 --cuda 5 --num_bits 8 --num_grad_bits 0 > ./train_logs/quant/style-mosaic_num_bits-8_num_grad_bits-0.txt 2>&1 & 
```

train quant 16 grad 0

```bash
CUDA_VISIBLE_DEVICES=6 nohup python -u neural_style_quant/neural_style_quant.py train --dataset ./dataset --style-image ./images/style-images/mosaic.jpg --style_name mosaic --save-model-dir ./logs/quant --epochs 2 --cuda 6 --num_bits 16 --num_grad_bits 0 > ./train_logs/quant/style-mosaic_num_bits-16_num_grad_bits-0.txt 2>&1 & 
```







Stylize

```bash
CUDA_VISIBLE_DEVICES=4 python neural_style/neural_style.py eval --content-image ./images/content-images/amber.jpg --model ./logs/full/epoch_2_Sun_Nov_15_23:50:36_2020_100000.0_10000000000.0.model --output-image ./output_images/full/amber_out.jpg --cuda 4
```





Quant Stylize candy

```bash
CUDA_VISIBLE_DEVICES=1 python neural_style_quant/neural_style_quant.py eval --content-image ./images/content-images/amber.jpg --model ./logs/quant/style-candy_num_bits16_num_grad_bits-16.model --output-image ./output_images/quant/style-candy_num_bits-16_num_grad_bits-16_amber_out.jpg --cuda 1
```

Quant Stylize mosaic

```bash
CUDA_VISIBLE_DEVICES=1 python neural_style_quant/neural_style_quant.py eval --content-image ./images/content-images/amber.jpg --model ./logs/quant/style-mosaic_num_bits16_num_grad_bits-16.model --output-image ./output_images/quant/style-mosaic_num_bits-16_num_grad_bits-16_amber_out.jpg --cuda 1
```

