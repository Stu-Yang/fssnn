## 神经网络模型训练实验报告

#### dnn-1.py

+ 模型结构：单隐层前馈全连接神经网络
+ 数据集：MNIST数据集
+ 损失函数：`nll_loss`
+ 迭代训练方法：`torch.optim.Adam()`
+ 学习率：0.001

<details>
<summary>详细实验记录，请点击展开</summary>

```
root@yp:~/yp_workplace/fssnn/plaintext-training# python dnn-1.py
---------- Training ----------
Train Epoch: 1 [0/60032 (0%)]   Loss: 2.419761
Train Epoch: 1 [6400/60032 (11%)]       Loss: 0.378025
Train Epoch: 1 [12800/60032 (21%)]      Loss: 0.381010
Train Epoch: 1 [19200/60032 (32%)]      Loss: 0.089337
Train Epoch: 1 [25600/60032 (43%)]      Loss: 0.237961
Train Epoch: 1 [32000/60032 (53%)]      Loss: 0.083655
Train Epoch: 1 [38400/60032 (64%)]      Loss: 0.192265
Train Epoch: 1 [44800/60032 (75%)]      Loss: 0.023349
Train Epoch: 1 [51200/60032 (85%)]      Loss: 0.153509
Train Epoch: 1 [57600/60032 (96%)]      Loss: 0.261063
Train Epoch: 2 [0/60032 (0%)]   Loss: 0.028988
Train Epoch: 2 [6400/60032 (11%)]       Loss: 0.038453
Train Epoch: 2 [12800/60032 (21%)]      Loss: 0.069734
Train Epoch: 2 [19200/60032 (32%)]      Loss: 0.024647
Train Epoch: 2 [25600/60032 (43%)]      Loss: 0.018293
Train Epoch: 2 [32000/60032 (53%)]      Loss: 0.049095
Train Epoch: 2 [38400/60032 (64%)]      Loss: 0.189005
Train Epoch: 2 [44800/60032 (75%)]      Loss: 0.138169
Train Epoch: 2 [51200/60032 (85%)]      Loss: 0.061854
Train Epoch: 2 [57600/60032 (96%)]      Loss: 0.059899
Train Epoch: 3 [0/60032 (0%)]   Loss: 0.116870
Train Epoch: 3 [6400/60032 (11%)]       Loss: 0.077124
Train Epoch: 3 [12800/60032 (21%)]      Loss: 0.029688
Train Epoch: 3 [19200/60032 (32%)]      Loss: 0.020064
Train Epoch: 3 [25600/60032 (43%)]      Loss: 0.015206
Train Epoch: 3 [32000/60032 (53%)]      Loss: 0.053854
Train Epoch: 3 [38400/60032 (64%)]      Loss: 0.103416
Train Epoch: 3 [44800/60032 (75%)]      Loss: 0.041321
Train Epoch: 3 [51200/60032 (85%)]      Loss: 0.024421
Train Epoch: 3 [57600/60032 (96%)]      Loss: 0.073558
Train Epoch: 4 [0/60032 (0%)]   Loss: 0.032608
Train Epoch: 4 [6400/60032 (11%)]       Loss: 0.022690
Train Epoch: 4 [12800/60032 (21%)]      Loss: 0.016534
Train Epoch: 4 [19200/60032 (32%)]      Loss: 0.071902
Train Epoch: 4 [25600/60032 (43%)]      Loss: 0.017176
Train Epoch: 4 [32000/60032 (53%)]      Loss: 0.007934
Train Epoch: 4 [38400/60032 (64%)]      Loss: 0.032427
Train Epoch: 4 [44800/60032 (75%)]      Loss: 0.020052
Train Epoch: 4 [51200/60032 (85%)]      Loss: 0.004956
Train Epoch: 4 [57600/60032 (96%)]      Loss: 0.042306
Train Epoch: 5 [0/60032 (0%)]   Loss: 0.040913
Train Epoch: 5 [6400/60032 (11%)]       Loss: 0.101003
Train Epoch: 5 [12800/60032 (21%)]      Loss: 0.011993
Train Epoch: 5 [19200/60032 (32%)]      Loss: 0.022521
Train Epoch: 5 [25600/60032 (43%)]      Loss: 0.007509
Train Epoch: 5 [32000/60032 (53%)]      Loss: 0.055093
Train Epoch: 5 [38400/60032 (64%)]      Loss: 0.005142
Train Epoch: 5 [44800/60032 (75%)]      Loss: 0.014972
Train Epoch: 5 [51200/60032 (85%)]      Loss: 0.080698
Train Epoch: 5 [57600/60032 (96%)]      Loss: 0.019284
Train Epoch: 6 [0/60032 (0%)]   Loss: 0.010727
Train Epoch: 6 [6400/60032 (11%)]       Loss: 0.017538
Train Epoch: 6 [12800/60032 (21%)]      Loss: 0.078000
Train Epoch: 6 [19200/60032 (32%)]      Loss: 0.013608
Train Epoch: 6 [25600/60032 (43%)]      Loss: 0.033894
Train Epoch: 6 [32000/60032 (53%)]      Loss: 0.021327
Train Epoch: 6 [38400/60032 (64%)]      Loss: 0.006204
Train Epoch: 6 [44800/60032 (75%)]      Loss: 0.016416
Train Epoch: 6 [51200/60032 (85%)]      Loss: 0.062550
Train Epoch: 6 [57600/60032 (96%)]      Loss: 0.011328
Train Epoch: 7 [0/60032 (0%)]   Loss: 0.011965
Train Epoch: 7 [6400/60032 (11%)]       Loss: 0.008691
Train Epoch: 7 [12800/60032 (21%)]      Loss: 0.010216
Train Epoch: 7 [19200/60032 (32%)]      Loss: 0.005899
Train Epoch: 7 [25600/60032 (43%)]      Loss: 0.018620
Train Epoch: 7 [32000/60032 (53%)]      Loss: 0.024563
Train Epoch: 7 [38400/60032 (64%)]      Loss: 0.008717
Train Epoch: 7 [44800/60032 (75%)]      Loss: 0.001434
Train Epoch: 7 [51200/60032 (85%)]      Loss: 0.008146
Train Epoch: 7 [57600/60032 (96%)]      Loss: 0.029489
Train Epoch: 8 [0/60032 (0%)]   Loss: 0.016574
Train Epoch: 8 [6400/60032 (11%)]       Loss: 0.001225
Train Epoch: 8 [12800/60032 (21%)]      Loss: 0.027724
Train Epoch: 8 [19200/60032 (32%)]      Loss: 0.001028
Train Epoch: 8 [25600/60032 (43%)]      Loss: 0.004817
Train Epoch: 8 [32000/60032 (53%)]      Loss: 0.037788
Train Epoch: 8 [38400/60032 (64%)]      Loss: 0.020584
Train Epoch: 8 [44800/60032 (75%)]      Loss: 0.007056
Train Epoch: 8 [51200/60032 (85%)]      Loss: 0.007116
Train Epoch: 8 [57600/60032 (96%)]      Loss: 0.046963
Train Epoch: 9 [0/60032 (0%)]   Loss: 0.001628
Train Epoch: 9 [6400/60032 (11%)]       Loss: 0.001640
Train Epoch: 9 [12800/60032 (21%)]      Loss: 0.006468
Train Epoch: 9 [19200/60032 (32%)]      Loss: 0.079808
Train Epoch: 9 [25600/60032 (43%)]      Loss: 0.026593
Train Epoch: 9 [32000/60032 (53%)]      Loss: 0.006139
Train Epoch: 9 [38400/60032 (64%)]      Loss: 0.013681
Train Epoch: 9 [44800/60032 (75%)]      Loss: 0.002903
Train Epoch: 9 [51200/60032 (85%)]      Loss: 0.014027
Train Epoch: 9 [57600/60032 (96%)]      Loss: 0.003050
Train Epoch: 10 [0/60032 (0%)]  Loss: 0.001097
Train Epoch: 10 [6400/60032 (11%)]      Loss: 0.004033
Train Epoch: 10 [12800/60032 (21%)]     Loss: 0.013155
Train Epoch: 10 [19200/60032 (32%)]     Loss: 0.003086
Train Epoch: 10 [25600/60032 (43%)]     Loss: 0.003378
Train Epoch: 10 [32000/60032 (53%)]     Loss: 0.053060
Train Epoch: 10 [38400/60032 (64%)]     Loss: 0.041658
Train Epoch: 10 [44800/60032 (75%)]     Loss: 0.040401
Train Epoch: 10 [51200/60032 (85%)]     Loss: 0.005178
Train Epoch: 10 [57600/60032 (96%)]     Loss: 0.008221
---------- Testing ----------

Test set: Average loss: 0.0877, Accuracy: 9792/10000 (98%)

```
</details>

#### dnn-2.py

+ 模型结构：单隐层前馈全连接神经网络
+ 数据集：MNIST数据集
+ 损失函数：`mse_loss`
+ 迭代训练方法：`torch.optim.SGD()`
+ 学习率：0.1

<details>
<summary>详细实验记录，请点击展开</summary>

```
---------- Training ----------
Train Epoch: 1 [0/60032 (0%)]   Loss: 6.058053
Train Epoch: 1 [6400/60032 (11%)]       Loss: 5.851298
Train Epoch: 1 [12800/60032 (21%)]      Loss: 5.851256
Train Epoch: 1 [19200/60032 (32%)]      Loss: 5.848225
Train Epoch: 1 [25600/60032 (43%)]      Loss: 5.846541
Train Epoch: 1 [32000/60032 (53%)]      Loss: 5.844421
Train Epoch: 1 [38400/60032 (64%)]      Loss: 5.843657
Train Epoch: 1 [44800/60032 (75%)]      Loss: 5.845916
Train Epoch: 1 [51200/60032 (85%)]      Loss: 5.842612
Train Epoch: 1 [57600/60032 (96%)]      Loss: 5.845046
Train Epoch: 2 [0/60032 (0%)]   Loss: 5.843408
Train Epoch: 2 [6400/60032 (11%)]       Loss: 5.844955
Train Epoch: 2 [12800/60032 (21%)]      Loss: 5.843951
Train Epoch: 2 [19200/60032 (32%)]      Loss: 5.844999
Train Epoch: 2 [25600/60032 (43%)]      Loss: 5.842420
Train Epoch: 2 [32000/60032 (53%)]      Loss: 5.843409
Train Epoch: 2 [38400/60032 (64%)]      Loss: 5.842717
Train Epoch: 2 [44800/60032 (75%)]      Loss: 5.842972
Train Epoch: 2 [51200/60032 (85%)]      Loss: 5.843401
Train Epoch: 2 [57600/60032 (96%)]      Loss: 5.842665
Train Epoch: 3 [0/60032 (0%)]   Loss: 5.842300
Train Epoch: 3 [6400/60032 (11%)]       Loss: 5.842464
Train Epoch: 3 [12800/60032 (21%)]      Loss: 5.842025
Train Epoch: 3 [19200/60032 (32%)]      Loss: 5.842281
Train Epoch: 3 [25600/60032 (43%)]      Loss: 5.841013
Train Epoch: 3 [32000/60032 (53%)]      Loss: 5.841524
Train Epoch: 3 [38400/60032 (64%)]      Loss: 5.841843
Train Epoch: 3 [44800/60032 (75%)]      Loss: 5.840415
Train Epoch: 3 [51200/60032 (85%)]      Loss: 5.842092
Train Epoch: 3 [57600/60032 (96%)]      Loss: 5.842103
Train Epoch: 4 [0/60032 (0%)]   Loss: 5.841815
Train Epoch: 4 [6400/60032 (11%)]       Loss: 5.841543
Train Epoch: 4 [12800/60032 (21%)]      Loss: 5.841352
Train Epoch: 4 [19200/60032 (32%)]      Loss: 5.840089
Train Epoch: 4 [25600/60032 (43%)]      Loss: 5.841389
Train Epoch: 4 [32000/60032 (53%)]      Loss: 5.841222
Train Epoch: 4 [38400/60032 (64%)]      Loss: 5.843036
Train Epoch: 4 [44800/60032 (75%)]      Loss: 5.840479
Train Epoch: 4 [51200/60032 (85%)]      Loss: 5.841539
Train Epoch: 4 [57600/60032 (96%)]      Loss: 5.841094
Train Epoch: 5 [0/60032 (0%)]   Loss: 5.841490
Train Epoch: 5 [6400/60032 (11%)]       Loss: 5.840541
Train Epoch: 5 [12800/60032 (21%)]      Loss: 5.841216
Train Epoch: 5 [19200/60032 (32%)]      Loss: 5.841224
Train Epoch: 5 [25600/60032 (43%)]      Loss: 5.841165
Train Epoch: 5 [32000/60032 (53%)]      Loss: 5.840399
Train Epoch: 5 [38400/60032 (64%)]      Loss: 5.841606
Train Epoch: 5 [44800/60032 (75%)]      Loss: 5.841685
Train Epoch: 5 [51200/60032 (85%)]      Loss: 5.841044
Train Epoch: 5 [57600/60032 (96%)]      Loss: 5.842269
Train Epoch: 6 [0/60032 (0%)]   Loss: 5.840602
Train Epoch: 6 [6400/60032 (11%)]       Loss: 5.841238
Train Epoch: 6 [12800/60032 (21%)]      Loss: 5.841222
Train Epoch: 6 [19200/60032 (32%)]      Loss: 5.841230
Train Epoch: 6 [25600/60032 (43%)]      Loss: 5.840969
Train Epoch: 6 [32000/60032 (53%)]      Loss: 5.840241
Train Epoch: 6 [38400/60032 (64%)]      Loss: 5.841067
Train Epoch: 6 [44800/60032 (75%)]      Loss: 5.839836
Train Epoch: 6 [51200/60032 (85%)]      Loss: 5.841802
Train Epoch: 6 [57600/60032 (96%)]      Loss: 5.841057
Train Epoch: 7 [0/60032 (0%)]   Loss: 5.841918
Train Epoch: 7 [6400/60032 (11%)]       Loss: 5.840887
Train Epoch: 7 [12800/60032 (21%)]      Loss: 5.840362
Train Epoch: 7 [19200/60032 (32%)]      Loss: 5.840958
Train Epoch: 7 [25600/60032 (43%)]      Loss: 5.841733
Train Epoch: 7 [32000/60032 (53%)]      Loss: 5.840253
Train Epoch: 7 [38400/60032 (64%)]      Loss: 5.841130
Train Epoch: 7 [44800/60032 (75%)]      Loss: 5.840352
Train Epoch: 7 [51200/60032 (85%)]      Loss: 5.840876
Train Epoch: 7 [57600/60032 (96%)]      Loss: 5.839400
Train Epoch: 8 [0/60032 (0%)]   Loss: 5.840098
Train Epoch: 8 [6400/60032 (11%)]       Loss: 5.839633
Train Epoch: 8 [12800/60032 (21%)]      Loss: 5.839721
Train Epoch: 8 [19200/60032 (32%)]      Loss: 5.839624
Train Epoch: 8 [25600/60032 (43%)]      Loss: 5.841210
Train Epoch: 8 [32000/60032 (53%)]      Loss: 5.839372
Train Epoch: 8 [38400/60032 (64%)]      Loss: 5.840246
Train Epoch: 8 [44800/60032 (75%)]      Loss: 5.840254
Train Epoch: 8 [51200/60032 (85%)]      Loss: 5.841022
Train Epoch: 8 [57600/60032 (96%)]      Loss: 5.841519
Train Epoch: 9 [0/60032 (0%)]   Loss: 5.841502
Train Epoch: 9 [6400/60032 (11%)]       Loss: 5.841007
Train Epoch: 9 [12800/60032 (21%)]      Loss: 5.841004
Train Epoch: 9 [19200/60032 (32%)]      Loss: 5.839887
Train Epoch: 9 [25600/60032 (43%)]      Loss: 5.840875
Train Epoch: 9 [32000/60032 (53%)]      Loss: 5.839470
Train Epoch: 9 [38400/60032 (64%)]      Loss: 5.840583
Train Epoch: 9 [44800/60032 (75%)]      Loss: 5.839631
Train Epoch: 9 [51200/60032 (85%)]      Loss: 5.840472
Train Epoch: 9 [57600/60032 (96%)]      Loss: 5.840898
Train Epoch: 10 [0/60032 (0%)]  Loss: 5.841048
Train Epoch: 10 [6400/60032 (11%)]      Loss: 5.841439
Train Epoch: 10 [12800/60032 (21%)]     Loss: 5.840178
Train Epoch: 10 [19200/60032 (32%)]     Loss: 5.841186
Train Epoch: 10 [25600/60032 (43%)]     Loss: 5.840291
Train Epoch: 10 [32000/60032 (53%)]     Loss: 5.841321
Train Epoch: 10 [38400/60032 (64%)]     Loss: 5.840518
Train Epoch: 10 [44800/60032 (75%)]     Loss: 5.840454
Train Epoch: 10 [51200/60032 (85%)]     Loss: 5.839942
Train Epoch: 10 [57600/60032 (96%)]     Loss: 5.840398

---------- Testing ----------
Test set: Average loss: 58.4067, Accuracy: 9603/10000 (96%)

```
</details>

