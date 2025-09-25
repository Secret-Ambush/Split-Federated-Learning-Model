Impact of backdoor attackers on SplitFL

Find the cut layers for the standaradised models
Check where the split is easy and possible
Three to Four of them
Three cut layers set up 
Extreme cases - whole model at client, one block at client - rest at server

Number of clients - at least 10 (low), 20
Dataset - CIFAR10, one more (MNIST, ImageNET (truncated))
Traffic Analysis Data - Stop signs, traffic signs (10,15 classes - 1k+ images across all)

JUST REPORT THE NUMBERS (As a Table)
WITHOUT ATTACKER
5 Models, let's say 4 cut layers - extreme + cut layers (20 configs)
Result in terms of accuracy - Impact of cutlayer without attackers
Training - non-IID

WITH BACKDOOR ATTACKERS
Introduction of attacker
0,10,20,30% attackers with static, size, pattern, random across all
Training - non-IID
ASR, Backdoor accuracy, Clean accuracy


1) Write down the tables in Overleaf