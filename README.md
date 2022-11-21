# MAT: Mask-Aware Transformer for Large Hole Image Inpainting (CVPR 2022 Best Paper Finalist, Oral)

This work is base on the MAT, but I remove the speical .cu .cpp files write for torch 1.7-1.9. It may have error when you try to run the model in some higher torch version. even in torch1.7, it also have some to compile the .cpp files. 
## some comperation
1. as I use the general operation, it may cost more memory of GPU, as I use the batchsize of 4 in 256x256, it cost 2-3 gig more compare to the origin.
2. It run slower, you may have to cost 25% more time to train the model.
3. the only advantage is you can use it at any version of pytorch(>1.7), and the result is all as same as the origin.