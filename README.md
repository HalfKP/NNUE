# HalfKP
![Layers](https://raw.githubusercontent.com/EarlyEdition/NNUE/main/Layers.png)

The so called HalfKP structure consists of two halves covering input layer and first hidden layer, each half of the input layer associated to one of the two kings, cross coupled with the side to move or not to move halves of the first hidden layer. For each either black or white king placement, the 10 none king pieces on their particular squares are the boolean {0,1} inputs, along with a relict from Shogi piece drop (BONA_PIECE_ZERO), 64 x (64 x 10 + 1) = 41,024 inputs for each half, which are multiplied by a 16-bit integer weight vector for 256 outputs per half, in total, 256 x 41,024 = 10,502,144 weights. The input weights are arranged in such a way, that color flipped king-piece configurations in both halves share the same index.[[1]](http://www.talkchess.com/forum3/viewtopic.php?f=7&t=75506&start=7)

The efficiency of NNUE is due to incremental update of the input layer outputs in make and unmake move, where only a tiny fraction of its neurons need to be considered in case of none king moves. The remaining three layers with 2x256x32, 32x32 and 32x1 weights are computational less expensive, hidden layers apply a ReLu activation, best calculated using appropriate SIMD instructions performing fast 8-bit/16-bit integer vector arithmetic, like MMX, SSE2 or AVX2 on x86/x86-64, or AVX-512.

## Training Guide
### Generating Training Data
Use the "no-nnue.nnue-gen-sfen-from-original-eval" binary. The given example is generation in its simplest form. There are more commands. 
```
uci
setoption name Threads value x
setoption name Hash value y
isready
gensfen depth 8 loop 100000000
```
- `depth` is the searched depth per move, or how far the engine looks forward. This value is an integer.
- `loop` is the amount of positions generated. This value is also an integer.

Specify how many threads and how much memory you would like to use with the `x` and `y` values.
This will save a file named "generated_kifu.bin" in the same folder as the binary. Once generation is done, move the file to a folder named "trainingdata" in the same directory as the binaries.
### Generating validation data
The process is the same as the generation of training data, except for the fact that you need to set loop to 1 million, because you don't need a lot of validation data. The depth should be the same as before or a little higher than the depth of the training data. 
```
uci
setoption name Threads value x
setoption name Hash value y
isready
gensfen depth 8 loop 1000000
```
Once generation is done, move the file to a folder named "validationdata" in the same directory as the binaries.
### Training a completely new network
Use the "halfkp_256x2-32-32.nnue-learn" binary. Create an empty folder named "evalsave" in the same directory as the binaries.
```
uci
setoption name SkipLoadingEval value true
setoption name Threads value x
isready
learn targetdir trainingdata loop 100 batchsize 1000000 eta 1.0 lambda 0.5 eval_limit 32000 nn_batch_size 1000 newbob_decay 0.5 eval_save_interval 10000000 loss_output_interval 1000000 mirror_percentage 50 validation_set_file_name validationdata\generated_kifu.bin
```
- `eta` is the learning rate.
- `lambda` is the amount of weight it puts to eval of learning data vs win/draw/loss results.

Nets get saved in the "evalsave" folder. Copy the net located in the "final" folder under the "evalsave" directory and move it into a new folder named "eval" under the directory with the binaries.
