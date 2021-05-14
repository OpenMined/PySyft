# Natural Language Inference (SNLI) with GloVe vectors, LSTMs, and torchtext

## Usage
```bash
python train.py
```
### Details
```bash
python train.py [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--d_embed D_EMBED]
        [--d_proj D_PROJ] [--d_hidden D_HIDDEN] [--n_layers N_LAYERS] [--log_every LOG_EVERY]
        [--lr LR] [--dev_every DEV_EVERY] [--save_every SAVE_EVERY] [--dp_ratio DP_RATIO]
        [--gpu GPU] [--save_path SAVE_PATH] [--vector_cache VECTOR_CACHE]
        [--word_vectors WORD_VECTORS] [--resume_snapshot RESUME_SNAPSHOT] [--dry-run DRY-RUN]
        [--no-bidirectional] [--preserve-case] [--no-projection] [--train_embed]

    --epochs            the number of total epochs to run; default: 50
    --batch_size        batch size; default: 128
    --d_embed           the size of each embedding vector; default: 100
    --d_proj            the size of each projection layer; default: 300
    --d_hidden          the number of features in the hidden state; default: 300
    --n_layers          the number of recurrent layers; default: 50
    --log_every         iteration period to output log; default: 50
    --lr                initial learning rate; default: 0.001
    --dev_every         log period of validation results; default: 1000
    --save_every        model checkpoint period; default: 1000
    --dp_ratio          probability of an element to be zeroed; default: 0.2
    --gpu               gpu id to use; default: 0
    --save_path         save path of results; default: "results"
    --vector_cache      name of vector cache directory, which saved input word-vectors;
                        default: PROJDIR/".vector_cache/input_vectors.pt")
    --word_vectors      one of or a list containing instantiations of the GloVe, CharNGram, or Vectors classes;
                        default: glove.6B.100d
                        #Alternatively, one of or a list of available pretrained    vectors:
                        #charngram.100d fasttext.en.300d fasttext.simple.300d
                        #glove.42B.300d glove.840B.300d glove.twitter.27B.25d
                        #glove.twitter.27B.50d glove.twitter.27B.100d   glove.twitter.27B.200d
                        #glove.6B.50d glove.6B.100d glove.6B.200d glove.6B.300d;
    --resume_snapshot   model snapshot to resume; default:""
    --dry-run           run only a few iterations; default: False
    --no-bidirectional  disable bidirectional LSTM
    --preserve-case     case-sensitivity
    --no-projection     disable projection layer
    --train_embed       enable embedding word training
```
