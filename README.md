
# Meliad

***This is not an officially supported Google product.***

_This code is provided "as-is" to the broader research community.  Google
does not promise to maintain or otherwise support this code in any way._

## Introduction

The Meliad library is collection of models which are being developed as part
of ongoing research into various architectural improvements in deep learning.
The name "meliad" is the Greek word for a tree nymph; a long-term goal of
this research is to design architectures that can understand recursive and
compositional structures, i.e. trees.

The library currently consists of several transformer variations, which explore
ways in which the popular transformer architecture can be extended to better
support language modeling over long sequences.

#### Transformer-XL with sliding window

This model is provided as a baseline.  It is similar to the [Transformer-XL
architecture](https://arxiv.org/abs/1901.02860), but uses a T5-style relative
position bias.  A long sequence, such as a book, is divided into segments of
fixed length, e.g. 4096 tokens.  The segments are processed in order, with one
segment per training step.

Attention within a segment is done locally using _sliding window_ that is
typically smaller than the segment length. A causal mask ensures that each
token can attend to exactly _W_ previous tokens, where _W_ is the window size,
e.g. 512 or 1024. The complexity of attention is quadratic with respect to
window size, but linear with respect to segment length, so the segment length 
is limited only by available device memory.
Like Transformer-XL, the model caches the keys and values from the last window
for use on the next training step, and thus implements truncated backpropagation
through time over very long (book-length) works.

If the window and segment lengths are the same, then there is no sliding window
(just the T-XL cache), and this model will behave like Transformer-XL.  However,
the cache is not differentiable, whereas the sliding window is, so there is
some benefit to using segments that are longer than the window length.
Gradients with the sliding window can potentially be backpropagated across the
length of the entire segment.

#### Memorizing Transformer

The [Memorizing Transformer](http://arxiv.org/abs/2203.08913) equips one layer
of the transformer with a large external memory that stores prior (key,value)
pairs.  Typical memory sizes are 32k or 64k tokens.  In addition to local
attention, the model can do k-nearest-neighbor lookup into the external memory,
which allows it to handle long-range dependencies; the range is limited only by
the size of the memory.

The external memory, like the T-XL cache, is not differentiable.  Memory and
the T-XL cache work well together; the memory is used for long-range lookups,
while the cache is used for short-range lookups.
However, memory should not be used with a sliding window, so the window and
segment length should be the same.



#### Block-Recurrent Transformer

The [Block-Recurrent Transformer](https://arxiv.org/abs/2203.07852) equips one
layer of the transformer with a recurrent cell.  The cell is structured
similarly to an LSTM cell, but it is several orders of magnitude larger, and
operates on _blocks_ of tokens and _blocks_ of recurrent state vectors.
Recurrence is integrated with the sliding window mechanism; the block size is
the same as the window size.

Recurrence serves a similar role to external memory, but is faster.  The
recurrent state has a fixed capacity, but unlimited range (in theory).


## Installation instructions

Create an activate a python virtual environment.
(Commands given are for linux).

```
python -m venv my_env
source my_env/bin/activate
```

Install required packages into the python virtual environment.  If you want to
use GPUs, then Jax must be upgraded to use CUDA.

```
pip install -r requirements.txt
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

On Unix systems, you may need to ensure that `PYTHONPATH` includes the
current directory.  All module names are given relative to the meliad root.

```
export PYTHONPATH=.:$PYTHONPATH
```

Run a small baseline model on a synthetic test dataset.

```shell
python transformer/ht_main.py --alsologtostderr \
--gin_file=base_htrans.gin \
--gin_file=size/small_test.gin
```

## Configuring and running the model

Meliad uses [gin](https://github.com/google/gin-config) to configure the model.
The first gin file should always be
`base_htrans.gin`, which supplies a default configuration.  Other options are
specified as additional files in the configs directory.  Most options are
orthogonal, but in some cases the order matters; inspect the contents of the
gin files to determine the correct order.

Some important options are:

- `size/medium150M.gin`  The 150M parameter model in the paper.
- `options/positions_t5.gin` Use a T5-style relative position bias.
- `options/seq_4096.gin` Use a segment length of 4096 tokens.
- `options/window_1024.gin` Use a sliding window of size 1024.
  (The default is 512).
- `options/lr_cosine_decay.gin` Cosine decay learning rate schedule.

Tasks are also defined in gin files:

- `tasks/pg19_tokens.gin` Run on PG19 with the default T5 sentencepiece
  vocabulary.

Other important command-line options:

- `--alsologtostderr` View the progress of the model.
- `--workdir=/my/work/directory` For checkpoints and tensorboard.
- `--load_dir=/location/of/pretrained/model` For finetuning.
- `--default_data_dir=/location/of/tfds/datasets` For tensorflow datasets.

For the Memorizing Transformer:

- `size/medium150M.gin`  The 150M parameter model in the paper.
- `options/positions_t5.gin` Use a T5-style relative position bias.
- `options/seq_512.gin` Segment length of 512.  (Window is 512 by default).
- `options/external_memory_32k.gin` Memorizing Transformer with a memory
  size of 32k.

For the Block-Recurrent Transformer:

- `size/medium150M.gin`  The 150M parameter model in the paper.
- `options/positions_t5.gin` Use a T5-style relative position bias.
- `options/seq_4096.gin` Segment length of 4096.  (Window is 512 by default).
- `recurrent/bias_skip.gin` The fixed:skip configuration.

