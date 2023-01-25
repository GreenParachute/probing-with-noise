
# Probing with Noise Implementation 

This is the official implementation of the following paper:

[Filip Klubička](https://twitter.com/lemoncloak) and John D. Kelleher. [Probing with Noise: Unpicking the Warp and Weft of Embeddings](https://arxiv.org/pdf/2210.12206.pdf). *Proceedings of the BlackBoxNLP 2022 Workshop (EMNLP2022).*

## Watch The Video

[![Watch the video](https://img.youtube.com/vi/rNp1Xe6gtBE/maxresdefault.jpg)](https://youtu.be/rNp1Xe6gtBE)

## Using and understanding the provided scripts

### Requirements

We recommend installing all requirements in a Conda virtual environment using the below commands:

* `conda install pytorch torchvision torchaudio -c pytorch`
* `conda install -c conda-forge scikit-learn`
* `pip install pytorch-pretrained-bert` (not available through conda)

If you prefer to install using pip, you can install the required libraries using:

* `pip install -r requirements.txt`

This code is confirmed to work with Python 3.9 on MacOS Ventura 13.1 and Ubuntu 20.04.
The code is written to train on a CPU, no GPU training is required.

### Run a full probing with noise scenario:

The `main.py` script will train an MLP probing classifier on the provided dataset and evaluate its performance on the given task. Based on the arguments provided at runtime, it will also introduce noise into the provided sentence vectors in order to disrupt the encoding and isolate the contribution of either the vector norm or the vector dimensions to the probe's performance on the task. 

The possible options and corresponding arguments are:

* -p, --path : Specify path to your data file, i.e. a pickle file containing a dictionary with candidate words/sentences, their embeddings and class labels.
* -c, --classes : Type of classification task performed. Options: binary, multiclass. Default: binary
* -e, --embedding : Specifies which type of encoder architecture generated the 'fixed' embeddings you want to load from the dataset file and use for training. Specifying bert_train will train new bert embeddings on new sentences, rather than load pretrained embeddings from the dictionary. Options: glove, bert, bert_train. Default: bert
* -n, --noise : Type of noise injections/embedding modifications to be performed before training. Options: rvec, vanilla, abn, abd, abnd, d1h, d2h. Default: vanilla (see meaning below)
* -b, --baseline : Instead of training on the input vectors, using this flag evaluates a baseline model that makes random label predictions over the given dataset.
* -r, --runs : Number of times the model will be trained. Default: 50

Types of noise/vector modifications:

* rvec : generates a vector with random dimension values, replacing the provided pre-trained vector (acts as a performance baseline)
* vanilla : makes no modification and uses the provided vector as is
* abn : ablates the information encoded in the vector norm, by replacing the vector's norm with a randomly generated norm and then scaling the vector's dimensions to match the new norm
* abd : ablates the information encoded in the vector dimensions, by replacing the vector's dimensions with randomly generated dimensions and then scaling them to match the vector's original norm
* abnd : applies the two above functions to the same vector, effectively generating a random vector (this is a sense-check, it should always perform comparably to rvec) 
* d1h : deletes the first half of the vector dimensions
* d2h : deletes the second half of the vector dimensions

When all the possible noise disruptions are run sequentially, the sequence of evaluation scores provides an evaluation framework that sheds light on how the given linguistic information is encoded in vector space.

Some example commands:

`python main.py --path data/bigram_shift_small.pkl --embedding bert --noise abd --classes binary --runs 20`

The above will load the 'frozen' BERT embeddings from the dataset, ablate the dimensions in each vector and then use these modified vectors them to train a binary probing classifier 20 times. 

`python main.py --path data/top_constituents_small.pkl --embedding glove --noise vanilla --classes multiclass --runs 45`

The above will load the 'frozen' GloVe embeddings from the dataset, will not apply any modification to them vectors and will then use them to train a multi-class probing classifier 45 times.

It is also worth noting that, if you have your own dataset of annotated English sentences, but do not have embeddings, the code provides a functionality to generate BERT sentence embeddings during training, by using the `--embedding bert_train` flag. This take the English sentence at input, embed it using BERT, and then apply the chosen noising function and continue as normal. This will however take significantly longer to run, and if running multiple iterations each iteration will re-train the BERT sentence embeddings, so it might be wise to save them locally to speed up future training.

If using 'frozen' embeddings, the runtime depends on the power of your CPU. Using the small toy datasets to train and evaluation a single iteration (`--runs 1`) usually takes under a minute, while using the full-sized datasets can take significantly longer--anything between 30 minutes and several hours--and heavily depends on the chosen task, embeddings and noise functions. 

### Full Dataset

Running our probing experiments requires an annotated probing task dataset formated as a dictionary containing several lists of dictionaries that contain a sentence, sentence label and corresponding sentence vector(s), as illustrated below.

For convenience, the `data` folder provide toy sample datasets in the required format, containing already embedded sentences for the `bigram_shift` binary classification task and the `top_constituents` multiclass classification task. 

It is worth noting that these toy datasets are not large enough to learn the task and reproduce our experimental results. You can download a full instance of the `bigram_shift` dataset [here](https://drive.google.com/file/d/1V3ZQLKaxTUSEIy6okqnR9dbPYpJlyk1T/view?usp=sharing), and a full instance of the `top_constituents` dataset [here](https://drive.google.com/file/d/1LdzDVaEyoUb5xWPLUlWXAnxnUOXtls7m/view?usp=sharing), both of which, which include BERT ang GloVE sentence embeddings, in addition to natural language sentences and their labels. Simply download them and place them in the `data` folder to be used with the code.

Both the toy and full datasets datasets are derived from the Conneau et al. (2018) [probing task datasets](https://github.com/facebookresearch/SentEval/blob/main/data/probing/), as described in their paper [What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties](https://arxiv.org/pdf/1805.01070.pdf). If you use our modified dataset or any of the other datasets provided by Conneau et al., in your research, please also cite their work.

### Required Dataset Format

```
{
  'train_set' :       [ 
                        {
                          'sent' : "Gretchen envied his relaxed , worry-free existence .',
                          'lab_str' : "O",
                          'bert' : array([-1.19660884e-01, -3.77814658e-02, ... , -3.39492172e-01,  1.08322740e-01], dtype=float32)
                          'glove' : array([1.56192994e+00,  7.78697953e-02, ... , -4.76773977e-02,  8.41352344e-01], dtype=float32)
                        },
                        {
                          'sent' : "..."
                          'lab_str' : "..."
                          'bert' : array(...)
                          'glove' : array(...)
                        }
                      ],

  'test_set' :        [
                        {
                          ...
                        },
                        ...,
                        {
                          ...
                        }
                      ],

  'validation_set' :  [
                        {
                          ...
                        },
                        ...,
                        {
                          ...
                        }
                      ],
}
```

# Citation:

If you use any of the above code or data in your research, please cite the following paper(s):

```
@inproceedings{klubicka-kelleher-2022-probing,
    title = "Probing with Noise: Unpicking the Warp and Weft of Embeddings",
    author = "Klubi\v{c}ka, Filip and Kelleher, John D.",
    booktitle = "Proceedings of the Fifth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/pdf/2210.12206.pdf",
}
```
You can download the paper [here](https://arxiv.org/pdf/2210.12206.pdf).

```
@phdthesis{klubicka-2022-thesis,
  title={Probing with Noise: Unpicking the Warp and Weft of Taxonomic and Thematic Meaning Representations in Static and Contextual Embeddings},
  author={Klubi{\v{c}}ka, Filip},
  year={2022},
  school={Technological University Dublin}
}
```
You can download the thesis [here](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1264&context=sciendoc).

This work has been funded by a grant from Science Foundation Ireland: Grant Number 13/RC/2106 and 13/RC/2106_P2.

# Licensing information:

Copyright © 2023 Filip Klubička. Technological University Dublin, ADAPT Centre.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
