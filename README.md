### PEGASUS library

Pre-training with Extracted Gap-sentences for Abstractive SUmmarization
Sequence-to-sequence models, or PEGASUS, uses self-supervised objective Gap
Sentences Generation (GSG) to train a transformer encoder-decoder model. The
paper can be found on [arXiv](https://arxiv.org/abs/1912.08777). ICML 2020 accepted.
check code source from [here](https://github.com/google-research/pegasus).


!['screen'](https://1.bp.blogspot.com/-TSor4o51jGI/Xt50lkj6blI/AAAAAAAAGDs/TrDe9jv13WEwk9NQNebQL63jtY8n6JFGwCLcBGAsYHQ/s640/image1.gif)


### Prerequisites
```
Python 3+
tensorflow==2.2.0
sentencepiece
numpy
```

### Usage

To run the summery, download pre-trained model on cnn_dailymail from [here](https://drive.google.com/file/d/1FVzZto4bf5_TCmRy3tNeirhPDdLrvum5/view?usp=sharing) or gigaword from [here](https://drive.google.com/file/d/1ZF2qO6bAnsTF2LSndLMir3e7NrlFL288/view?usp=sharing). Unzip it and put it to `model/`.

`python scripts/summery.py --article example_article --model_dir model/ --model_name cnn_dailymail`

### Finetuning Dataset

Two types of dataset format are supported: TensorFlow Datasets (TFDS) or TFRecords.
The [pn-summary dataset](https://github.com/hooshvare/pn-summary) can be used for this purpose. pn-summary comprises numerous articles of various categories that have been crawled from six news agency websites. Each document (article) includes the long original text as well as a human-generated summary.

### To Do

- [ ] Collab demo
- [ ] fine-tune on persian dataset
