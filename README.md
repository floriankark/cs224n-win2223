# <p align="center">CS224N: Natural Language Processing with Deep Learning</p>
## <p align="center">[Stanford / Winter 2023](http://web.stanford.edu/class/cs224n/index.html)</p>
Walkthrough of the schedule and solutions of the assignments of the Stanford CS224N: Natural Language Processing with Deep Learning course from winter 2022/23. If you come across any errors, please let me know at florian.kark@hhu.de

Reading papers is an important part of this course and crucial for completing the assignments successfully. Therefore I recommend to have a look at [How to read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)

## My Schedule

**Apr.18.2023**
- watch [Lecture 1](https://youtu.be/rmVRLeJRkl4) and [Lecture 2](https://youtu.be/gqaHkPEZAew)
- read [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) and [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- finish [assignment 1](https://github.com/floriankark/cs224n-win2223/tree/main/a1)
- go through [Python Review Session](https://colab.research.google.com/drive/1hxWtr98jXqRDs_rZLZcEmX_hUcpDLq6e?usp=sharing) ([slides](http://web.stanford.edu/class/cs224n/readings/cs224n-python-review-2023.pdf))
 
 **Apr.19.2023**
- read [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf), [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016) and [Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036)
- watch [Lecture 3](https://youtu.be/X0Jw4kgaFlg) and [Lecture 4](https://youtu.be/PSGIodTN3KE)

 **Apr.20.2023**
- read [matrix calculus notes](http://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf), [Review of differential calculus](http://web.stanford.edu/class/cs224n/readings/review-differential-calculus.pdf), [CS231n notes on network architectures](http://cs231n.github.io/neural-networks-1/), [CS231n notes on backprop](http://cs231n.github.io/optimization-2/), [Derivatives, Backpropagation, and Vectorization](http://cs231n.stanford.edu/handouts/derivatives.pdf) and [Learning Representations by Backpropagating Errors](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)

**Apr.21.2023**

- read [Understanding word vectors](https://gist.github.com/aparrish/2f562e3737544cf29aaf1af30362f469) (my own suggestion, not included in original cs224n)
- finish [assignment 2 written](https://github.com/floriankark/cs224n-win2223/tree/main/a2_written)

**Apr.22.2023**

- read "additional readings" [A Latent Variable Model Approach to PMI-based Word Embeddings](http://aclweb.org/anthology/Q16-1028), [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320), [On the Dimensionality of Word Embedding](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf), [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b), [Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

**Apr.23.2023**

- finish [assignment 2](https://github.com/floriankark/cs224n-win2223/tree/main/a2)
- watch [Lecture 5](https://youtu.be/PLryWeHPcBs) and [Lecture 6](https://youtu.be/0LixFSa7yts)

**Apr.24.2023**

- complete [PyTorch Tutorial](https://colab.research.google.com/drive/13HGy3-uIIy1KD_WFhG4nVrxJC-3nUUkP?usp=sharing)
- read [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) (textbook chapter), [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (blog post overview), [Sequence Modeling: Recurrent and Recursive Neural Nets](http://www.deeplearningbook.org/contents/rnn.html) (Sections 10.1 and 10.2), [On Chomsky and the Two Cultures of Statistical Learning](http://norvig.com/chomsky.html), [Sequence Modeling: Recurrent and Recursive Neural Nets](http://www.deeplearningbook.org/contents/rnn.html) (Sections 10.3, 10.5, 10.7-10.12), [Learning long-term dependencies with gradient descent is difficult](http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf) (one of the original vanishing gradient papers), [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/pdf/1211.5063.pdf) (proof of vanishing gradient problem), [Vanishing Gradients Jupyter Notebook](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/lectures/vanishing_grad_example.html) (demo for feedforward networks), [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (blog post overview) 

**Apr.25.2023**

- read papers from assignment 3 [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf), [Tricks from the actual Adam update](https://cs231n.github.io/neural-networks-3/#sgd), [Dropout: A Simple Way to Prevent Neural Networks from
Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

**Apr.26.2023**

- finish [assignment 3 written](https://github.com/floriankark/cs224n-win2223/tree/main/a3_written)

**Apr.27.2023**

- read [An Explanation of Xavier Initialization](https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization), 
- PyTorch documentation [nn.Parameters](https://pytorch.org/docs/stable/nn.html#parameters), [Initialization](https://pytorch.org/docs/stable/nn.init.html), [Dropout](https://pytorch.org/docs/stable/nn.html#dropout-layers), [Index select](https://pytorch.org/docs/stable/torch.html#torch.index_select), [Gather](https://pytorch.org/docs/stable/torch.html#torch.gather), [View](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view), [Flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html), [Matrix product](https://pytorch.org/docs/stable/torch.html#torch.matmul), [ReLU](https://pytorch.org/docs/stable/nn.html?highlight=relu#torch.nn.functional.relu), [Adam Optimizer](https://pytorch.org/docs/stable/optim.html), [Cross Entropy Loss](https://pytorch.org/docs/stable/nn.html#crossentropyloss), [Optimizer Step](https://pytorch.org/docs/stable/optim.html#optimizer-step)
- finish [assignment 3](https://github.com/floriankark/cs224n-win2223/tree/main/a3)

**Apr.28.2023**

- watch [Lecture 7](https://youtu.be/wzfWHP6SXxY) and [Lecture 8](https://youtu.be/gKD7jPAdbpE)
- read [Statistical Machine Translation slides, CS224n 2015](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1162/syllabus.shtml) (lectures 2/3/4), [Statistical Machine Translation](https://www.cambridge.org/core/books/statistical-machine-translation/94EADF9F680558E13BE759997553CDE5) (book by Philipp Koehn), [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) (original paper), [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf) (original seq2seq NMT paper), [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/pdf/1211.3711.pdf) (early seq2seq speech recognition paper), [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) (original seq2seq+attention paper), [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/) (blog post overview), [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf) (practical advice for hyperparameter choices), [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](https://arxiv.org/abs/1604.00788.pdf), [Revisiting Character-Based Neural Machine Translation with Capacity and Compression](https://arxiv.org/pdf/1808.09943.pdf)

**Mai.6.2023**

- read [Embedding Layer](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding), [LSTM](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM), [LSTM Cell](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell), [Linear Layer](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear), [Dropout Layer](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout), [Conv1D Layer](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)

- finish [assignment 4](https://github.com/floriankark/cs224n-win2223/tree/main/a4)

**Mai.7.2023**

- finish [assignment 4 written](https://github.com/floriankark/cs224n-win2223/tree/main/a4_written)

**Mai.8.2023**

- watch [Lecture 9](https://youtu.be/ptuGllU5SQQ)
- read [Attention Is All You Need](https://arxiv.org/abs/1706.03762.pdf), [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), [Transformer](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) (Google AI blog post), [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf), [Image Transformer](https://arxiv.org/pdf/1802.05751.pdf), [Music Transformer: Generating music with long-term structure](https://arxiv.org/pdf/1809.04281.pdf)

**Mai.9.2023**

- watch [Lecture 10](https://youtu.be/j9AcEI98C0o)
