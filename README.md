# Humans

... enacting Black Mirror's "[Be Right Back](https://www.imdb.com/title/tt2290780/)"?

# Setup

Prepare virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install requirements:

```bash
pip install -U pip
pip install -r requirements.txt
```

**Preparing data.**
To set up a dataset, define folders for `filelists`, `video` and `face-landmarks`.
As an example see the GRID dataset at the following location (on the `lenovo` machines):
```bash
ls /home/doneata/work/humans/data/grid
```

**Preparing models.**
The base path for models is
```bash
ls /home/doneata/work/humans/models
```

# Main functionality

Extract face landmarks:

```bash
python scripts/extract_face_landmarks.py --dataset grid --filelist tiny --n-cpu 4 -v
```

# Contributing

**Code style.**
Please use the [black](https://github.com/psf/black) code formatter to ensure uniform code style.

**Typing.**
Use [type annotations](https://docs.python.org/3/library/typing.html) where it is sensible to do so and check with [mypy](https://github.com/python/mypy), for example:
```bash
mypy scripts/extract_face_landmarks.py
```

# Technical overview

This section presents an overview of the task and our approach.

**Task.**
We want to automatically map the audio of a person speaking to the corresponding lip movement.

**Architecture.**
The two signals (audio and lip movements) are aligned (modulo discretization).
This means we can potentially use any of the popular architectures (such as recurrent neural networks, convolutional networks, attention-based networks with possibly diagonally-constrained attention).
If we are interested in an online scenario, maybe recurrent networks are the simplest variant, although the others can certainly be amended for such a scenario;
if online prediction wouldn't be an issue, I would go with a Transformer architecture.
I would predict the output in an auto-regressive manner to help smooth out the lip movements.

**Representing audio.**
It might be worthwhile to take inspiration from audio encoders in ASR models (see for example [ESPNet](https://github.com/espnet/espnet)).
Might be possible to leverage their pre-trained models as well.

**Representing lips.**
There are multiple ways of representing lip landmarks;
here are some options:
- Heatmaps, one for each landmark.
(See work on 2d pose estimation: [convolutional pose mahcines](https://arxiv.org/abs/1602.00134) or [stacked hourglass networks](https://arxiv.org/abs/1603.06937).)
- Low-dimensional projection of the concatenated positions.
We will probably need to center and normalize the positions.
See the Obamanet paper or the paper by [Suwajanakorn et al. (2017)](https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf), in the references:

> To compute the mouth shape representation, we first detect and frontalize Obama’s face in each video frame using the approach in [Suwajanakorn et al., 2014].
> For each frontalized face, we detect mouth landmarks using [Xiong and De la Torre 2013] which gives 18 points along the outer and inner contours of the lip.
> We reshape each 18-point mouth shape into a 36-D vector, apply PCA over all frames, and represent each mouth shape by the coefficients of the first 20 PCA coefficients; this step both reduces dimensionality and decorrelates the resulting feature set.
> Finally, we temporally upsample the mouth shape from 30 Hz to 100 Hz by linearly interpolating PCA coefficients, to match the audio sampling rate.
> Note that this upsampling is only used for training.

**Extracting face landmarks.**
Currently, I'm using [dlib](http://dlib.net/),
Dragoș also suggested [this library](https://github.com/1adrianb/face-alignment).
Maybe try both for a few videos and check differences and timings.

**Data.**
What datasets should we use for training the model?
We need pairs of audio and front video recordings;
there should be plenty of such datasets.
For example, we have access to the Lip Reading in the Wild dataset (although non-commercial conditions might apply).
But if multiple speakers are present in the training dataset will we need to encode the speaker identity (in order to account for mouth shape variability)?
Or is this information (the speaker identity) already present in the input audio stream?
To explicitly circumvent the speaker variability, I would be very tempted to use a single-person dataset (_e.g._, Obama for which there are many video recording available).
Although, on a second thought, this approach has the disadvantage of being specific to a single type of voice.

_Q_ Business-wise is better to target a single speaker or does the method need to work for any speaker from the beginning?

| dataset | num. hours | num. speakers | status     | observations | link |
|---------|------------|---------------|------------|--------------|------|
| GRID    |            |               | downloaded | limited vocabulary, constrained conditions | |
| LRW     |            |               | downloaded |              | [link](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) |
| LRS2    |            |               | TODO       |              | [link](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) |
| LRS3    |            |               | TODO       |              | [link](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html) |
| Lip2Wav | 100        | 5             | TODO       |              | [link](https://cove.thecvf.com/datasets/363) |

**Intermediate phonetic representation.**
The model goes between two aligned signals: from audio to lips.
We could explicitly enforce an intermediate layer to correspond to the phonetic representation,
but what advantages would bring such a design? Maybe better interpretability?
In my opinion it would be nice to be able to have choose from two possible input modalities—audio or text—
but adapting the architecture for this might complicate it too much.

**An experiment in phoneme-to-lip mapping.**
A straightforward way of mapping phonemes to lip movements (_visemes_) is by aligning the phonetized transcription to the audio of the video.
At test time, we can generate lip movements for a new transcription by sequencing the corresponding visemes.

Here are some [initial results](http://speed.pub.ro/xts/backup-2020-05-20) that show 128 samples of lip movements for twelve phones (vowels):
AA, AE, AH, AO, AW, AY, EH, EY, IH, IY, OW, UW.
For this experiment, we have used the [GRID dataset](https://pubmed.ncbi.nlm.nih.gov/17139705/) and
obtained the alignments by using a pre-trained automatic speech recognition (ASR) model trained on TED-LIUM with the Kaldi framework
(but we can also use alignments extracted with [Gentle](https://github.com/lowerquality/gentle)).
faces and their landmarks were detected using the [`dlib` toolkit](http://dlib.net/).

# References

- Suwajanakorn, Supasorn, Steven M. Seitz, and Ira Kemelmacher-Shlizerman. "Synthesizing Obama: learning lip sync from audio." ACM Transactions on Graphics (ToG) 36.4 (2017): 1-13. [pdf](https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf)
- Kumar, Rithesh, et al. "Obamanet: Photo-realistic lip-sync from text." arXiv preprint arXiv:1801.01442 (2017). [pdf](https://arxiv.org/pdf/1801.01442.pdf) · [code](https://github.com/karanvivekbhargava/obamanet) (original implementation) · [code](https://github.com/acvictor/Obama-Lip-Sync) (separate implementation)
- Fried, Ohad, et al. "Text-based editing of talking-head video." ACM Transactions on Graphics (TOG) 38.4 (2019): 1-14. [pdf](https://dl.acm.org/doi/pdf/10.1145/3306346.3323028)
- Prajwal, K. R., et al. "A lip sync expert is all you need for speech to lip generation in the wild." Proceedings of the 28th ACM International Conference on Multimedia. 2020. [pdf](http://cvit.iiit.ac.in/research/projects/cvit-projects/a-lip-sync-expert-is-all-you-need-for-speech-to-lip-generation-in-the-wild/)
