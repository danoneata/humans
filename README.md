# Humans

... enacting Black Mirror's "[Be Right Back](https://www.imdb.com/title/tt2290780/)"?

# Task

Our goal is to map the audio of a person speaking to the corresponding lip movement.

**Architectures.**
The two signals (audio and lip movements) are aligned (modulo discretization).
This means we can use any of the popular architectures (such as recurrent neural networks, convolutional networks, attention-based networks with possibly diagonally-constrained attention).
If we are interested in an online scenario, maybe recurrent networks are the simplest variant, although the others can certainly be used for such a scenario.
I would use an auto-regressive output to help smooth out the lip movements.

**Representing lips.**
There are multiple possibilities of representing lip landmarks;
here are some options:
- Heatmaps, one for each landmark.
(See work on 2d pose estimation.)
- Low-dimensional projection of the concatenated positions.
We will probably need to center and normalize the positions.
(See the Obamanet paper.)

**Data.**
What datasets should we use for training the model?
We need pairs of audio and front video recordings;
there should be plenty of such datasets.
For example, we have access to the Lip Reading in the Wild dataset (although non-commercial conditions might apply).
But if multiple speakers are present in the training dataset will we need to encode the speaker identity (in order to account for mouth shape variability)?
Or is this information (the speaker identity) already present in the input audio stream?
To explicitly circumvent the speaker variability, I would be very tempted to use a single-person dataset (_e.g._, Obama for which there are many video recording available).

**Intermediate phonetic representation.**
The model goes between two aligned signals: from audio to lips.
We could explicitly enforce an intermediate layer to correspond to the phonetic representation,
but what advantages would bring such a design? Maybe better interpretability?
In my opinion it would be nice to be able to have choose from two possible input modalities—audio or text—
but adapting the architecture for this might complicate it too much.

**An experiment in phoneme to lip movement mapping.**
A straightforward way of mapping phonemes to lip movements (_visemes_) is by aligning the phonetized transcription to the audio of the video.
At test time, we can generate lip movements for a new transcription by sequencing the corresponding visemes.

Here are some [initial results](http://speed.pub.ro/xts/backup-2020-05-20) that show 128 samples of lip movements for twelve phones (vowels):
AA, AE, AH, AO, AW, AY, EH, EY, IH, IY, OW, UW.
For this experiment, we have used the [GRID dataset](https://pubmed.ncbi.nlm.nih.gov/17139705/) and
obtained the alignments by using a pre-trained automatic speech recognition (ASR) model trained on TED-LIUM with the Kaldi framework
(but we can also use alignments extracted with [Gentle](https://github.com/lowerquality/gentle)).
faces and their landmarks were detected using the [`dlib` toolkit](http://dlib.net/).

# References

- Kumar, Rithesh, et al. "Obamanet: Photo-realistic lip-sync from text." arXiv preprint arXiv:1801.01442 (2017). [pdf](https://arxiv.org/pdf/1801.01442.pdf)
- Fried, Ohad, et al. "Text-based editing of talking-head video." ACM Transactions on Graphics (TOG) 38.4 (2019): 1-14. [pdf](https://dl.acm.org/doi/pdf/10.1145/3306346.3323028)
- Prajwal, K. R., et al. "A lip sync expert is all you need for speech to lip generation in the wild." Proceedings of the 28th ACM International Conference on Multimedia. 2020. [pdf](http://cvit.iiit.ac.in/research/projects/cvit-projects/a-lip-sync-expert-is-all-you-need-for-speech-to-lip-generation-in-the-wild/)
