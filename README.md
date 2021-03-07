# Humans

... enacting Black Mirror's "[Be Right Back](https://www.imdb.com/title/tt2290780/)"?

## Phoneme to lip movement

A straightforward way of mapping phonemes to lip movements (_visemes_) is by aligning the phonetized transcription to the audio of the video.
At test time, we can generate lip movements for a new transcription by sequencing the corresponding visemes.

Here are some [initial results](http://speed.pub.ro/xts/backup-2020-05-20) that show 128 samples of lip movements for twelve phones (vowels):
AA, AE, AH, AO, AW, AY, EH, EY, IH, IY, OW, UW.
For this experiment, we have used the [GRID dataset](https://pubmed.ncbi.nlm.nih.gov/17139705/) and
obtained the alignments by using a pre-trained automatic speech recognition (ASR) model trained on TED-LIUM with the Kaldi framework;
faces and their landmarks were detected using the [`dlib` toolkit](http://dlib.net/).
We also have access to the Lip Reading in the Wild dataset (although non-commercial conditions might apply) and we can also use alignments extracted with Gentle.

**Representing lips.**
Here are some options of representing lip landmarks:

- Heatmaps, one for each landmark.
(See work on 2d pose estimation.)
- Low-dimensional projection of the concatenated positions.
We will probably need to center and normalize the positions.
(See the Obamanet paper.)

**Multi-speaker datasets.**
If multiple speakers are present in the training dataset will we need to encode the speaker identity (in order to account for mouth shape variability)?
Or is this information (the speaker identity) already present in the input audio stream?

**Intermediate phoneme representation.**
The model goes from audio to lips;
we could explicitly enforce an intermediate layer to correspond to the phonetic.
But what advantages such a design would bring? Better interpretability?
In my opinion it would be nice to be able to have choose from two possible input modalities—audio or text—
but such a feature might complicate the architecture too much.

## References

- Kumar, Rithesh, et al. "Obamanet: Photo-realistic lip-sync from text." arXiv preprint arXiv:1801.01442 (2017). [pdf](https://arxiv.org/pdf/1801.01442.pdf)
- Fried, Ohad, et al. "Text-based editing of talking-head video." ACM Transactions on Graphics (TOG) 38.4 (2019): 1-14. [pdf](https://dl.acm.org/doi/pdf/10.1145/3306346.3323028)
- Prajwal, K. R., et al. "A lip sync expert is all you need for speech to lip generation in the wild." Proceedings of the 28th ACM International Conference on Multimedia. 2020. [pdf](http://cvit.iiit.ac.in/research/projects/cvit-projects/a-lip-sync-expert-is-all-you-need-for-speech-to-lip-generation-in-the-wild/)
