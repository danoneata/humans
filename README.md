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
# install our package locally to allow for imports
pip install -e .
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
ls /home/doneata/work/humans/output/models
```

# Datasets

## Obama's weekly addresses

**Data preparation.**
Steps to reproduce the processing for the Obama dataset.
1. Clone the repo [`synthesizing_obama_network_training`](https://github.com/supasorn/synthesizing_obama_network_training), which contains video names and information regarding the video shots;
then create the filelist with video names.
```bash
git clone git@github.com:supasorn/synthesizing_obama_network_training.git synthesizing-obama-network-training
/bin/ls -1 synthesizing-obama-network-training/obama_data > data/obama/filelists/video-names.txt
```
2. Download the dataset:
```bash
bash scripts/obama/download.sh
```
3. Split the videos:
```bash
INFO_PATH=synthesizing-obama-network-training python scripts/obama/process_videos.sh -t split-video
```
4. Extract the audio:
```bash
INFO_PATH=synthesizing-obama-network-training python scripts/obama/process_videos.sh -t extract-audio
```
5. Resize the videos:
```bash
INFO_PATH=synthesizing-obama-network-training python scripts/obama/process_videos.sh -t resize-videos -r 360p
```
6. Generate train, validation and test splits (note that two video shots were removed since they do not contain Obama, see #16):
```bash
python scripts/obama/generate_train_test_splits.py
wc -l data/obama/filelists/{train,valid,test}.txt
#   798 data/obama/filelists/train.txt
#    80 data/obama/filelists/valid.txt
#    80 data/obama/filelists/test.txt
#   958 total
```

**Characteristics.**
- Duration: 17 hours
- Number of frames: 1.8M
- Frames per second: 29.97
- Video resolutions:
```
count | resolution
------|------------
   32 | 1920 × 1080
  260 | 1280 ×  720
    1 |  854 ×  480
    8 |  640 ×  360
```
Obtained with the following command:
```bash
/bin/ls -1 data/obama/video/ | parallel ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 data/obama/video/{} | sort | uniq -c
```

## LRS3

**Data preparation.**
Using this dataset for evaluation only; so we are using only it's test split.
1. Data is downloaded from [here](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html).
2. Prepare filelists:
```bash
find /mnt/private-share/speechDatabases/lrs3/test/ -name '*.mp4' | tr '/' ' ' | cut -f1 -d'.' | awk '{print $6, $7, $5}' | sort > data/lrs3/filelists/test.txt
```

**Characteristics.**
- Frames per second: 25
- Video resolution: 224 × 244; all videos have the face cropped and the same resolution.

# Main functionality

Extract face landmarks:

```bash
python scripts/extract_face_landmarks.py --dataset grid --filelist tiny --n-cpu 4 -v
```

**Visualize results.**
Using [Streamlit](https://streamlit.io/), we can easily view the intermediate steps of our pipeline.
```bash
streamlit run scripts/show_results.py
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
- Displacements of each landmark relative to previous frame.

**Extracting face landmarks.**
Various options possible:
- [dlib](http://dlib.net/), currently used in `scripts/extract_face_landmarks.py`
- [Adrian Bulat's `face-alignment` library](https://github.com/1adrianb/face-alignment) (suggested by Dragoș)
- [TensorFlow.js FaceMesh](https://github.com/tensorflow/tfjs-models/tree/master/face-landmarks-detection) (suggested by Adriana)

Maybe try all on a small subset and compare them (also in terms of execution time).

Running the methods on 10 video samples from the GRID dataset at resolution 360 × 288 and Obama dataset at downscaled to 640 × 360.

| method                 | data  | resolution  | time (s) / frame | examples |
|------------------------|-------|-------------|------------------|----------|
| dlib (1 CPU)           | GRID  |  360 × 288  | 0.04             | TODO     |
| face-alignment (1 GPU) | GRID  |  360 × 288  | 0.09             | TODO     |
| dlib (1 CPU)           | Obama |  640 × 360  | 0.09             | TODO     |
| face-alignment (1 GPU) | Obama |  640 × 360  | 0.11             | TODO     |

**Data.**
What datasets should we use for training the model?
We need pairs of audio and front video recordings;
there should be plenty of such datasets.
For example, we have access to the Lip Reading in the Wild dataset (although non-commercial conditions might apply).
But if multiple speakers are present in the training dataset will we need to encode the speaker identity (in order to account for mouth shape variability)?
Or is this information (the speaker identity) already present in the input audio stream?
To explicitly circumvent the speaker variability, I would be very tempted to use a single-person dataset (_e.g._, Obama for which there are many video recording available).
Although, on a second thought, this approach has the disadvantage of being specific to a single type of voice.

- _Q_ Business-wise is better to target a single speaker or does the method need to work for any speaker from the beginning?
- Follow Dragoș's idea and have a separate upscaling model that learns to interpolate lips landmarks for high resolution videos?
This approach would allow us to learn on lower resolution videos.
- If we do not use a multi-task approach (see next sub-sections) then we do not need transcripts and we could potentially crawl our own data from YouTube.

| dataset   | num. hours | num. speakers | resolution  | fps | transcripts | observations                                                      | status                                                      | links                                                                                                                                                       |
|-----------|------------|---------------|-------------|-----|-------------|-------------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GRID      | 28         | 33            | 720 × 576   | 25  | ✓           | limited vocabulary, constrained conditions                        | downloaded (small videos, 360 × 288)                        | [paper](https://pubmed.ncbi.nlm.nih.gov/17139705/) [data](http://spandh.dcs.shef.ac.uk/gridcorpus/)                                                         |
| TCD-TIMIT |            | 62            | 1920 × 1080 | 30  | ✓           | constrained conditions, three professionally-trained lip speakers | [N/A?](http://www.mee.tcd.ie/~sigmedia/Resources/TCD-TIMIT) | [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7050271)                                                                                       |
| LRW       |            |               |             |     | ✓           | single word                                                       | downloaded                                                  | [data](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)                                                                                         |
| LRS2      |            |               |             |     | ✓           |                                                                   | downloaded                                                  | [data](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)                                                                                         |
| LRS3      | 438        | 10,000        | 224 × 224   | 25  | ✓           | short sentences, strong head movement, cropped around face        | downloaded                                                  | [data](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)                                                                                         |
| AVSpeech  |            |               |             |     | ?           |                                                                   |                                                             | |
| APB2Face  |            |               |             |     | ?           |                                                                   |                                                             | [project](https://github.com/zhangzjn/APB2FaceV2)                                                                                         |
| Lip2Wav   | 120        | 5             |             |     | ✗           | YouTube videos, diverse vocabulary                                | TODO                                                        | [project](http://cvit.iiit.ac.in/research/projects/cvit-projects/speaking-by-observing-lip-movements) [data](https://cove.thecvf.com/datasets/363)          |
| Obama     | 17         | 1             | 1280 × 720  | 30? | ?           | YouTube videos, diverse vocabulary                                | in progress                                                 | [paper](https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf) [data](https://github.com/supasorn/synthesizing_obama_network_training) |

**Intermediate phonetic representation.**
The model goes between two aligned signals: from audio to lips.
However, we could explicitly enforce the intermediate layer to correspond to phonetic representation,
for example, by using a multi-task approach and having a branch that transcribes the audio (predicts text) from the encoded activations.

**An experiment in phoneme-to-lip mapping.**
A straightforward way of mapping phonemes to lip movements (_visemes_) is by aligning the phonetized transcription to the audio of the video.
At test time, we can generate lip movements for a new transcription by sequencing the corresponding visemes.

Here are some [initial results](http://speed.pub.ro/xts/backup-2020-05-20) that show 128 samples of lip movements for twelve phones (vowels):
AA, AE, AH, AO, AW, AY, EH, EY, IH, IY, OW, UW.
For this experiment, we have used the [GRID dataset](https://pubmed.ncbi.nlm.nih.gov/17139705/) and
obtained the alignments by using a pre-trained automatic speech recognition (ASR) model trained on TED-LIUM with the Kaldi framework
(but we can also use alignments extracted with [Gentle](https://github.com/lowerquality/gentle)).
Faces and their landmarks were detected using the [`dlib` toolkit](http://dlib.net/).

# Experimental results

We report mean squared error between the lip coördinates of the aligned representation and the predicted data on the validation and test splits.
Since the predictions are in the 8D PCA space, we also report the PCA reconstruction error;
see the `evaluate.py` script.
Below "best" indicates using the model that yielded the best validation loss throughout the training,
while "ave" indicates a model that averages the best ten models.

| method                                 | valid  | test   |
|----------------------------------------|--------|--------|
| PCA reconstruction                     | 0.0007 | 0.0007 |
| baseline                               | 0.0275 | 0.0264 |
| ... + audio encoder fine-tuning (best) | 0.0138 | 0.0138 |
| ... + audio encoder fine-tuning (ave)  | 0.0134 | 0.0132 |

Qualitative examples are available [here](https://zevo-tech.com/humans/).

# References

- Suwajanakorn, Supasorn, Steven M. Seitz, and Ira Kemelmacher-Shlizerman. "Synthesizing Obama: learning lip sync from audio." ACM Transactions on Graphics (ToG) 36.4 (2017): 1-13. [pdf](https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf) · [code](https://github.com/supasorn/synthesizing_obama_network_training)
- Kumar, Rithesh, et al. "Obamanet: Photo-realistic lip-sync from text." arXiv preprint arXiv:1801.01442 (2017). [pdf](https://arxiv.org/pdf/1801.01442.pdf) · [code](https://github.com/karanvivekbhargava/obamanet) (original implementation) · [code](https://github.com/acvictor/Obama-Lip-Sync) (separate implementation)
- Fried, Ohad, et al. "Text-based editing of talking-head video." ACM Transactions on Graphics (TOG) 38.4 (2019): 1-14. [pdf](https://dl.acm.org/doi/pdf/10.1145/3306346.3323028)
- Prajwal, K. R., et al. "A lip sync expert is all you need for speech to lip generation in the wild." Proceedings of the 28th ACM International Conference on Multimedia. 2020. [project](http://cvit.iiit.ac.in/research/projects/cvit-projects/a-lip-sync-expert-is-all-you-need-for-speech-to-lip-generation-in-the-wild/) · [code](https://github.com/Rudrabha/Wav2Lip)
- Manocha, Prateek, and Prithwijit Guha. "Facial Keypoint Sequence Generation from Audio." arXiv preprint arXiv:2011.01114 (2020). [paper](https://arxiv.org/pdf/2011.01114v1.pdf)
- Zhang, Jiangning, et al. "APB2Face: Audio-guided face reenactment with auxiliary pose and blink signals." ICASSP, 2020. [paper](https://arxiv.org/pdf/2004.14569.pdf) · [code & data](https://github.com/zhangzjn/APB2FaceV2)
- Monrose, F., M. Reiter, and Q. Li. "Voice2Mesh: Cross-Modal 3D Face Model Generation from Voices." arXiv preprint arXiv:2104.10299. [paper](https://arxiv.org/abs/2104.10299)
