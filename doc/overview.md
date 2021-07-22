# Humans: Controllable avatar animation

Our task is to animate the video of a person talking starting from a given text (and optionally a specified identity).
We propose a pipeline consisting of three main modules:
1. text to speech
2. speech to keypoints
3. keypoints to video

Specifying these steps separately allows for
ⅰ. controllability and
ⅱ. to evaluate the components independently.

The figure below show the components of the pipeline, the datasets used for training and the adaptation procedure to a new identity 
(note that we can mix and match the identities from each stage of the pipeline—for example, we can generate audio in the voice of Obama, using the lip shape of Trump and the face of Bush).
![](imgs/2021-07-21-humans-pipeline.png)

**TODO** We show that:
- the speech-to-keypoints component is robust to various types of audio (real or synthesized, gender, race, language, accent)
- we leverage the flexibility of the TTS system to evaluate the first two components (that is, the text-to-lip module) as a whole

## Text to speech

We use the [FastPitch](https://fastpitch.github.io) (Łańcucki, 2021) method which has the ability to predict the audio in a parallel, non-autoregressive manner.
Its main idea lies in inferring high-level structure (F0 pitch and phone durations) from the audio, which allows to generate the corresponding audio directly without relying on the previously generated data.
Allowing to specify the duration is crucial not only for prediction, but it also allows us to evaluate the text-to-keypoint component as a whole, as we will see in the experimental section.

**Domain adaptation.**
Is performed by fine-tuning.

## Audio to keypoints

Lips can appear in a video at various locations, with different scales and rotations.
For this reason we normalize their absolute coordinates.
We also project the data into 8D using PCA.
Hence, our method maps a stream of audio to a list of 8D points.
These are the easily inverted via PCA.

**Domain adaption.**
When predicting the lip movements for a unseen identity, the lips dynamics are accurate, but unsurprisingly their shape resembles the one of the trained subject.
We adapt the lip shape to the one of the new person using a single frame, by replacing the training subject's mean with the one of the target speaker.
This operation is inexpensive (we only need a single frame) and works well in practice.

## Keypoints to video

**TODO** Work in progress; see [issue #36](https://gitlab.com/zevo-tech/humans/-/issues/36).

## Experiments

This sections present the experimental results.

**Evaluating the landmark detection methods.**
In this set of experiments, we quantify the quality of our lip predictions.
The automatically extracted lips have to be precise because they are used both at training and at testing time—we use them as targets in the audio-to-lip methods and we evaluate the audio-to-lip pipeline on them.
Here the goal is to compare the prediction of our methods to the annotations from existing datasets and try to improve the lip detection methods
(for example, we have noticed that the predicted lip landmarks are unstable from frame to frame, so we expect temporal smoothing to help).

See [issue #35](https://gitlab.com/zevo-tech/humans/-/issues/35).

**Evaluating text-to-keypoints.**
To evaluate the first two modules (text to speech and audio to lip) jointly,
we leverage the flexibility of the text-to-speech module, which allows to separately specify the duration of each phone in the input.
We obtain these dureation using a forced aligner between the evaluation text and audio.
The generated audio is transformed into landmarks by the audio-to-lip network and, finally, we compare the generated lips to the automatically extract landmarks using the root mean squared error score (lower values are better).
A schematic illustration of this process is shown below.
![](imgs/2021-07-21-humans-evaluate-text-to-keypoints.png)

For this evaluation we need evaluation data consisting of aligned text, audio and video.

## References

- Łańcucki, Adrian. "FastPitch: Parallel text-to-speech with pitch prediction." ICASSP, 2021. [paper](https://arxiv.org/abs/2006.06873)
