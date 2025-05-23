import pdb
import requests
import sys

from tqdm import tqdm

from data import load_data


GENTLE_URL = "http://10.90.100.17:32768/transcriptions?async=false"
SILENCE_TOKENS = "sil,sp".split(",")


def align_gentle(sample):
    # Prepare data for Gentle Aligner
    transcript = " ".join(
        word for word in sample.sentence.split() if word not in SILENCE_TOKENS
    )
    audio_path = f"data/audio-from-video/{sample.speaker}/{sample.key}.wav"
    with open(audio_path, "rb") as f:
        audio = f.read()
    # Make request
    return requests.post(GENTLE_URL, data=dict(audio=audio, transcript=transcript))


def get_alignments_str(sample):
    response = align_gentle(sample)
    words = response.json()["words"]

    # Utterance id
    id_ = int(sample.speaker[1:])
    utt_id = f"s{id_:02d}_{sample.key}"

    # Format output
    lines = []
    sil_start_duration = words[0]["start"]
    lines.append(f"{utt_id} 1 0.00 {sil_start_duration:.2f} SIL")

    for word in words:
        start_time = word["start"]
        for phone in word["phones"]:
            duration = phone["duration"]
            phone_name = phone["phone"].upper()
            lines.append(f"{utt_id} 1 {start_time:.2f} {duration:.2f} {phone_name}")
            start_time += duration

    sil_end_start = words[-1]["end"]
    sil_end_duration = 3 - sil_end_start
    lines.append(f"{utt_id} 1 {sil_end_start:.2f} {sil_end_duration:.2f} SIL")

    return "\n".join(lines)


def main():
    split = sys.argv[1]
    with open(f"data/alignments-{split}-gentle.txt", "w") as f:
        for sample in tqdm(load_data(split)):
            try:
                lines = get_alignments_str(sample)
                f.write(lines)
                f.write("\n")
            except Exception as e:
                print("ERROR → ", sample.speaker, sample.key)
                # print(e)


if __name__ == "__main__":
    main()
