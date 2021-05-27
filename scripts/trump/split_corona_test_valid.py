import os

from scripts.subsample_espnet_data import ESPNET_EGS_PATH, write_scp, load_scp


def main():
    dataset = "trump-chunks-corona"

    path_wav_scp = os.path.join(ESPNET_EGS_PATH, "data", dataset, "wav.scp")
    path_lip_scp = os.path.join(ESPNET_EGS_PATH, "data", dataset, "lip.scp")

    wav_scp = load_scp(path_wav_scp)
    lip_scp = load_scp(path_lip_scp)

    keys = [key for key, _ in wav_scp]
    keys_split = {
        "valid": keys[-5:],
        "test": keys[:-5],
    }

    for split in "valid test".split():
        dir_name = f"{dataset}-{split}"

        path_wav_ss_scp = os.path.join(ESPNET_EGS_PATH, "data", dir_name, "wav.scp")
        path_lip_ss_scp = os.path.join(ESPNET_EGS_PATH, "data", dir_name, "lip.scp")

        wav_ss_scp = [datum for datum in wav_scp if datum[0] in keys_split[split]]
        lip_ss_scp = [datum for datum in lip_scp if datum[0] in keys_split[split]]

        write_scp(path_wav_ss_scp, wav_ss_scp)
        write_scp(path_lip_ss_scp, lip_ss_scp)


if __name__ == "__main__":
    main()
