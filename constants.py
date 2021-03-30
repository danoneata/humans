SEED = 1337
NUM_LANDMARKS = 68
LANDMARKS_INDICES = {
    "face": (0, 17),
    "eyebrow-l": (17, 22),
    "eyebrow-r": (22, 27),
    "nose": (27, 31),
    "nostril": (31, 36),
    "eye-l": (36, 42),
    "eye-r": (42, 48),
    "lips": (48, 60),
    "teeth": (60, 68),
}
# Indices for lips and teeth
LIPS_INDICES = (48, 68)
LEN_LIPS = LIPS_INDICES[1] - LIPS_INDICES[0]
