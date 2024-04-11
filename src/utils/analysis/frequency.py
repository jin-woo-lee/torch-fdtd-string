import crepe
from src.utils.misc import suppress_stdout_stderr

def compute_harmonic_parameters(x, sr):
    with suppress_stdout_stderr():
        time, f0, confidence, activation = crepe.predict(x, sr, viterbi=True)
    return dict(
        f0=f0,
    )

