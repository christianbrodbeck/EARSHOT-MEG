# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os
from pathlib import Path
import pickle
from typing import Literal, Union


BUILTIN_DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR_ENV = 'EARSHOT_DATA_DIR'  # environment variable to set the data directory
DEFAULT_DATA_DIR = '~/Data/EARSHOT'
_data_dir = None  # set through set_data_dir()


def get_data_dir() -> Path:
    """Directory containing EARSHOT data (spectrograms, trained models, ...)

    Determined by, in order of precedence: :func:`set_data_dir`, the
    ``EARSHOT_DATA_DIR`` environment variable, and the default ``~/Data/EARSHOT``.
    """
    if _data_dir is not None:
        return _data_dir
    return Path(os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR)).expanduser()


def set_data_dir(path: Union[str, Path, None]) -> Path:
    """Set the directory containing EARSHOT data (``None`` to reset to the default)

    Call this before creating :class:`Lexicon` or :class:`Model` objects, which
    resolve their paths when they are created.
    """
    global _data_dir
    if path is None:
        _data_dir = None
    else:
        path = Path(path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"EARSHOT data directory does not exist: {path}")
        _data_dir = path
    return get_data_dir()


def __getattr__(name):
    # Backwards compatibility: DATA_DIR used to be a module-level constant
    if name == 'DATA_DIR':
        return get_data_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


VOWELS = ('AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW')
CONSONANTS = ('B', 'C', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH')
PHONES = VOWELS + CONSONANTS


def read_words(
        lexicon: Literal['earshot', 'burgundy', 'MALD-NEIGHBORS-1000'] = 'earshot',
):
    if lexicon.startswith('MALD'):
        path = BUILTIN_DATA_DIR / f'Words-{lexicon}.txt'
        words = [line.split() for line in path.open()]
        return [row[0] for row in words if len(row) == 2]
    return (BUILTIN_DATA_DIR / f'Words-{lexicon.capitalize()}.txt').read_text().splitlines()


def read_pronunciations(
        lexicon: Literal['earshot', 'burgundy', 'MALD-NEIGHBORS-1000'] = 'earshot',
        split_phones: bool = False,  # turn pronunciations into list of str
):
    if lexicon == 'burgundy':
        lexicon = 'MALD-NEIGHBORS-1000'
    pronunciation_file = get_data_dir() / f'Pronunciation-{lexicon}.pickle'
    with pronunciation_file.open('rb') as file:
        pronunciations = pickle.load(file)
    if split_phones:
        pronunciations = {word: pronunciation.split(' ') for word, pronunciation in pronunciations.items()}
    return pronunciations
