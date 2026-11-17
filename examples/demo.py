import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="EARSHOT demo")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EARSHOT demo

    EARSHOT is a recurrent neural network model of human speech recognition.
    It listens to speech in the form of a spectrogram, one 10 ms frame at a time,
    and at every frame it reports how strongly it believes that each word in its lexicon is currently being said.

    This notebook walks through the basic steps for working with a pretrained model
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import earshot
    from earshot.train_earshot import Model, Word, gen_input_from_words

    return Model, Word, earshot, gen_input_from_words, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data

    The spectrograms and model weights are too large for the code repository, so they live in a separate *data directory*.
    This demo needs the following files:

    ```
    EARSHOT/                      <- the data directory
    ├── GAMMATONE_64_100/         <- model input: one spectrogram per word and speaker
    │   ├── Agnes/
    │   │   ├── ABLE_AGNES.pickle
    │   │   └── ...
    │   ├── ...
    │   └── MALD/
    │       ├── ABLE_MALD.pickle
    │       └── ...
    └── Models/
        └── LSTM-192x192x192x192-dw1024to10-RMSProp-1e-3-32-25-s0 OneHot-box MALD-1000-test-64-2-20/
            ├── 4419.hdf5         <- network weights after the last training epoch
            └── log.csv           <- training loss
    ```

    By default, `earshot` looks for the data directory at `~/Data/EARSHOT`.
    If you put it somewhere else, either

    - set the `EARSHOT_DATA_DIR` environment variable before starting marimo, or
    - call `earshot.set_data_dir()` in the cell below.
    """)
    return


@app.cell
def _(earshot, mo):
    # Uncomment and edit this line if your data is not in ~/Data/EARSHOT:
    # earshot.set_data_dir('/path/to/EARSHOT')

    DATA_DIR = earshot.get_data_dir()

    _missing = [name for name in ['GAMMATONE_64_100', 'Models'] if not (DATA_DIR / name).is_dir()]
    mo.stop(bool(_missing), mo.callout(mo.md(f"**Data not found.** `{DATA_DIR}` does not contain {' and '.join(f'`{name}/`' for name in _missing)}. Set the data directory in the cell above."), kind='danger'))
    mo.md(f"Using data in `{DATA_DIR}`")
    return (DATA_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The model

    A `Model` object is a *description* of a model: its architecture and how it was trained.
    It does not contain the network itself yet (we will load that in section 5),
    but it knows where to find the weights, because the name of the model directory is derived from these parameters.

    | Parameter | Value | Meaning |
    |---|---|---|
    | `lexicon_name` | `'MALD-1000-test'` | The words the model knows, and which recordings are used for training (see below) |
    | `n_bands` | `64` | Number of frequency bands in the input spectrogram |
    | `mechanism`, `hidden` | `'LSTM'`, `'192x192x192x192'` | Four stacked LSTM layers with 192 units each |
    | `target_space` | `'OneHot'` | One output unit for each word |
    | `target_shape` | `'box'` | The output unit for a word should be 1 while the word is being said, and 0 otherwise |
    | `loss` | `'dw1024to10'` | While a word is unfolding, activating *other* words is penalized 1024 times less. The full penalty applies only during the last 10 frames (100 ms) of each word, and in silence. |
    | `stimulus_sequence` | `'2'` | Training stimuli are a continuous stream of words, with a 200–500 ms pause after every second word on average |
    | `snr` | `20` | During training, noise is added to the input at a signal-to-noise ratio of 20 dB |
    | `optimizer`, `learning_rate` | `'RMSProp'`, `0.001` | Training algorithm |
    | `batch_size`, `steps_per_epoch` | `32`, `25` | One epoch consists of 25 batches of 32 ten-second stimuli |
    | `seed` | `0` | Random seed for the initial weights |
    """)
    return


@app.cell
def _(DATA_DIR, Model, mo):
    model = Model(
        lexicon_name='MALD-1000-test',
        n_bands=64,
        mechanism='LSTM',
        hidden='192x192x192x192',
        target_space='OneHot',
        target_shape='box',
        loss='dw1024to10',
        stimulus_sequence='2',
        snr=20,
        optimizer='RMSProp',
        learning_rate=1e-3,
        batch_size=32,
        steps_per_epoch=25,
        seed=0,
    )
    lexicon = model.lexicon

    mo.stop(not model.checkpoints, mo.callout(mo.md(f"**No trained model found.** Expected weights (`*.hdf5`) in `{model.model_dir}`"), kind='danger'))
    mo.md(f"""
    Model name: `{model.name}`

    Weights: `{(model.model_dir / f'{model.checkpoints[-1]}.hdf5').relative_to(DATA_DIR)}`
    """)
    return lexicon, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The lexicon

    The model's `lexicon` defines which words the model knows and which speakers it has heard.
    Each combination of a speaker and a word is one *token*, i.e., one recording.

    In the `MALD-1000-test` lexicon, 15 speakers are synthetic text-to-speech voices.
    The 16th, `MALD`, is a human speaker (from the Massive Auditory Lexical Decision database),
    and is the voice that participants heard in the MEG experiment.
    For the roughly 1000 words that were used in that experiment (called *Burgundy* in the code),
    the model was trained on the synthetic voices only – the `MALD` recordings of those words were held out.
    This lets us test whether the model can recognize words from a speaker for whom it has never heard those words.
    """)
    return


@app.cell
def _(lexicon, mo):
    # Tokens are (speaker, word) tuples
    test_items = set(lexicon.test_items)

    mo.md(f"""
    - **{lexicon.n_words} words**, for example: {', '.join(lexicon.words[1000:1012])}, ...
    - **{len(lexicon.speakers)} speakers**: {', '.join(lexicon.speakers)}
    - **{len(lexicon.train_items)} tokens** for training, for example: `{lexicon.train_items[0]}`
    - **{len(test_items)} tokens** held out for testing, {sum(speaker == 'MALD' for speaker, _ in test_items)} of which are from the `MALD` speaker
    """)
    return (test_items,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Model input

    The model does not hear sound waves directly.
    Like the human auditory system, it receives a *spectrogram*: a representation of how much energy the sound contains at different frequencies, and how this changes over time.
    Specifically, the input is a gammatone spectrogram with 64 frequency bands (from 50 Hz to 10 kHz, spaced like in the cochlea), sampled at 100 Hz – that is, one frame every 10 ms.
    (The repository's README shows how to create such a spectrogram from a sound file.)

    `lexicon.inputs` gives access to the spectrogram for each token, as an array with shape `(time, frequency)`.
    The files are loaded from disk when they are first needed.

    Pick a word to see how different speakers say it:
    """)
    return


@app.cell(hide_code=True)
def _(lexicon, mo):
    example_word_input = mo.ui.dropdown(lexicon.words, value='MOUNTAIN', label='Word', searchable=True)
    example_word_input
    return (example_word_input,)


@app.cell
def _(np, plt):
    TSTEP = 0.01  # duration of one frame in seconds

    plt.rcParams.update({
        'figure.constrained_layout.use': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 9,
        'axes.titlesize': 9,
    })

    def plot_spectrogram(ax, spectrogram, t_start=0.):
        "Plot a (time, frequency) spectrogram, with time in seconds"
        n_times, n_bands = spectrogram.shape
        extent = (t_start, t_start + n_times * TSTEP, 0, n_bands)
        ax.imshow(spectrogram.T, origin='lower', aspect='auto', extent=extent, cmap='Blues', vmin=0, vmax=0.1, interpolation='none')
        ax.set_ylabel('Frequency band')
        ax.set_yticks([0, n_bands], ['low', 'high'])

    def mark_words(ax, words, labels=False):
        "Draw a line at the onset and offset of each word (see section 4)"
        for word in words:
            ax.axvline(word.t0 * TSTEP, color='0.4', linewidth=0.5)
            ax.axvline(word.t1 * TSTEP, color='0.4', linewidth=0.5, linestyle=':')
            if labels:
                ax.text((word.t0 + word.t1) / 2 * TSTEP, 1.02, word.word, ha='center', va='bottom', transform=ax.get_xaxis_transform())

    def show(figure):
        "Display a figure in the notebook, and close it to free up memory"
        plt.close(figure)
        return figure

    def frames_to_seconds(n_frames):
        return np.arange(n_frames) * TSTEP

    return TSTEP, frames_to_seconds, mark_words, plot_spectrogram, show


@app.cell
def _(TSTEP, example_word_input, lexicon, plot_spectrogram, plt, show):
    _speakers = ['MALD', 'AGNES', 'BRUCE', 'PRINCESS']
    _spectrograms = [lexicon.inputs[speaker, example_word_input.value] for speaker in _speakers]
    _duration = max(len(spectrogram) for spectrogram in _spectrograms) * TSTEP

    _fig, _axes = plt.subplots(1, len(_speakers), figsize=(10, 2.2), sharex=True, sharey=True)
    for _ax, _speaker, _spectrogram in zip(_axes, _speakers, _spectrograms):
        plot_spectrogram(_ax, _spectrogram)
        _ax.set_title(f'{_speaker}: {_spectrogram.shape[0]} frames × {_spectrogram.shape[1]} bands')
        _ax.set_xlabel('Time (s)')
        _ax.label_outer()
    _axes[0].set_xlim(0, _duration)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The same word looks quite different depending on who says it: speakers differ in pitch, in speech rate, and in how they pronounce individual sounds.
    To recognize words, the model has to learn what stays the same across speakers.

    ## 4. Creating a stimulus

    The model processes continuous speech: there is no signal that tells it where one word ends and the next one begins.
    We create a stimulus by placing the spectrograms of several words one after another on a timeline.

    Each word in a stimulus is represented by a `Word` object, which combines a token (speaker and word) with a time, `t0`, at which the word starts.
    Times are counted in frames, so `t0=20` means that the word starts 200 ms into the stimulus.

    Choose the stimulus here (use up to 8 words from the lexicon; the default words are all held-out `MALD` tokens):
    """)
    return


@app.cell(hide_code=True)
def _(lexicon, mo):
    words_input = mo.ui.text('MOUNTAIN PISTOL CARTOON WHISPER UMBRELLA', label='Words', full_width=True)
    speaker_input = mo.ui.dropdown(lexicon.speakers, value='MALD', label='Speaker')
    silence_input = mo.ui.slider(0, 500, step=50, value=200, label='Silence between words (ms)', show_value=True)
    mo.vstack([words_input, mo.hstack([speaker_input, silence_input], justify='start', gap=2)])
    return silence_input, speaker_input, words_input


@app.cell
def _(Word, lexicon, mo, silence_input, speaker_input, words_input):
    def concatenate_words(word_names, speaker, silence=10, t_start=20):
        "Place words one after another, separated by ``silence`` frames"
        words = []
        t0 = t_start
        for word_name in word_names:
            word = Word(t0, speaker, word_name, lexicon)
            words.append(word)
            # word.t1 is the frame at which this word ends
            t0 = word.t1 + silence
        return words

    _word_names = words_input.value.upper().split()
    _unknown = [name for name in _word_names if name not in lexicon.word_index]
    mo.stop(bool(_unknown), mo.callout(mo.md(f"**Not in the lexicon:** {', '.join(_unknown)}. Try one of the {lexicon.n_words} words in `lexicon.words`."), kind='warn'))
    mo.stop(not 0 < len(set(_word_names)) <= 8, mo.callout("Enter between 1 and 8 different words.", kind='warn'))

    words = concatenate_words(_word_names, speaker_input.value, silence=silence_input.value // 10)
    return (words,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Input and target

    `gen_input_from_words()` turns a sequence of `Word` objects into arrays.
    It always generates segments of 1000 frames (10 s), which is the length of stimulus that the network was trained on.
    For each segment it returns two arrays (the first dimension is for batches, and we only have a single stimulus):

    - The **input**, with shape `(1, time, frequency)`: silence (zeros) with the spectrogram of each word added at the right time.
    - The **target**, with shape `(1, time, word)`: what the output of a perfect model would look like.
      With one-hot targets, each word in the lexicon has its own output unit.

    (The same function is used to generate stimuli for training the model, from random sequences of words.)
    """)
    return


@app.cell
def _(gen_input_from_words, lexicon, mo, np, words):
    _segments = list(gen_input_from_words(iter(words), lexicon))
    inputs = np.concatenate([segment_inputs[0] for segment_inputs, _ in _segments])
    targets = np.concatenate([segment_targets[0] for _, segment_targets in _segments])

    # For plotting, we only need the part with the words, plus half a second
    n_frames = min(words[-1].t1 + 50, len(inputs))

    mo.md(f"""
    - `inputs`: {inputs.shape[0]} frames × {inputs.shape[1]} frequency bands
    - `targets`: {targets.shape[0]} frames × {targets.shape[1]} words
    - The words end after {words[-1].t1} frames, so we will plot the first {n_frames} frames ({n_frames / 100:g} s)
    """)
    return inputs, n_frames, targets


@app.cell
def _(words):
    # A color for each word in the stimulus
    PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
    word_colors = {}
    for _word in words:
        if _word.word not in word_colors:
            word_colors[_word.word] = PALETTE[len(word_colors)]
    return (word_colors,)


@app.cell
def _(
    frames_to_seconds,
    inputs,
    lexicon,
    mark_words,
    n_frames,
    plot_spectrogram,
    plt,
    show,
    targets,
    word_colors,
    words,
):
    _time = frames_to_seconds(n_frames)

    _fig, _axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    # Input
    plot_spectrogram(_axes[0], inputs[:n_frames])
    mark_words(_axes[0], words, labels=True)
    # Target: plot only the output units for the words in the stimulus
    for _name, _color in word_colors.items():
        _axes[1].plot(_time, targets[:n_frames, lexicon.word_index[_name]], color=_color, label=_name)
    mark_words(_axes[1], words)
    _axes[1].set_ylabel('Target')
    _axes[1].set_ylim(-0.05, 1.05)
    _axes[1].set_xlabel('Time (s)')
    _axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Solid vertical lines mark word onsets, dotted lines mark word offsets.
    The target for a word is a "box": it is 1 from the onset to the offset of the word.
    All the other output units (not shown) have a target of 0 throughout.

    Note that this is an ambitious target: it asks the model to recognize a word from its very first frame.
    That is impossible, because many words start with the same sounds.
    How the model deals with this uncertainty is the interesting part.

    ## 5. Loading the network

    `model.load_model()` builds the neural network and loads the weights from the last training epoch.
    Two arguments are worth knowing about:

    - `batch_size=1`: we want to process one stimulus at a time (training used 32 in parallel).
    - `output_hidden=True`: the network returns the activity in each hidden layer, in addition to the word outputs.
    """)
    return


@app.cell
def _(mo, model):
    network = model.load_model(batch_size=1, output_hidden=True)

    _lines = []
    network.summary(print_fn=_lines.append)
    mo.plain_text('\n'.join(_lines))
    return (network,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reading the summary from top to bottom follows the flow of information:
    the input (1 stimulus × 1000 frames × 64 frequency bands) passes through four LSTM layers with 192 units each,
    and a final `Dense` layer turns the activity of the last LSTM layer into one output for each of the 2934 words.

    LSTM layers are *recurrent*: what they do at each frame depends not only on the current input, but also on their own state from the previous frame.
    This memory is what allows the model to integrate information over the duration of a word.

    ### Training

    The model directory also contains a log of the *loss* during training.
    The loss measures the mismatch between the model's output and the target, so lower is better.
    """)
    return


@app.cell
def _(model, plt, show):
    _epochs, _loss = zip(*model.load_log(raw=True))

    _fig, _ax = plt.subplots(figsize=(6, 2.5))
    _ax.plot(_epochs, _loss, color='#2a78d6', linewidth=1)
    _ax.set_yscale('log')
    _ax.set_xlabel('Training epoch')
    _ax.set_ylabel('Loss')
    _ax.set_title(f'Training loss (final weights: epoch {model.checkpoints[-1]})', loc='left')
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Running the model

    `model.predict()` generates the input for our `words` (as in section 4), runs it through the network, and collects the results in a dictionary.

    One detail: the network is *stateful*. At the end of a 10 s segment it remembers its state, so that a longer stimulus can be processed as a sequence of segments without interruption.
    Before presenting a new stimulus, we therefore reset the state with `network.reset_states()`.
    """)
    return


@app.cell
def _(mo, model, network, words):
    network.reset_states()
    data = model.predict(words=words, model=network)

    mo.md("`data` contains:\n\n" + '\n'.join(f"- `{key!r}`: array with shape `{value.shape}`" for key, value in data.items() if key != 'loss'))
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Word activations

    `data['outputs']` has the same shape as the target: one value between 0 and 1 for each frame and each word in the lexicon.
    We can think of this value as the *activation* of the word – how strongly the model believes that it is hearing this word right now.

    The figure below shows the input together with the activation of **all 2934 words**.
    The words that are actually in the stimulus are drawn in color; all other words are grey.
    The bottom panel counts how many words are active (activation above 0.5) at each moment.
    """)
    return


@app.cell
def _(
    data,
    frames_to_seconds,
    lexicon,
    mark_words,
    n_frames,
    plot_spectrogram,
    plt,
    show,
    word_colors,
    words,
):
    outputs = data['outputs'][:n_frames]
    _time = frames_to_seconds(n_frames)

    _fig, _axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 3, 1.5])
    # Input
    plot_spectrogram(_axes[0], data['inputs'][:n_frames])
    mark_words(_axes[0], words, labels=True)
    # Output: all words in grey, stimulus words in color
    _axes[1].plot(_time, outputs, color='0.5', linewidth=0.3, alpha=0.3, rasterized=True)
    for _name, _color in word_colors.items():
        _axes[1].plot(_time, outputs[:, lexicon.word_index[_name]], color=_color, linewidth=2, label=_name)
    _axes[1].set_ylabel('Word activation')
    _axes[1].set_ylim(0, 1.05)
    _axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    # Number of active words
    _axes[2].step(_time, (outputs > 0.5).sum(axis=1), color='0.2', linewidth=1, where='post')
    _axes[2].set_ylabel('Active words')
    _axes[2].set_ylim(0)
    _axes[2].set_xlabel('Time (s)')
    for _ax in _axes[1:]:
        mark_words(_ax, words)
    show(_fig)
    return (outputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A few things to notice:

    - **Many words become active at word onset.** As soon as a word starts, the model activates a whole set of candidate words – typically those that start with similar sounds (the *cohort*).
    - **Competitors drop out as the word unfolds.** With every additional sound, fewer candidates are compatible with the input, and the number of active words goes down.
    - **Ideally, a single word is left at the end,** and it is the one that was actually said.
    - **The model is quiet during silence.**

    This behavior is a consequence of how the model was trained (the `loss` parameter in section 2):
    while a word is unfolding, the model is barely penalized for activating other words, so it can afford to keep all plausible candidates active.
    Only during the last 100 ms of each word does it have to commit.

    ### Competition within each word

    To see *which* words compete, the next figure zooms in on each word of the stimulus.
    Each panel shows the target word in color, together with the four other words that reached the highest activation while the word was presented.
    The shaded region marks the last 100 ms of the word, where the training loss applies the full penalty for activating a wrong word.
    """)
    return


@app.cell
def _(TSTEP, lexicon, np, outputs, plt, show, word_colors, words):
    N_COMPETITORS = 4
    _linestyles = ['-', '--', '-.', ':']

    _n_columns = min(len(words), 4)
    _n_rows = int(np.ceil(len(words) / _n_columns))
    _fig, _axes = plt.subplots(_n_rows, _n_columns, figsize=(2.5 * _n_columns, 3 * _n_rows), sharey=True, squeeze=False)
    for _ax in _axes.flat[len(words):]:
        _ax.set_visible(False)
    for _ax, _word in zip(_axes.flat, words):
        _target = lexicon.word_index[_word.word]
        # Show the output from word onset until 100 ms after word offset
        _output = outputs[_word.t0: _word.t1 + 10]
        _time_ms = np.arange(len(_output)) * TSTEP * 1000
        _duration_ms = (_word.t1 - _word.t0) * TSTEP * 1000
        # Find competitors: words with the highest peak activation during the word
        _peak = outputs[_word.t0: _word.t1].max(axis=0)
        _peak[_target] = -1  # exclude the target itself
        _competitors = np.argsort(_peak)[::-1][:N_COMPETITORS]
        # Plot
        _ax.axvspan(_duration_ms - 100, _duration_ms, color='0.92', linewidth=0)
        for _index, _linestyle in zip(_competitors, _linestyles):
            _ax.plot(_time_ms, _output[:, _index], color='0.35', linewidth=1, linestyle=_linestyle, label=lexicon.words[_index])
        _ax.plot(_time_ms, _output[:, _target], color=word_colors[_word.word], linewidth=2.5, label=_word.word)
        _ax.set_title(_word.word, loc='left', fontweight='bold')
        _ax.set_xlabel('Time since word onset (ms)')
        _ax.set_ylim(0, 1.05)
        _ax.set_xlim(0, _time_ms[-1])
        _ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), frameon=False, ncols=2, handlelength=2.5, columnspacing=1)
    for _row in _axes:
        _row[0].set_ylabel('Word activation')
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The competitors usually share their beginning with the target word.
    Often they stay active until the input contains a sound that rules them out
    (try `PISTOL` and `PISTON`, or `CARTON` and `CARTOON`, which only differ late in the word).

    ### Did the model recognize the words?

    To decide which word the model "heard", we take the word with the highest activation during the last 100 ms of each word in the stimulus.
    """)
    return


@app.cell
def _(lexicon, mo, outputs, test_items, words):
    def recognized_word(word):
        "Word with the highest activation in the last 10 frames (100 ms) of ``word``"
        final_activation = outputs[word.t1 - 10: word.t1].max(axis=0)
        return lexicon.words[final_activation.argmax()]

    _rows = []
    for _word in words:
        _recognized = recognized_word(_word)
        _rows.append({
            'Word': _word.word,
            'Recognized as': _recognized,
            'Correct': '✓ yes' if _recognized == _word.word else '✗ no',
            'Used for training': 'no (held out)' if (_word.speaker, _word.word) in test_items else 'yes',
        })
    _n_correct = sum(row['Correct'].startswith('✓') for row in _rows)

    mo.vstack([
        mo.ui.table(_rows, selection=None),
        mo.md(f"**{_n_correct} of {len(_rows)}** words recognized correctly."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now go back to section 4 and change the stimulus. For example:

    - Switch to one of the synthetic speakers. Tokens that were used for training are recognized almost perfectly.
    - With the `MALD` speaker, try some other words from the MEG experiment, for example `ATOM BALCONY ASPIRIN`.
      The model has never heard this speaker say these words, and it makes mistakes: it recognizes about 60% of these held-out tokens.
      When it is wrong, look at the competitors – the errors are often similar-sounding words.
    - Set the silence between words to 0. The model has to find the word boundaries by itself.
    - Set the silence to 50 or 100 ms. During training, words either followed each other directly or were separated by 200–500 ms of silence.
      Short pauses are something the model has never encountered, and with the `MALD` speaker it tends to make more mistakes.

    ## 7. Hidden layers

    So far we have looked at the input and the output of the model.
    The computations that turn sound into words happen in between, in the four LSTM layers.
    Because we loaded the network with `output_hidden=True`, `data` also contains the activity of each hidden unit: `data['hidden0']` (the layer closest to the input) to `data['hidden3']` (the layer closest to the output).
    Each is an array with shape `(time, unit)`, with values between −1 and 1.
    """)
    return


@app.cell
def _(TSTEP, data, mark_words, n_frames, plt, show, words):
    hidden_layers = {key: value[:n_frames] for key, value in data.items() if key.startswith('hidden')}

    _fig, _axes = plt.subplots(len(hidden_layers), 1, figsize=(10, 1.6 * len(hidden_layers)), sharex=True)
    for _ax, (_key, _hidden) in zip(_axes, hidden_layers.items()):
        _extent = (0, n_frames * TSTEP, 0, _hidden.shape[1])
        _image = _ax.imshow(_hidden.T, origin='lower', aspect='auto', extent=_extent, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='none')
        mark_words(_ax, words, labels=_ax is _axes[0])
        _ax.set_ylabel(f'{_key}\nUnit')
    _axes[-1].set_xlabel('Time (s)')
    _fig.colorbar(_image, ax=_axes, label='Activation', shrink=0.4, aspect=15)
    show(_fig)
    return (hidden_layers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each row is one hidden unit; red is positive and blue is negative activation.
    In the first layer, activity is moderate and still follows the input fairly closely.
    Deeper layers use their full range, and their activity looks increasingly different from the spectrogram.
    The last layer changes abruptly at the end of each word, when the model's output returns to silence.

    ### Summarizing hidden layer activity

    To compare a model like this to brain responses, the paper summarizes the activity of each layer with two time series:

    - **Magnitude**: the sum of the absolute activation of all units in the layer – how active is the layer overall?
    - **Onsets**: the sum of all *increases* in absolute activation from one frame to the next – how much new activity appears at each moment?

    These time series are then used as predictors for MEG responses to the same stimuli.
    """)
    return


@app.cell
def _(
    frames_to_seconds,
    hidden_layers,
    mark_words,
    n_frames,
    np,
    plt,
    show,
    words,
):
    _time = frames_to_seconds(n_frames)
    # Layers are ordered from input to output: use light to dark shades of one color
    _colors = ['#9ec5ee', '#5b9be0', '#2a6fc4', '#123f7a']

    _fig, _axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    for (_key, _hidden), _color in zip(hidden_layers.items(), _colors):
        _magnitude = np.abs(_hidden).sum(axis=1)
        _change = np.diff(np.abs(_hidden), axis=0, prepend=0)
        _onsets = np.clip(_change, 0, None).sum(axis=1)
        _axes[0].plot(_time, _magnitude, color=_color, label=_key)
        _axes[1].plot(_time, _onsets, color=_color, label=_key)
    _axes[0].set_ylabel('Magnitude')
    _axes[1].set_ylabel('Onsets')
    _axes[1].set_xlabel('Time (s)')
    _axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    mark_words(_axes[0], words, labels=True)
    mark_words(_axes[1], words)
    for _ax in _axes:
        _ax.set_ylim(0)
        _ax.set_xlim(0, _time[-1])
    show(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to go from here

    - **Noise**: the model was trained with noise in the input. Use `model.predict(words=words, model=network, snr=20)` to add noise at a signal-to-noise ratio of 20 dB.
      (To scale the noise, this loads the spectrograms for all training tokens, so it requires the complete `GAMMATONE_64_100` directory and takes a moment.)
    - **Random stimuli**: `earshot.train_earshot.gen_words()` generates endless random word sequences like the ones used for training.
    - **Your own recordings**: any spectrogram with 64 gammatone bands at 100 Hz can be used as input – pass it to `Word(..., pattern=spectrogram)`.
      The README shows how to create one from a sound file.
    - **Training a model**: after installing the library, the `earshot-train` command trains a new model; see `earshot-train --help` for the options, which correspond to the `Model` parameters in section 2.
    """)
    return


if __name__ == "__main__":
    app.run()
