# KIKIGAKI Vocal

This repository contains the source code of the interactive music transcription editor developed for the following paper:

Yu Sugimoto, Ryo Nishikimi, and Eita Nakamura, "Dual-Time Representation Framework for Interactive Music Transcription," Proc. ISMIR, 2026.

## Installation

The editor has been tested on **macOS with Python 3.12**. It may not work properly on other operating systems.

### 1. Install Python dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### 2. Install FluidSynth

[FluidSynth](https://www.fluidsynth.org/) is required for MIDI playback. Please install it according to your operating system.

### 3. Download the SoundFont

The editor uses `MuseScore_General.sf2` for MIDI playback. This file is not included in this repository.

Download `MuseScore_General.sf2` from the [MuseScore General SoundFont repository](https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/) and place it in the same directory as `kkgk_editor.py`.

**Do not rename the file.** The filename must remain exactly `MuseScore_General.sf2`.

The SoundFont is distributed separately under the MIT License.

## Usage

Run the editor using:

```bash
python kkgk_editor.py
```

For instructions on how to use the editor, please refer to the [User Manual](manual.pdf).
