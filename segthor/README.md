# Running this nnU-Net fork on SegTHOR (Snellius)

Self-contained: everything here works from a plain clone of **this repo alone**
(`git@github.com:bellavg/nnUNet.git`). No other project needs to be on Snellius.

## Label mapping -- read before you train

`segthor/prepare_segthor_dataset.py` writes `dataset.json` with `--label-order
verified` by default: `{background:0, aorta:1, heart:2, trachea:3, esophagus:4}`.
This matches the actual voxels (sagittal aortic-arch shape, axial
bifurcation signature at the arch, ~274mm cranio-caudal extent -- label 1 is not
a straight uniform-caliber tube, which rules out esophagus). The published
SegTHOR challenge's own label order is the reverse for indices 1 and 4
(`esophagus:1, ..., aorta:4`) -- pass `--label-order official` if you want
dataset.json to use those names instead. Neither flag changes which voxels get
which index; both agree index 4 has **zero voxels in every patient** in this
release, whichever name you attach to it. That class will not be learned or
scored no matter which order you pick.

## 1. Get the code onto Snellius

```bash
ssh your_username@snellius.surf.nl
git clone git@github.com:bellavg/nnUNet.git ~/nnUNet
cd ~/nnUNet
mkdir -p logs
```

## 2. Get the data onto Snellius

From your own machine (not Snellius), copy just the data zip -- one file, one
command:

```bash
scp data/segthor_part1.zip your_username@snellius.surf.nl:~/nnUNet/data/segthor_part1.zip
```

If a terminal command is unwelcome, use a GUI SFTP client instead (e.g.
Cyberduck, free): connect to `snellius.surf.nl` with your username/password,
then drag the zip from Finder/Explorer into `~/nnUNet/data/` on the remote side.

Then on Snellius, unzip it:

```bash
cd ~/nnUNet/data && unzip -q segthor_part1.zip && rm -f segthor_part1/.DS_STORE
```

You should end up with `~/nnUNet/data/segthor_part1/train/Patient_01/` etc.

## 3. Run, in order

```bash
cd ~/nnUNet

# a. Build the venv (once, or after a module update breaks it)
sbatch segthor/slurm/rebuild_venv.job

# b. Convert data + preprocess (wait for (a) to finish)
sbatch segthor/slurm/preprocess.job

# c. Train all 5 CV folds in parallel (wait for (b) to finish)
sbatch --array=0-4 segthor/slurm/train.job
# or a single fold:
#   FOLD=0 sbatch segthor/slurm/train.job

# d. Once training finishes: predict/ensemble
INPUT_DIR=/path/to/images OUTPUT_DIR=/path/to/predictions \
    sbatch segthor/slurm/predict.job
```

Check progress with `squeue -u $USER` and `cat logs/<jobname>_<jobid>.out`
between steps. (a)->(b)->(c) must run in that order; (d) is whenever you want a
prediction, using folds trained so far.

## Env vars, summarized

| Variable | Set by | Points to |
|---|---|---|
| `nnUNet_raw` | preprocess/train/predict jobs | `$HOME/nnunet_segthor/nnUNet_raw` |
| `nnUNet_preprocessed` | preprocess/train/predict jobs | `.../nnUNet_preprocessed` |
| `nnUNet_results` | preprocess/train/predict jobs | `.../nnUNet_results` |
| `VENV_PATH` | you, at `sbatch` time | default `$HOME/.venv_nnunet` |
| `BASE_DIR` | you, at `sbatch` time (or `SLURM_SUBMIT_DIR`) | this checkout, default `$HOME/nnUNet` |
