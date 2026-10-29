# Results

One folder per run in `runs/`, exactly as the stick wrote it. Licence: [CC BY 4.0](LICENSE). The format, its
version and the compatibility rules: [FORMAT.md](FORMAT.md).

```
runs/<UTC time>-<machine>-<random>/
    run.json          format version, stick version, machine (salted hash), CPU, board, RAM, GPUs, every step's
                      result and the exact tool versions behind it
    manifest.json     every component's version: system image, seed, and what actually ran
Credit lives outside the run folders (which stay exactly as the stick wrote them): `results/attribution/<run>.json`
holds `{"github": "<username>"}` — shown on the data page with a link to the profile — or `{"withheld": true}` when the
submitter chose to keep their username out of the stored result data and off the page. Git history and the
submission itself show who contributed either way.

    inventory/        lscpu, dmidecode, lshw, lspci, memory SPD, nvidia-smi … (serials, UUIDs, MACs removed);
                      pcie.json: every card's path through root ports and switches, with each link
    q/<question>/     bench runner output per question (results.jsonl, samples, baseline)
    vkprobe/ rocm/ oneapi/ llama/ mlc/ opencl/ blender/   raw output per tool
    samples-drm.jsonl AMD / Intel GPU clocks, busy share, power every 200 ms
    stick.log
```

`<machine>` is a salted hash of the board's identifiers: runs of the same machine group together, but the machine
itself is not identifiable from it.

## Adding runs

See [Contributing runs](../README.md#contributing-runs): `tools/contribute.sh` or an upload on GitHub, both ending
in a pull request that `tools/validate.py` checks — only new run folders, exactly the files the stick writes, from
a finished run (`status` "done" or "done with failures"), no MAC addresses or identifier lines. By sharing runs you
agree to their publication under CC BY 4.0 and confirm the machines' owners agreed.

Check runs yourself before sharing: `python3 tools/validate.py results/runs/<run>`.
