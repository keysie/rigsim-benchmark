# Results

One folder per run in `runs/`, exactly as the stick wrote it. Licence: [CC BY 4.0](LICENSE).

```
runs/<UTC time>-<machine>-<random>/
    run.json          what ran: machine (salted hash), CPU, board, RAM, GPUs, driver, status of every step
    inventory/        lscpu, dmidecode, lshw, lspci, memory SPD, nvidia-smi … (serials, UUIDs, MACs removed)
    q/<question>/     bench runner output per question (results.jsonl, samples, baseline)
    vkprobe/ rocm/ oneapi/ llama/ mlc/ opencl/ blender/   raw output per tool
    samples-drm.jsonl AMD / Intel GPU clocks, busy share, power every 200 ms
    stick.log
```

`<machine>` is a salted hash of the board's identifiers: runs of the same machine group together, but the machine
itself is not identifiable from it.

## Adding runs

```bash
tools/import.sh /run/media/$USER/RIGSIM     # copies new runs here, rebuilds docs/data.json, lists what is new
```

Before committing, for every new run:

1. **Consent:** the machine's owner agreed to publication (the stick's start screen says results may be published).
2. **Review:** nothing identifying slipped through — `grep -ril -e "$(hostname)" -e serial results/runs/<run>` and a
   look at `inventory/` and `stick.log`.
3. Commit the run folders together with `docs/data.json`.
