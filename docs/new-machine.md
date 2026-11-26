# Running on a new machine

How to run the modelling experiments on a machine other than the course VM.
You don't need the raw exchange captures, the ITCH parser or `/db_pcaps`:
the cleaned order book data for all 61 study days is in the team's Box
folder, and every script reads from there.

The worked example is **E10, the direction model** (ticket T7b). The other
model experiments (E4–E6) follow the same pattern; their commands are on the
[Experiments](experiments.md) page.

!!! warning "The data is licensed"
    The Nasdaq data and everything built from it is for this course only.
    Never share it with a public link, never commit it to git, never upload it
    anywhere other than the team Box folder. Only people allowed to use the
    course data should be given access.

## 0. What you need

| What | Details |
|---|---|
| GitHub | access to `FIN-556-Market-Microstructures/project_repo` |
| Box | an invitation to the team's `fin556` folder, from its owner. Accept it in Box before you start |
| Machine | Linux (or macOS), Python 3.9+, `git`, `tmux`, internet access |
| Memory | 32 GB RAM minimum, 64 GB or more comfortable |
| Space | about 3 GB for the E10 data, in RAM (`/dev/shm`) or on disk |

The Box folder holds:

| Folder | Contents | Size |
|---|---|---|
| `fin556/silver` | per day: messages, trades, order book (5 levels), quality report | 13.4 GiB |
| `fin556/gold` | features on a 100 ms clock, per day (E4–E6) | 1.1 GiB |
| `fin556/gold_event10` | features at every 10th book event, per day (E10); built in step 3 | ~1.5 GiB |
| `fin556/results` | results of each experiment | small |

## 1. Get the code

```bash
git clone https://github.com/FIN-556-Market-Microstructures/project_repo.git
cd project_repo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q          # should end with "passed" and no failures (~2 min)
```

## 2. Connect to Box with rclone

Use **your own** Box login. Never copy someone else's `~/.config/rclone/rclone.conf`:
it contains their Box access token.

**Install rclone.** Without admin rights, put it in `~/bin`:

```bash
mkdir -p ~/bin && cd /tmp
curl -LO https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip -o rclone-current-linux-amd64.zip && cp rclone-*-linux-amd64/rclone ~/bin/
export PATH="$HOME/bin:$PATH"          # add this line to ~/.bashrc too
cd ~/project_repo && rclone version
```

**Create the remote**, named `box` (the scripts assume that name):

```bash
rclone config
#   n                      (new remote)
#   name>       box
#   Storage>    box        (type "box" or pick its number)
#   client_id, client_secret, box_config_file, access_token: press Enter (blank)
#   box_sub_type>          user
#   Edit advanced config?  n
#   Use web browser to automatically authenticate?
#       y  if this machine has a browser: it opens, you log in to Box
#       n  if it has none (SSH only): run the command it prints,
#          rclone authorize "box", on a computer that has a browser and rclone,
#          log in, and paste the token it gives back here
#   Keep this "box" remote?  y
#   q                      (quit)
```

**Check you can see the team folder:**

```bash
rclone lsd box:fin556            # should list gold, results, silver (and gold_event10 if built)
```

If it says "directory not found", the invitation hasn't been accepted yet, or
the folder sits somewhere else in your Box: find it with `rclone lsd box:`.

## 3. Get the E10 data

**If `rclone lsd box:fin556` already lists `gold_event10`**, someone has built
it. Just copy it:

```bash
mkdir -p /dev/shm/fin556
rclone copy box:fin556/gold_event10 /dev/shm/fin556/gold_event10 --transfers 8
ls /dev/shm/fin556/gold_event10 | wc -l          # 61
```

**Otherwise, build it** from the order books in `silver` (about 15–30
minutes with 2 jobs; each day is downloaded, turned into features at every book
event, labelled with the direction of the next price move, thinned to every
10th event and saved):

```bash
python scripts/build_features.py --source box:fin556/silver \
    --dest /dev/shm/fin556/gold_event10 --sample event --keep-every 10 --jobs 2
ls /dev/shm/fin556/gold_event10 | wc -l          # 61
rclone copy /dev/shm/fin556/gold_event10 box:fin556/gold_event10 --transfers 8   # so nobody has to rebuild
```

Each job holds one whole day of book events in memory: a few GB on a normal
day, but up to ~25 GB on the busiest days (7–10 April, the tariff selloff).
Use `--jobs 1` on a 32 GB machine, `--jobs 2` with 64 GB, up to `--jobs 4`
with 128 GB.

No `/dev/shm`, or too small (`df -h /dev/shm`)? Use any folder on disk
instead, e.g. `--dest ~/fin556/gold_event10 --work ~/fin556/work`, and pass
that folder as `--gold` in step 4.

## 4. Run E10

In tmux, so it keeps running if you disconnect (`Ctrl-b` then `d` to detach,
`tmux attach -t e10` to come back):

```bash
tmux new -s e10
source .venv/bin/activate
python scripts/run_direction.py --gold /dev/shm/fin556/gold_event10 \
    --out results/e10 --threads 16 2>&1 | tee e10.log
```

- `--threads`: the number of CPU cores LightGBM may use. Check with `nproc`
  and leave some free on a shared machine.
- It prints progress lines with times. Expect about 1 hour, up to 2 on a
  smaller machine; most of it is LightGBM. It needs about 20 GB of RAM.
- It finishes with `done in X min; results in results/e10`. A Python
  `Traceback` instead means it failed: send the last 30 lines of `e10.log`.
- Short on memory (`MemoryError` or the machine freezing)? Add `--every 2`
  to use half the rows, and say so when reporting the results.

## 5. Send the results back

```bash
rclone copy results/e10 box:fin556/results/e10
rclone copy models box:fin556/models          # the trained models, needed for T8
rclone copy e10.log box:fin556/results/e10
```

Results and models go to Box, never to git.

## 6. Clean up

```bash
rm -rf /dev/shm/fin556          # frees the RAM; the data stays in Box
```
