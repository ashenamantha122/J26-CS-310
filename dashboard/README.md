# Local GUI for the Decision Engine

Two files:

- `export_decisions.py` — runs the real `DecisionEngine.decide()` over an
  alert JSONL file (same engine, same `rule_matrix.yaml` your other tests
  use) and writes the input+output of every decision to `decisions.json`.
- `dashboard.html` — reads `decisions.json` and shows it as a GUI: four
  headline numbers, a bar chart of containment levels, a routing summary
  (eBPF/LSM vs SDN), and a sortable/searchable table of every alert with
  the exact rule it matched.

No internet, no CDN, no pip install needed for the dashboard itself — it's
one self-contained HTML file. Safe to demo offline.

## Setup (one time)

Copy both files into your `~/project/` folder (next to `decision_engine/`
and `config/rule_matrix.yaml`), so `export_decisions.py` can import the
same engine your other tests use:

```
cp export_decisions.py dashboard.html ~/project/
```

## Run it

```
cd ~/project
python3 export_decisions.py c3_sample_alerts_100.jsonl
```

This writes `decisions.json` next to it. Any alert JSONL works — your
10-alert file, the 100-alert file, or a real feed later.

Then serve the folder and open it in a browser. Browsers block a page from
`fetch()`-ing a local file when you just double-click the HTML (file://
URLs are sandboxed), so start a tiny local server first:

```
python3 -m http.server 8000
```

Then open **http://localhost:8000/dashboard.html** in a browser on the
same machine. Leave the `http.server` command running in that terminal
for as long as you want the page open — close it with Ctrl+C when done.

## Re-running after a change

Whenever you edit `rule_matrix.yaml` or generate a new alert dataset,
re-run `export_decisions.py` and refresh the browser tab — no server
restart needed, the page just re-fetches `decisions.json`.

## For the PP1 demo

This is the same decision logic as `tests/test_decision_engine.py` and
`supervisor.py` — `export_decisions.py` doesn't change any behavior, it
just captures the same `decide()` output as structured JSON instead of a
printed line. If the panel asks "how do you know this number is right,"
the table's **Rule matched** column names the exact row in
`rule_matrix.yaml` behind every decision, and that file is the thing you
hand them to check it by hand.
