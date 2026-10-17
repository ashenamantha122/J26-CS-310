# Patch for your existing `containment_switch.py`

Your current `_watch_trigger()` reads one line and discards the rest
of the file if more than one command was ever written before the
next poll. Replace it with this version, which processes every queued
line in order, then clears the file:

```python
def _watch_trigger(self):
    """Poll TRIGGER_FILE for one or more queued commands, each on its
       own line:
         ISOLATE <ip>
         RELEASE <ip>
       All queued lines are processed, in order, each poll cycle, then
       the file is cleared. This handles multiple containment decisions
       arriving faster than POLL_INTERVAL without dropping any of them."""
    while True:
        if os.path.exists(TRIGGER_FILE):
            try:
                with open(TRIGGER_FILE) as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                for line in lines:
                    self._handle_command(line)
            except Exception as e:
                self.logger.error("Failed to read trigger file: %s", e)
            finally:
                try:
                    os.remove(TRIGGER_FILE)
                except OSError:
                    pass
        time.sleep(POLL_INTERVAL)
```

This is the only change needed — `_handle_command()`, `isolate_host()`,
and `release_host()` stay exactly as they are in your current version.

## Why this matters now

It didn't matter during your manual tests (`echo "ISOLATE ..." > ...`
with a few seconds between commands), but `supervisor.py` can emit
several containment decisions within milliseconds of each other when
processing a batch of alerts — which is exactly what will happen when
you replay your real alert dataset or run a live demo with more than
one triggering event close together. Without this fix, some of those
ISOLATE/RELEASE commands would silently never reach your switch.

## Apply this on your actual Kali/Ubuntu machine

1. Open your working `containment_switch.py`
2. Replace the `_watch_trigger` method with the version above
3. Restart the controller (`python osken_launcher.py containment_switch.py`)
4. Re-verify with your existing test sequence (isolate one host, confirm
   `pingall`, release, confirm `pingall` again) before moving on to
   wiring `supervisor.py` in
