# Audio-guided horizontal hand responsiveness, protocol v1

This is a **prospective protocol for new recordings**. It does not change the
windows, thresholds, or verdict of the first manual-stopwatch experiment. That
experiment retained 1,199 valid observations and passed independent DSP replay;
its four intended responsiveness legs remain inconclusive because the actual
movements did not match the declared windows.

The new launcher belongs to Engineer 4's offline evaluation tools. It does not
change the filter, tracker, model, core engine, or mouse controller. It provides
instructions and records when those instructions were issued. It does not
measure the participant's physical reaction time.

## Two separate experiments

Run and review **slow horizontal** first. Run **fast horizontal** separately only
after reviewing the slow recording. Each experiment has one outward movement
and one return, with long holds between them. Report both attempts, including
failures or inconclusive results; do not select only the better direction.

- Remain on `codex/engineer4-dsp-recorder` and use the verified executable.
- Hand parameters remain `1.2,4,1`; gaze is disabled.
- Use `--filter dsp --os log --no-gaze --seconds 55`.
- Keep the tray paused. Do not change settings during the recording.
- Use the same relaxed, open-hand pose, with all fingers naturally extended.
- Place two comfortable reference marks about 20 cm apart, at the same height
  and depth. Align the index fingertip with each mark. Keep the whole hand
  visible at both ends; avoid positioning the right endpoint near the image
  edge. The recorded normalized excursion, rather than centimeters, determines
  whether the measurement has adequate amplitude.
- Move the whole hand horizontally without rotating the palm. Follow the
  spoken instructions and tones; there is no stopwatch schedule to memorize.
- Slow: take approximately **3 seconds** between movement tones. Fast: take
  approximately **0.5 seconds**, briskly and comfortably rather than flicking.
- Hold the position when instructed. Do not add practice movements during a
  recording. Tell the reviewer if a cue was missed or an extra movement occurred.

The 55-second engine duration provides startup and finishing margin. At the
recent observed rate of about 20 Hz it would produce about 1,100 records, below
the recorder's 4,096-frame capacity. Actual counts, overflow, timing, and
completion must still be verified for every run.

## Readiness and cue sequence

The launcher waits for **two consecutive periodic status reports** showing
positive capture and AI processing, finite normalized Neutral hand coordinates,
and an increasing OS-submission count. A camera light or startup banner is not
sufficient. Readiness is only a launch condition: periodic reports cannot prove
that every intervening frame was valid. The complete trace supplies that evidence.

The first periodic report must arrive within 30 seconds of process launch.
Readiness must follow within 8 seconds of the first report. Otherwise, do not
issue movement commands and mark the guided session unsuccessful. The existing
engine still has its own finite run duration.

The following offsets are relative to the launcher's readiness event, not to the
recorder's capture-clock epoch. Let `D = 3` seconds for slow and `D = 0.5` seconds
for fast; let `E = 11 + D`, and `F = E + 14 + D`.

| Offset | Instruction or signal |
| --- | --- |
| 0 s | Hold LEFT; keep the hand visible and open. |
| 5 s | Prepare to move RIGHT. |
| 8, 9, 10 s | Spoken countdown: three, two, one. |
| 11 s | High start tone: move RIGHT. |
| E | Low arrival tone and instruction: hold RIGHT. |
| E + 8 s | Prepare to move LEFT. |
| E + 11, E + 12, E + 13 s | Spoken countdown: three, two, one. |
| E + 14 s | High start tone: move LEFT. |
| F | Low arrival tone and instruction: hold LEFT. |
| F + 8 s | Finished; keep still until the application closes. |

The last instruction is scheduled 39 seconds after readiness for slow, and
34 seconds after readiness for fast. Prompts are prepared before camera launch.
Speech playback must not block the process-output reader. The launcher records
planned and actual cue-issue times, startup/readiness, process exit, and any
late, failed, or cancelled cues. An early process exit cancels pending commands.

An issued playback call is **not a measurement of when sound reached the ear**.
Speaker latency, speech duration, operating-system scheduling, and participant
reaction time remain outside this measurement. The launcher and recorder have
different clock origins. Do not subtract cue timestamps from capture timestamps
to claim camera latency or exact movement onset.

## Data and preservation requirements

Each manual launch creates a new session outside the repository. Preserve:

- The unedited numerical recording, completion marker, header, and footer.
- Console output, source/build identity, command, and actual filter parameters.
- The separate bounded cue/event log, including failures or truncation.
- This protocol version, the selected slow/fast mode, and the participant's
  account of missed commands, extra movements, or loss of hand visibility.
- Paired CSV, analysis results, and the rules used to select samples.

The recorder continues to capture the same frame IDs, raw input, actual DSP
output, published point, precise observation time, freshness, state, and reset
metadata. The cue launcher does not rewrite those measurements. The original
manual-stopwatch recording and all earlier hand/gaze recordings stay untouched.

The existing importer validates complete recordings and paired observations.
`evaluate_holds.py` measures stationary dispersion; **it is not a movement
scorer**. The selection and timing calculations below must be independently
checked before any responsiveness PASS is reported. Implementing or running the
audio launcher alone does not complete that analysis.

## Prospective, raw-only movement labeling

This replaces the difficult manual-clock windows **only for future audio-guided
trials**. It does not move any windows in the earlier recording. Labels depend
on the raw tracker trajectory and the prescribed order of movements, never on
whether the filtered output would pass a timing limit.

Use the precise recorded DSP timestamp. Define zero as
`ceil(first_stored_timestamp_seconds * 1000) / 1000`, retaining the same rounding
convention used for earlier experiments. Do not shift zero to the first good
sample or to a cue.

An eligible row has all of the following: detector count 42, valid upstream
result, fresh `Tracked` input, Neutral state 0, `Filtered` DSP status, a published
point, and finite normalized raw/filtered X/Y. Preserve every chronological row
for replay, including initialization and excluded observations.

1. **Initial LEFT reference:** use the fixed half-open window `[3,5)` seconds.
   It must contain at least 25 eligible observations, no ineligible observation,
   and no adjacent observation gap greater than 100 ms. Define its mean raw X/Y
   and raw X population standard deviation. Do not substitute a later baseline
   if this one is unsuitable.
2. **Stable-window definition:** a nominal two-second half-open window must have
   at least 25 eligible observations, no ineligible row or gap over 100 ms, raw X
   standard deviation at most 0.01, raw X range at most 0.03, and absolute
   ordinary-least-squares X-versus-time slope at most 0.005 normalized units/s.
   These are initial engineering labeling limits, not universal tracking
   standards. Apply them to the initial reference and all arrival windows.
3. **First departure:** after the reference window ends, locate the first run of
   three consecutive eligible observations whose signed X displacement from the
   reference mean is consistently greater than `max(0.03, 6 * reference_SD_X)`
   in magnitude. The first expected direction is increasing X, consistent with
   the current mirrored pipeline and earlier LEFT-to-RIGHT recordings. A first
   sustained departure in the wrong direction makes the attempt unsuitable;
   do not skip it and choose a later movement.
4. **First arrival:** examine two-second candidate windows starting at successive
   observations after departure. Choose the first complete stable window whose
   raw mean X differs from the starting reference by at least 0.15 in the
   expected direction. The full window must end within 15 seconds of departure
   confirmation. Its raw mean is the RIGHT reference. If none qualifies, mark
   this attempt inconclusive; do not extend the search or substitute a later
   successful attempt.
5. **Return:** use that accepted RIGHT window unchanged as the return's starting
   reference. After it ends, apply the same first-departure and first-arrival
   rules, now expecting decreasing X. Do not slide the RIGHT reference to a
   quieter later hold. The first qualifying return window is the LEFT endpoint.
6. **Candidate crossing search:** for each leg, search from the end of its
   starting reference window through the end of its arrival window. The
   departure threshold identifies the attempt; it is **not** the 10% crossing
   and must not truncate earlier movement samples.
7. **Ambiguity:** a pause partway through a movement, extra practice movement,
   wrong-direction departure, or continued outward travel after the putative
   arrival can make the first endpoint ambiguous. Report this and the first
   attempt; never replace it with a later endpoint just because results improve.
   Cue records and the participant's account help identify such deviations, but
   do not create frame-exact movement timestamps.

If either baseline, direction, or candidate is unsuitable, preserve the run and
report the reason. Do not call a missing, undersampled, or mislabeled movement a
DSP failure. Do not smooth the raw signal again merely to make labeling pass.

## Responsiveness calculations

For each eligible leg, let `a` and `b` be the mean raw X values of the fixed
starting and arrival reference windows. Use those **same** endpoints for raw
and filtered progress:

```text
amplitude = abs(b - a)
progress = (x - a) / (b - a)
```

Require amplitude at least 0.15 and absolute difference between raw endpoint mean
Y values no greater than 0.05. Keep raw and filtered X/Y units normalized; image
coordinates do not establish a distance in centimeters or screen pixels.

Locate the first directional 10%, 50%, and 90% crossings in the candidate search.
Record recrossings and any non-monotonic movement; do not pick later crossings
because they produce a smaller delay. A crossing lies between the last below-
threshold observation and the first at/above-threshold observation. Use actual
timestamps, not a nominal 20, 30, or 60 FPS.

For a raw crossing bracket `[r0,r1]` and filtered bracket `[f0,f1]`:

```text
added crossing delay interval = [f0 - r1, f1 - r0]
raw 10-to-90 duration interval = [raw90_low - raw10_high,
                                 raw90_high - raw10_low]
```

Report 50% and 90% added delay for both legs. Do not claim sub-frame precision
from interpolation. A negative lower bound can reflect sampling uncertainty; it
does not prove prediction or zero physical latency.

For settling, use the same endpoint `b` and common band for both streams:

```text
band = max(0.05 * amplitude, 3 * arrival_raw_SD_X)
```

If the band exceeds 0.10 times amplitude, the endpoint is unsuitable for this
settling assessment. Otherwise, after each stream's 90% crossing, find its first
entry into `b +/- band` followed by at least 500 ms of observed in-band samples.
Use the preceding outside-band observation and the entry observation as the
settling bracket. If the transition into the band is not bracketed, report the
measurement limitation. There must be no invalid row or gap above 100 ms within
the observed dwell. Compare raw and filtered settling brackets with the same
interval-subtraction rule. Observed dwell does not prove behavior between frames.

Also report X and 2D RMS filtered-minus-raw difference, maximum displacement
difference, endpoint mean difference, and raw/filtered overshoot relative to the
same endpoint. For movement-fidelity statistics use matching rows from the raw
10% crossing's preceding sample through the later raw/filtered 90% crossing's
upper sample. For endpoint differences use the fixed arrival window. For
overshoot use the raw 90% crossing through the arrival-window end, counting
only displacement beyond the final endpoint in the movement direction. These
are descriptive results, not physical tracking-accuracy measurements.

Do **not** apply the stationary evaluator's 500 ms wash-in to movement onset.
That would discard the transient being measured. Replay every accepted update
in order with the existing configuration, even if a row cannot be scored for
responsiveness. Never reset the offline replay at an audio cue or label boundary.

## Predeclared verdicts

The following timing limits are initial project engineering targets, not claims
about universal human perception or optimal parameters.

| Check | Requirement |
| --- | --- |
| Recording | Complete, unchanged configuration, all submitted frames accounted for, no overflow, correct pairing, independent mathematical replay passes. |
| Baselines and amplitude | The raw-only reference/arrival rules above pass; excursion >= 0.15; endpoint mean Y difference <= 0.05. |
| Movement integrity | Fresh, Neutral, published, accepted `Filtered` observations through the scored movement and both settling dwells; no Held/Lost, reset, rejection, nonfinite values, or boundary saturation. |
| Timing resolution | No adjacent gap > 100 ms; at least three observed intervals span the raw 10-to-90 transition. |
| Slow movement | Raw 10-to-90 duration interval lies wholly within 1.5-3.5 seconds. |
| Fast movement | Raw 10-to-90 duration interval lies wholly within 0.15-0.8 seconds. |
| Added 50% and 90% delay | At most 200 ms for each threshold. |
| Added settling delay | At most 250 ms using the common capped band and observed 500 ms dwell. |

For an otherwise valid delay measurement, **PASS** requires the uncertainty
interval's upper bound to be within its limit. **FAIL** requires its lower bound
to exceed the limit. An interval straddling the limit is **INCONCLUSIVE at this
sampling resolution**. The same conservative treatment applies to a movement-
duration interval crossing a speed-category boundary; an out-of-category speed
is a protocol issue, not a DSP defect.

If raw input never establishes an adequate movement or endpoint, the leg is
unmeasurable. If suitable raw input and clean observation coverage exist but the
filtered signal fails to reach the required threshold or settle before the
arrival-window end, report that demonstrated response failure rather than
silently omitting it. Distinguish recording failure, input/protocol failure,
sampling uncertainty, and a valid DSP performance failure in the final report.

Report both directions separately. A successful trial requires both prescribed
legs to meet the data and responsiveness requirements. Never discard a failed
trial or pool it away with a later success.

## What this does and does not complete

This measures DSP smoothing delay and fidelity relative to the live tracker
input. It does not establish physical fingertip accuracy, camera/AI/queue delay,
OS action latency, per-update CPU cost, or the time at which a sound was heard.
The numerical recorder certifies submission to Engineer 5's receiving boundary,
not an acknowledgement of physical mouse movement. Real mouse control stays off.

The old manual horizontal experiment, the hand stationary result, and both gaze
stationary trials remain part of the evidence. New audio-guided trials do not
retroactively change their verdicts. After horizontal slow/fast review, remaining
work includes vertical hand responsiveness, gaze responsiveness, immediate
direction reversals, tracking loss/reacquisition, combined hand/gaze integration,
and final regression tests and documentation.

No live experiment is authorized by this document. The user manually approves
and launches each recording; offline preparation and tests must not open the
webcam, enable real mouse input, or play audio unexpectedly.
