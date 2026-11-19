# Vertical hand responsiveness · SLOW · v1

This prospective experiment measures the delay that Engineer 4 adds to live hand coordinates. Score TOP → BOTTOM and BOTTOM → TOP separately. Both must pass in the same recording for the trial to pass. Preserve every attempt and its verdict.

## Coordinates and comparison with the horizontal experiment

Use normalized **image Y** for departure, arrival, progress, crossing, settling and overshoot. Use normalized **image X** for sideways alignment. Image Y increases downward, from 0 at the top to 1 at the bottom. With the default tip anchor, the integrated tracker passes image landmark 8's Y through clamping to [0,1]. The capture loop mirrors horizontally with `cv::flip(frame, mirrored, 1)`. The extractor packs image coordinates before world coordinates. These steps preserve the Y sign. Source locations are `core_engine/src/main.cpp` (capture loop), `ai_pipeline/bridge/landmark_extractors.py` (HandLandmarkExtractor) and `core_engine/include/hand_math.hpp` (HandNative::process).

The horizontal SLOW method in `RESPONSIVENESS_PROTOCOL.md` supplies the thresholds, sample selection, interval arithmetic and verdicts below. Only the movement axis, alignment axis, mark placement and spoken direction names change. Equal normalized excursions along X and Y can represent different numbers of pixels on a rectangular image. The experiment measures response relative to the tracker input; physical fingertip accuracy, acoustic reaction time and end-to-end cursor latency require separate evidence.

## Setup and cues

Place two marks on one vertical line near the middle of the camera view, at the same depth. Start about 15–20 cm apart at a comfortable forearm reach. Leave room above TOP for the fingers and below BOTTOM for the wrist. The measured normalized excursion determines acceptance. Keep the same open hand with all fingers extended, thumb separated, palm facing the camera and index fingertip aligned with the mark. Choose support that allows the whole hand to travel vertically without leaning, rotating the palm or sliding sideways. Keep camera position and lighting fixed for the run.

Use a new external `vertical-slow` session. Rehearse audio only first. For a separately approved live run, keep the tray paused, use camera 0, DSP enabled, hand only, hand parameters `1.2,4,1`, `--os log` and a 55-second engine duration. Settle at TOP before manual RECORD confirmation. Wait for two qualifying consecutive live Neutral status reports with positive capture/AI rates and an increasing OS count. The existing launcher checks the executable, branch, saved settings, single-instance condition and unused recording directory.

| Seconds from readiness | Participant action |
|---|---|
| 0 | Hold TOP. |
| 5; then 8, 9, 10 | Prepare to move down. Hold through “Three, two, one.” |
| 11 to 14 | Start downward on **MOVE!** Arrive at BOTTOM on **STOP!** |
| 14 through the next MOVE | Hold BOTTOM continuously, including through the preparation cue at 22 and countdown at 25, 26, 27. |
| 28 to 31 | Start upward on **MOVE!** Arrive at TOP on **STOP!** |
| 39 | Finished. Hold TOP until the engine closes. |

Each word pair uses one 22,050 Hz waveform with word onsets 66,150 samples (3.000 s) apart. Record the launcher's clock and cue events separately. Score movements using observation timestamps. Speaker latency and the relationship of acoustic onset to the camera clock are unmeasured. Record missed cues or extra movements with the run.

## Recording integrity and eligible observations

1. Preserve the trace, completion marker, console, manifest, cue log, copied protocol, paired CSV, participant notes and numerical verdicts. Verify configuration, digests, processed/submitted/stored counts, no overflow, unique ordered frame IDs, capture/observation timestamps and every publication. Reconcile any gaps with the console. Check cue order and completion.
2. Replay every accepted DSP update chronologically on both axes with the recorded observation times, initializations and gap policy. Compare independently with actual DSP and published values (absolute tolerance 1e-12 normalized units). Verify cached Held publications separately. Never reset replay at a cue or label boundary.
3. Define study zero as `ceil(first_stored_timestamp_seconds * 1000) / 1000`. Subtract it from precise observation timestamps. Preserve all chronological rows. A scoring row requires 42 detector rows, an OK upstream result, fresh Tracked, Neutral state 0, accepted Filtered status, and finite normalized raw and published filtered X/Y. Exclude initialization from scoring. Require uninterrupted eligible coverage through the scored movement and both dwells, without Held, Lost, rejection, reset, saturation or a gap over 100 ms. Never fill a missing crossing by interpolation.

## Fix references and departures using raw data

1. Use exactly the half-open interval **[3,5) seconds** for the initial TOP reference. Each reference or arrival is a two-second half-open window with at least 25 eligible observations, no ineligible rows and no adjacent gap over 100 ms. Compute raw mean X/Y, Y population SD, Y range and ordinary least-squares Y slope versus seconds. Require SD ≤0.01, range ≤0.03, and absolute slope ≤0.005/s. If the initial reference fails, report both legs inconclusive.
2. After the reference ends, take the first run of three consecutive eligible observations displaced consistently to the same side of the mean Y by **more than max(0.03, 6 × reference Y SD)**. First expect increasing Y. An earlier wrong-direction departure invalidates the attempt. Never skip it for a later one.
3. From successive observations after departure confirmation, choose the first complete stable window whose mean Y has moved at least 0.15 in the expected direction. The entire window must end within 15 s of confirmation. This is BOTTOM. Start the return from this exact window, expect decreasing Y, and apply the same departure and arrival rules to find the returned TOP. If BOTTOM cannot be established, the return has no valid starting reference.
4. For each leg require **absolute difference between raw endpoint mean X values ≤0.05**. Start the crossing search at the end of the starting reference and continue through the end of the arrival window. Preserve first crossings, recrossings, pauses, drift or ambiguous endpoints. Report ambiguity without moving the windows or choosing the better attempt.

## Responsiveness measurements

For start and arrival mean raw Y values a and b, use A = |b − a| and progress = (Y − a)/(b − a) for both raw and filtered series. This makes upward progress positive even though raw Y decreases. For 10%, 50% and 90%, use the first directional crossing from below threshold to at/above. Bracket it with the two observation timestamps. Require at least three observed intervals spanning the raw 10%–90% transition.

| Metric | Computation and target |
|---|---|
| Raw movement duration | [raw90 lower − raw10 upper, raw90 upper − raw10 lower]. Entire bracket must lie within **1.5–3.5 s**. |
| Added delay at 50% and at 90% | For raw bracket [r0,r1] and filtered bracket [f0,f1], report **[f0 − r1, f1 − r0]** separately at each threshold. Upper bound ≤**200 ms**. |
| Settling band | Use the same endpoint b and half-width max(0.05 × A, 3 × arrival raw Y SD). A half-width exceeding 0.10 × A makes settling unsuitable. |
| Added settling delay | After each series' 90% crossing, find first entry into the common band followed by ≥500 ms of uninterrupted observed in-band samples. Bracket entry with the preceding outside sample and first inside sample. Apply the same interval subtraction. Upper bound ≤**250 ms**. Report an unbracketed entry as a measurement limitation. |

Report fresh pair counts and gaps. For fidelity use matched observations from the sample preceding raw 10% through the later upper raw/filtered 90% sample inclusive. Report Y and 2D RMS filtered-minus-raw differences, peak absolute Y and peak 2D difference. Report mean filtered-minus-raw X/Y over the fixed arrival window. Report the furthest raw and filtered excursions beyond b in the movement direction (zero if none) from the raw 90% crossing through arrival-window end. Describe later drift separately. Do not apply stationary-jitter wash-in to movement onset.

## Per-direction decision

- **PASS** requires every integrity, input, geometry, speed, freshness and timing requirement to pass, including the upper bounds of both crossing delays and the settling delay.
- **FAIL** for a demonstrated DSP response problem requires otherwise suitable input and coverage. A delay lower bound beyond its limit, or filtered failure to cross or settle before arrival-window end when the raw measurement is suitable, demonstrates a failure.
- **INCONCLUSIVE** covers unusable or ambiguous references, unsuitable input speed or alignment, missing observations, inadequate resolution, failed recording integrity, or a timing interval straddling a required boundary. Identify numerical replay disagreements separately and investigate before accepting a timing verdict. Preserve any otherwise measurable descriptive values.

Only horizontal SLOW Trial 2 has already passed both directions. Vertical performance requires new live evidence. Horizontal FAST remains inconclusive pending Engineer 2 review.
