# FIVRS Proctor Guide

How to run a participant through the **resident study** (automated feedback) or the
**graphics study** (two renderers) on the FIVRS station. One-page participant summaries
are in [`study_descriptions/`](study_descriptions/).

---

## 1. Before each participant

1. **Quest headset:** charged and on. On this computer, open the **WiVRn dashboard**.
   In the Quest, open **WiVRn** and connect to this computer.
2. **Launcher:** double-click the **FIVRS** icon on the desktop
   (`scripts/study_gui/launch_study_gui.sh`). Open the **Study** tab and choose
   **Resident study** or **Graphics study**.
3. **System status:** the box must be all green before Start is enabled.
   - **WiVRn server** and **Quest headset (connected)**
   - **Haptic device (Touch)**, **Foot pedal** and **Microscope handle button**
   - **RTX renderer**, **Feedback distance maps** and **Simulator closed**

   Press **Recheck** after fixing anything.
4. **Development boxes:** make sure **Practice run**, **Desktop view** and **Rehearse**
   are **unticked** for a real participant. A practice run is saved separately and never
   uses a participant Study ID.
5. **Stylus:** seat the Touch stylus in its dock. Have the participant sit at the rig
   with the stylus in their dominant hand and the foot pedals under their feet.

---

## 2. Simulator controls

| Action | How |
|---|---|
| Move the drill | Move the Touch stylus |
| Reposition the stylus without moving the drill | Hold the stylus **action button**, move, release |
| Recenter the drill in view | **Double-tap** the stylus action button |
| Run the drill | **Hold the right pedal** |
| Move the microscope (camera) | **Hold the microscope handle button** and move the head |
| Zoom in / out | Microscope button **+ left pedal** / microscope button **+ middle pedal** |
| Next tutorial page | **Tap the left pedal** (when the step is done) |
| Open the pause menu | **Hold the left pedal** (about half a second) |

**In the pause menu** (the simulator is frozen and the trial clock stops):

| | |
|---|---|
| Move down the list | Tap the left pedal |
| Select | Middle pedal |
| Leave a list | Select **Back** (the last row of each list) |
| Close the menu | Select **Resume**, or hold the left pedal and choose **Yes** |

Menu rows: **Saved camera views** · **Burr type** (cutting / diamond) · **Burr size**
· **End trial** (only during the timed trial) · **Resume**. *End trial* asks to confirm,
with **No** highlighted, so a stray tap cannot end the trial.

---

## 3. Resident study

### 3.1 Start

1. Enter the participant's name as `lastname_firstname` (for example `smith_jane`).
   The name stays only in the study key; the session gets a **Study ID**.
2. Press **Start study session**. The simulator opens with the RTX view in the headset,
   and the bone loads (about 20–40 s).
3. Help the participant put on the headset.

### 3.2 Tutorial (in the headset)

Each page turns green when its step is done. The participant then **taps the left pedal**
to move on. The proctor panel shows the current page and whether it is done.

| # | Page | What the participant does |
|---|---|---|
| — | Welcome | Read; tap the left pedal |
| 1 | Moving the drill | Move the stylus around the field |
| 2 | Repositioning the haptic device | Hold the action button, move the stylus, release |
| 3 | Recentering the drill in view | Double-tap the action button |
| 4 | Moving the microscope | Hold the microscope button and move the head until the yellow target is inside the centre circle |
| 5 | Zooming in and out | Microscope button + left pedal (in), then + middle pedal (out). Keep the microscope button held, or the left pedal moves to the next page |
| 6 | Activating the drill | Hold the right pedal and remove some bone. The camera returns to its start-up view and zoom when this page opens. Practice bone is restored before the trial |
| 7 | The pause menu | Hold the left pedal to open it, then close it (instructions appear above the menu) |

When all pages are done, the headset reads **Tutorial complete**. The participant may keep
practising, then **taps the left pedal** to begin the study session.

**Proctor controls during the tutorial:**
- **Skip tutorial step:** moves a stuck participant to the next page. The Right arrow key
  on the keyboard does the same.
- **Begin timed trial:** starts the study introduction without waiting for the participant.
- **Recenter headset view:** places the view again from where the participant's head is
  now.
- **Tilt:** how far below the participant's line of sight the view sits (default 15°).
  Changing it re-places the view straight away.

### 3.3 Study introduction and timed trial

1. **Introduction:** it covers the view while the bone is restored, the burr goes back to
   6 mm cutting and the recording starts. The participant reads it and taps the left pedal.
2. **Trial:** the panel reads **Begin drilling when ready**. The **10-minute timer** starts
   at the first bone removed. Time in the pause menu does not count. A **1 minute
   remaining** notice appears near the end.
3. **The task:** a cortical mastoidectomy, identifying the short process of the incus.
   Finishing is not required.
4. **Finishing early:** the participant holds the left pedal, selects **End trial** and
   confirms **Yes**. The proctor can also press **End trial early**.
5. **End:** the drill locks and the headset reads *Session complete. Please remove the
   headset.* A **Session complete** box appears on the monitor. Press **Proceed**.

### 3.4 Feedback reporting system (browser, on the monitor)

1. **Background questionnaire:** it opens straight away in the browser while the session
   is analysed (about a minute). The page continues on its own when the feedback is ready.
2. **Initial self-assessment:** one rating of their own performance.
3. **The two feedback formats,** in the order chosen automatically for this participant
   (it alternates across participants). Each is followed by a four-question
   questionnaire.
   - **Video-Based Feedback.** Safety and drilling-motion metrics, each opening a table
     of selected moments (Event #, Duration, Phase, Expert comparison or Structure,
     Viewed?). Clicking a moment plays it in the **Observed event** video. Below the
     video it shows the expert distribution, the participant's value and a short
     explanation. Critical structure moments name the structure.
     **Next unlocks once every moment has been opened.**
   - **Metric-Based Feedback.** Phase tabs (Early Drilling, Antrum, Incus) with
     **Safety Metrics** and **Drilling Motion Metrics** sections. The values are compared
     with experts. Choosing a phase does not move the video.
     **Next unlocks once both sections have been opened in every phase.** When a phase
     has no safety events, Safety already reads *No safety events occurred* and counts
     as opened.
   - **What to say if Next is greyed out:** the item still to open blinks.
4. **Preference:** which format they preferred (either one, a combination, or no
   preference).
5. **Post Sim Reflection:** whether they would practise on the simulator, plus optional
   comments.
6. **Session Complete:** the participant presses **Return to the simulator study
   window**. The launcher then reads *Session complete: the resident finished the
   questionnaires and feedback.*

---

## 4. Graphics study

1. **Set up:** enter the name (`lastname_firstname`) and choose the **Compare** pair:
   - *Upstream AMBF/CHAI3D look vs FIVRS RTX*
   - *Current AMBF/CHAI3D look vs FIVRS RTX*

   The drilling time limit is 10 minutes per session.
2. **Start graphics study:** session 1 opens in a random, balanced renderer, with the same
   tutorial as the resident study (section 3.2).
3. **Session 1:** the same task. When the participant says they see the incus short
   process, press **Incus identified**; this ends the session and records the time. The
   participant can also finish with **End trial** in the pause menu. The proctor's button
   is **End session early**.
4. **NASA-TLX:** headset off, on the monitor. If the questionnaire window is closed
   without saving, use **Open questionnaire**.
5. **Start session 2:** press it once the headset is back on. Session 2 uses the other
   renderer, with no tutorial. Then the second NASA-TLX.
6. **Final questions:** the background questionnaire, then *Which session looked more
   realistic?* and *Which session would you rather use to train?*, then the Post Sim
   Reflection.

The anatomy, controls, headset framing and task are the same in both sessions; only the
graphics differ. If the simulator closes during a session, use **Restart this session**.

The graphics study panel has no Recenter or Tilt controls; its sessions use the default
15° placement.

---

## 5. Troubleshooting

| Problem | What to do |
|---|---|
| Start stays grey | Read **System status**; fix the red item and press **Recheck**. Check the name format and that the development boxes are set as intended |
| View uncomfortable, too low or off to one side | Resident study: **Recenter headset view**, or change **Tilt** |
| Participant stuck on a tutorial page | **Skip tutorial step** (or the Right arrow key) |
| Drill will not run | The pause menu may be open (close it). The drill is locked during the study introduction and after the trial ends |
| No force feedback | Close the simulator, power-cycle the Touch, then start again |
| WiVRn closes or crashes on the Quest | Reopen WiVRn on the Quest and reconnect. If there is no image, end the session and start again from the launcher |
| Simulator closed during the trial | The recording stops at that point. Resident study: press **Proceed to feedback** (no *Session complete* box appears in this case). Graphics study: **Restart this session** |
| Feedback page did not open, or analysis failed | **Proceed to feedback** prepares the reports again |
| Bone appears a moment after cutting | Normal: about 0.2 s from removal to the headset image |
