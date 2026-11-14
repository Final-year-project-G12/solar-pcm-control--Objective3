# Objective 3 - The Whole Story, Explained Simply

Read this top to bottom and you can explain the project to anyone, even a
ten-year-old. Numbers are the corrected ones (after the weather-year bug fix,
2026-11-13). Where something is not finished, it says so.

---

## 1. The big picture: a hot-water tank with a thermos inside

Imagine a rooftop **solar water heater**. Sunlight heats a plate (the
collector). A pump moves heat into a tank. Inside the tank we put a special wax
called a **PCM** (phase-change material). When the wax melts it *soaks up lots
of heat without getting much hotter*, and when it freezes again it *gives the
heat back*. So the wax is a **heat battery**: store sun in the daytime, use it
for a hot bath in the evening.

Objective 1 chose the **climate regions** of Tamil Nadu (3 "regimes").
Objective 2 chose the **hardware**: tank size, collector size, which wax.
**Objective 3 (this one) only chooses what to do every hour.** The hardware is
frozen. We are only the driver, not the car designer.

## 2. What can the driver do? Only three buttons

Every hour the controller presses one of:

| Button | Meaning | Like... |
|---|---|---|
| **CHARGE** | send solar heat into the tank/wax | filling a bucket |
| **IDLE** | do nothing | waiting |
| **BYPASS** | skip the tank, don't add heat | closing the tap |

## 3. The problem: the "too hot!" alarm

If we always press CHARGE, the tank gets dangerously hot (limits: water 75 C,
wax 65 C). So the system has a **safety shield**: a guard that switches the heat
off when water reaches 72 C or wax 62 C (3 degrees before the real limit).
Think of a **smoke alarm that turns off the stove**.

The always-charge rule (**B0**) is safe, but the alarm goes off **tens of
thousands of times a year**. That is wear and tear, wasted heat, and it means we
are driving by slamming the brakes instead of looking ahead.

**Our question:** can a *learned* driver give the same hot water, but make the
alarm go off far less, and never cross the real limit?

## 4. How the computer practises: the video-game idea (Reinforcement Learning)

Reinforcement learning (RL) is like training a puppy, or playing a video game
millions of times:

1. The **agent** (the driver) looks at the situation (sun now, tank
   temperatures, wax melted %, hot-water demand...).
2. It presses a button.
3. The simulator says what happened and gives a **score** (the *reward*).
4. It adjusts to get higher scores next time.

We do this on a **simulator** (the physics from Objective 2 wrapped as a
"gym" game, `pcm_env.py`) because we cannot afford to break real hardware
while learning. The simulator is hourly, 8,760 hours = one year per game.

## 5. The score (reward) - and the first big lesson

- **R1 (first try):** score = hot water delivered, minus penalties for
  breaking limits. Result: *every* algorithm just pressed CHARGE forever.
  Why? The shield already makes real limit-breaking impossible, so the
  penalty is always zero. The computer never "felt" the alarm going off, so it
  never cared. **Lesson: if something isn't in the score, the learner ignores
  it.**
- **R2 (fixed):** add a penalty of 10 points every time the shield alarm
  fires. Now avoiding the alarm pays. Pushing the penalty to 20 made learning
  *unstable*, so we kept 10.

## 6. The cast of characters (what each algorithm is, and why it is here)

### The rule-followers (no learning) - the "kids' table" to beat
- **B0 "always charge":** the Objective 2 behaviour. Safe, delivers the most,
  but the alarm screams. *Why:* it is the honest starting line; learning only
  matters if it beats this.
- **B1 "reactive rules":** if-then rules ("if it's nearly too hot, skip;
  if it's evening, hold"). *Why:* tells us whether you even need AI, or if
  sensible rules are enough.
- **B2 "rules + look-ahead":** B1 plus peeking at the next 4 hours of sun.
  *Why:* tests if knowing the future helps.
- **B4 "random":** presses random safe buttons. *Why:* a floor; if the AI can't
  beat dice, something is broken.

### The learners
- **DQN (Deep Q-Network).** Imagines a giant cheat-sheet: "in this situation,
  button X is worth Y points." A small neural network *estimates* the points.
  It sometimes presses a random button on purpose (**epsilon-greedy**: "try
  something silly 5-30% of the time") and keeps a **memory (replay buffer)**
  of past hours to re-study. *Why:* classic, small (about 10,000 numbers,
  runs in 1 ms), works with our 3 discrete buttons.
- **SAC (Soft Actor-Critic).** Two brains: an **actor** that chooses and a
  **critic** that grades. It is paid a little *bonus for staying
  unpredictable* (**entropy**), so it keeps exploring instead of getting
  stuck. Also uses a replay memory. SAC is built for smooth dials, so we show
  it a dial from -1 to +1 and cut it into three zones = the three buttons.
  *Why:* a second, different way of exploring; two independent methods
  agreeing is stronger evidence than one.
- **PPO, RecurrentPPO, MaskablePPO.** PPO learns from only its latest
  practice and takes careful small steps (stable, popular). Recurrent adds a
  memory (LSTM); Maskable hides illegal buttons (e.g. CHARGE at night).
  **They all gave up and copied B0** ("always charge"): once they got
  confident they stopped trying new things, so they never found the better
  trick. We kept them as honest negative results.
- **QR-DQN.** DQN that tracks the *spread* of possible outcomes, not just the
  average. Also got stuck on our setup.

**Why DQN and SAC win (careful wording):** both keep *forcing themselves to
explore* (random tries / entropy bonus) and both reuse old memories. PPO-family
do not. That fits the results, but it is our *interpretation*; we did not run
an experiment that isolates it.

## 7. Fair testing: the exam rules

Like school: **study** with some years, **practice test** with others, **final
exam** once.

- Train: 2016-2020 and 2024-2025 (2024-25 added by the student's instruction;
  note these were Objective 2's own test years - say so in the thesis).
- Validate (choose models): 2021-2023.
- **Locked test (final exam): 2011-2015, looked at once.**

The selection rule was fixed *before* seeing the test: (1) zero real-limit
violations, (2) delivers at least 95% of B0's hot water, (3) then the fewest
alarms. We never pick by reward alone.

## 8. The bug that nearly fooled us

The code was **secretly using 2016 weather for every year**. So "train",
"validate" and "test" were the same weather - like handing out the exam
questions as homework. Everything before 2026-11-13 was too optimistic. The
fix: load the right year's file, and **crash loudly** if a year is missing
instead of quietly substituting. Then everything was re-trained. Always check
that your exam really is different from your homework.

## 9. What we found (corrected numbers, regime 0, locked test)

| Controller | Hot water (kWh/yr) | Alarms/yr | Real-limit violations |
|---|---|---|---|
| B0 always charge | 2097 | 46,969 | 0 |
| B1 rules | 1971 | 20,588 | 0 |
| **SAC** | **2045 (97.5%)** | **21,406 (-54%)** | **0** |

DQN, per the deployment file: 96.6% of B0's delivery with 59% fewer alarms.
Efficiency about 40-41% (B0 41.6%). Decision time: DQN ~1 ms, SAC ~5 ms - the
controller only needs one decision per hour, so this is trivially fast.

**Stress tests:** weather noise up to 20%, +/-2 C ambient, wax starting
melted, worst case combined - **zero violations**. Sensor faults (dead
sunlight sensor, +2 C temperature error, ...) - **zero violations**, because
the shield reads the *true* temperatures, not what the AI was told. Caveat:
when the sun sensor dies the alarm count rises (SAC 18k to 38k), which is the
safe but costly reaction.

**Regime 1 honest finding:** SAC still learns (99% delivery, 38% fewer
alarms, validation). **DQN does not** - it fell back to always-charge (2%
fewer alarms). So "both algorithms work everywhere" is *not* supported; SAC
is the more robust choice.

## 10. What is finished and what is not (as of 2026-11-13)

- Done: bug fix; regime 0 full pipeline.
- Regime 1: both trained; DQN locked test + robustness done; SAC locked test
  pending.
- Regime 2: DQN trained; SAC resuming (~206k/300k steps); locked
  test/robustness/deployment pending.
- Not done: re-running the "Generation-2" extras (Pareto weights,
  15-minute control, domain randomisation, regime coverage). Earlier numbers
  for those came from the buggy data and must not be quoted.
- Not done: most docs/dashboards still show old numbers (the FINAL_v1 docx is
  updated - see its Section 20).
- Note: `phase6_final_test_dqn.csv` was overwritten by the regime-1 run, so
  DQN's regime-0 locked-test table needs regenerating before citing it.

## 11. Odd words, simply

- **Checkpoint / resume:** saving your video game. If the computer crashes,
  you load the save instead of starting over. (Our 300k-step SAC runs take
  5-6 hours.)
- **Domain randomisation:** practising with the sun and demand wobbled at
  random so the driver doesn't only know one perfect day.
- **Pareto curve:** the menu of trade-offs "more hot water vs fewer alarms."
  You cannot max both.
- **Forecast / smart persistence (planned):** to guess future sunshine, take
  the clear-sky amount for that hour and multiply by "how cloudy is it right
  now compared with clear?" It assumes today's cloudiness persists. Not
  started yet; currently the agent sees the true future sun (optimistic).
- **Sim-to-real gap:** all results are simulation. Real hardware is
  Objective 4.

---

## 12. Do the existing project references cover these methods? No.

`tamilnadu_pipeline/sources/references.txt` is PCM, solar-water-heating,
forecasting-survey and dataset references. Only one DRL-for-energy paper
(Emami 2026, Elsevier) is relevant, and the Sivaraj paper is about ship
steering. **Nothing there supports DQN, SAC, PPO, the shield, reward shaping,
domain randomisation, or smart persistence.**

Proposed references. **I wrote these details from memory; please open each
publisher page and confirm authors, volume, pages and DOI before you cite.**
Marked IEEE / Springer where applicable; several foundational RL papers are in
AAAI/ICML proceedings (neither publisher), which are the standard citations
for those methods.

| Method / claim | Reference |
|---|---|
| DQN | V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518, 529-533, 2015, doi:10.1038/nature14236 (Springer Nature) |
| Q-learning basis | C. Watkins, P. Dayan, "Q-learning," *Machine Learning* 8, 279-292, 1992, doi:10.1007/BF00992698 (Springer) |
| Double DQN | H. van Hasselt, A. Guez, D. Silver, AAAI 2016 |
| SAC | T. Haarnoja et al., "Soft actor-critic...", ICML (PMLR 80), 2018 |
| PPO | J. Schulman et al., arXiv:1707.06347, 2017 |
| Safety shield | M. Alshiekh et al., "Safe reinforcement learning via shielding," AAAI 2018 |
| Real-world RL challenges | G. Dulac-Arnold et al., *Machine Learning* 110, 2419-2468, 2021, doi:10.1007/s10994-021-05961-4 (Springer) |
| Domain randomisation | J. Tobin et al., IEEE/RSJ IROS 2017, doi:10.1109/IROS.2017.8202133 (IEEE) |
| Reward shaping | A. Ng, D. Harada, S. Russell, ICML 1999 |
| DRL for building energy | E. Mocanu et al., *IEEE Trans. Smart Grid* 10(4), 3698-3708, 2019 (IEEE) |
| RL for heating loads | F. Ruelens et al., *IEEE Trans. Smart Grid* 8(5), 2149-2159, 2017 (IEEE) |
| Smart persistence / forecasting | R. H. Inman, H. T. C. Pedro, C. F. M. Coimbra, *Prog. Energy Combust. Sci.* 39(6), 535-576, 2013 (Elsevier) |
| Seed variance in RL | P. Henderson et al., "Deep RL that matters," AAAI 2018 |

The same list (R1-R13) is in Section 20.10 of
`docs/objective3_plans/Objective_3_Implementation_Plan_FINAL_v1.docx`.
