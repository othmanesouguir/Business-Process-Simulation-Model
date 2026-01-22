Data-Driven Business Process Simulation (BPIC17)
================================================

Hi,
this repository contains our data-driven simulator for the BPIC17 Loan Application process.
The simulation follows the BPMN control-flow and uses the BPIC17 historical log to learn the
sub-models (arrivals, durations, next activity, availability, permissions, etc.).


What you need
-------------
1) BPMN model
   - data/Signavio_Model.bpmn

2) Historical event log (CSV)
   - data/bpi2017.csv

IMPORTANT:
The simulator expects the CSV to be named exactly "bpi2017.csv" and placed inside the data/ folder.


BPIC17 dataset preparation (XES → CSV)
--------------------------------------
The original BPIC17 log is usually provided as a .xes file.
Before running, please convert the .xes into CSV and name it:

  bpi2017.csv

Then place it here:

  data/bpi2017.csv


How to run
----------
From the project root, run:

  python src/run_simulation.py

This will run the simulation and create an output event log.


Outputs
-------
After running, the generated log will be written to:

  outputs/

Example:
  outputs/simulated_log.csv

(The exact filename may depend on the current run configuration.)


Model caching (pkl files)
-------------------------
On the first run, the simulator trains the models from the BPIC17 CSV and stores them as .pkl files:

  models/

This is done once to avoid retraining every time. On later runs, the simulator loads the cached
models from models/ directly, which makes execution faster.


Notes
-----
- The first run can take longer since the models are trained and then cached.
- Please keep the filenames and paths as described above (especially inside data/).
