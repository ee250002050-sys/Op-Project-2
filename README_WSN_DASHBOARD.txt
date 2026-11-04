WSN Lifetime Optimization Dashboard
==================================

Files
-----
topology_editor_component/index.html  Drag-and-drop graph editor
wsn_dashboard.py                       Main Streamlit dashboard
wsn_lifetime_lp.py                     LP model, radio-energy model and baseline
wsn_packet_sim.py                      Discrete-time packet-level simulator
requirements.txt                       Python dependencies

Setup (macOS)
-------------
1. Keep all files and the topology_editor_component folder together.
2. Open Terminal in this project folder.
3. (Recommended) Create and activate a virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

4. Install dependencies:

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

5. Start the dashboard:

   streamlit run wsn_dashboard.py

6. Open http://localhost:8501 in your browser.

Network architecture
--------------------
The network contains n_sensors ordinary sensor nodes, one additional
battery-powered node labelled SOURCE, and one SINK. Every ordinary sensor
continues to generate its own measurement traffic, and the designated SOURCE
also generates traffic. Both ordinary sensors and SOURCE can forward traffic
and have battery constraints. The SINK is the sole destination. Thus the
SOURCE is a real LP node, not just a visual tag.

Main features
-------------
- Six tabs: Overview, Network Simulator, Energy Simulation,
  Optimization & Analysis, Experiments & Reports, and Packet Simulation &
  Failures.
- Drag nodes, add links by clicking two nodes, and remove links by clicking
  an edge in the custom topology editor.
- Designated SOURCE (orange hexagon), ordinary sensor nodes (blue circles),
  and SINK (purple diamond).
- Per-node generation-rate controls and communication-range filtering.
- LP versus shortest-path comparison using the same radio-energy model.
- Average-flow battery depletion playback with Play/Pause/Reset and speed
  options up to 100x.
- Per-node TX/RX energy, bottleneck and battery shadow-price analysis.
- Complementary-slackness and LP primal-dual diagnostics.
- Multi-seed, network-density and battery-scale experiments.
- Packet-level (discrete-time) simulation with packet generation, queues,
  hop-by-hop store-and-forward forwarding, probabilistic link delivery,
  bounded retransmissions, throughput and packet delivery/drop counts.
- Idle-listening and duty-cycled sleep energy accounting in packet simulation.
- Dedicated single-link failure experiment: compare the intact network with
  the same network after removing a selected physical connection, with routes
  recomputed around the failure when a path remains.
- CSV exports for packet/failure summaries, node energy breakdown and link
  delivery statistics, in addition to the existing LP exports.

Packet simulator assumptions and limitations
---------------------------------------------
The packet simulator is a discrete-time educational abstraction, separate
from the LP average-flow solver. Each awake node can process at most one
queued packet per time step. A packet follows a minimum-transmission-energy
path; after a link failure, paths are recomputed. Each attempt succeeds with
a configurable probability. A failed attempt is retried up to the configured
maximum number of additional retransmissions; otherwise the packet is dropped.
The selected awake fraction represents a deterministic awake window in each
one-second cycle. Idle listening energy is charged during the awake fraction,
and sleep energy during the remainder. TX and RX energy are charged per
packet attempt/success respectively. Packets still queued at the end of the
run are reported separately from dropped packets.

This is not a full IEEE 802.15.4/MAC or physical-layer simulator. It does not
model collisions, interference, fading, carrier sensing, channel contention,
CSMA/CA backoff, acknowledgements as separate frames, packet fragmentation,
realistic radio-state transition costs, or hardware-specific timing. The
link-success probability is a user-controlled abstraction, not a calculated
radio link budget. The one-packet-per-node-per-step service rule is a simplified
scheduler. Treat results as comparative educational experiments, not hardware
predictions.

The existing average-flow animation remains an LP-derived average-power
model; it is not replaced by the packet simulator. The shortest-path baseline
in the LP comparison sends each generating node's own traffic along its
minimum-transmission-energy path; the LP may split flows across paths.

Playback speed
--------------
The average-flow animation refreshes at a target 100 ms interval. At 100x, it
advances by up to 10 simulated seconds per refresh. Browser, network and
Streamlit rendering overhead may make the achieved speed lower than the
selected target.

Validation
----------
The Python scripts pass syntax compilation. The LP solver and packet
simulator were run on sample networks; a selected-link removal changed packet
outcomes on tested route-critical links. A live Streamlit UI launch could not
be performed in the build environment because Streamlit is not installed
there. Install requirements.txt and launch locally before a live demonstration.


UI theme
-------
This package uses the Industrial Graphite dark theme with amber accents. The Streamlit theme is configured in .streamlit/config.toml; dashboard CSS, Plotly chart templates and the embedded topology editor styling are aligned to the same palette. Simulation and optimization logic are unchanged.
