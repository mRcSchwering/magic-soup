# Carbon Fixation

Here, I am trying to bring the cells to fix CO2.
Reactions and molecules of the Wood Ljungdahl pathway are defined.
During the simulation CO2 levels will constantly be kept high on the molecule map.
Energy is constantly added by interchanging ADP with ATP and NADP with NADPH.
Up to a total of 1000 cells, cells spawn randomly.

With this chemistry alone cells can't produce other molecules, so molecule degradation will be switched off.
All cells can do is:

$$
\begin{align*}
CO2 + NADPH & \rightleftharpoons formiat + NADP \\
formiat + FH4 + ATP & \rightleftharpoons formylFH4 + ADP \\
formylFH4 + NADPH & \rightleftharpoons methylenFH4 + NADP \\
methylenFH4 + NADPH & \rightleftharpoons methylFH4 + NADP \\
methylFH4 + NiACS & \rightleftharpoons FH4 + methylNiACS \\
methylNiACS + CO2 + HSCoA & \rightleftharpoons NiACS + acetylCoA
\end{align*}
$$

Evolutionary pressure is applied by the selection of cells which are allowed to divide and the selection of cells which die.
The likelihood of a cell dying is increased for cells with low ATP and NADPH contents.
Furthermore, cells which live long but don't replicate have an increased likelihood of dying.
Cell division is allowed if acetyl-CoA concentrations are high enough.
The cell division will convert acetyl-CoA back to SH-CoA.
Cells with increased acetyl-CoA have an increased likelihood of dividing.

```
python -m experiments.e1_co2_fixing.main --n_steps=10
...
tensorboard --logdir=./experiments/e1_co2_fixing/runs
```
