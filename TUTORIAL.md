# TUTORIAL.md

It is assumed you have gone through the `README.md` setup instructions before following these steps:

### GPMPC

GPMPC is currently only supported within the venv (not Docker).

```bash
# from <smallsat-sim-dir>
poetry shell
python3 experiments/astrobee_CL.py --headless --log
```

Plotting:

```bash
./smallsat_sim/misc/trajectory.py
```

### Debugging

You might manually need to install the following:

- t_renderer:
  - https://discourse.acados.org/t/problems-with-t-renderer/438/2

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<smallsatsim>/deps/acados/acados/lib/`