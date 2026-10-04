# Fluid dynamics

This example runs coupled D2Q9 fluid and D2Q5 passive-smoke lattice Boltzmann
steps as one ZGC graph and renders the result with Raylib. The application feeds
both recurrent population fields back into the model each step. The model also
outputs smoke concentration, density, and both velocity components for rendering.

Run from this directory:

```sh
zig build run -Doptimize=ReleaseFast
```

The 320 × 180 initial field contains two opposing vortices. Its outer edges are
closed no-slip walls: populations that reach a wall bounce into their opposite
D2Q9 direction instead of crossing or wrapping around the domain.

- `Space`: pause or resume
- `N`: advance one step and pause
- `R`: restore the initial vortices
- `P`: switch between advected smoke and diagnostic data views
- `V`: switch between speed and density coloring
- `A`: show or hide velocity arrows sampled across the field
- `[` / `]`: decrease or increase the BGK relaxation value, `omega`
- Left mouse drag: stir fluid in the direction of the drag

Mouse motion supplies a soft, localized force field to the graph for the next
solver step. The force is clipped naturally by the closed domain and creates no
additional solid or barrier cells.

The speed view maps `sqrt(ux² + uy²)` to color. The density view shows deviations
around the equilibrium density of one. Both fields are read from model outputs;
rendering does not alter the solver graph.

The smoke view renders the model's concentration output directly. A soft source
inside the left vortex continuously injects smoke into the D2Q5 state; the model
advects and diffuses it with the fluid velocity and applies a small decay. This
replaces the old renderer-owned particle approximation.

`omega` is copied into a model input each step, so changes apply without rebuilding
the graph. The controls keep it between 0.6 and 1.7. Velocity arrows use the same
output fields as the color view; arrow direction shows flow direction and arrow
length shows relative speed.
