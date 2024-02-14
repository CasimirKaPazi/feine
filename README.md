# Free Energy In Neuronal Evolution - FEINE Simulation

This program combines neural networks with evolution to explore ideas of self-organization and emergent stability.
You are presented with a 2D grid of cells, where each cell is a neuron connected to it's neighbors. The neurons try to predict their future activation level (minimize surprise). Patterns emerge and those that are stable over time proliferate.

## Purpose and function

The inspiration to this program came from my attempt trying to understand Karl Friston's Free Energy principle. Insofar it's not only useful to play with neural networks and evolution but to look at the question: How do things (life) persist?

Learning:

Every cell has a list of weights associated with it's neighbors. Neuron activations are calculated as weighted sum of the neighbor cells activation. Negative activation is set to zero (fancy term: ReLU activation function).

The cells activation is compared to it's previous activation.

Updates neuron weights to minimize surprise, incorporating a quadratic loss function.

Evolution:

When a neurons activation drops to low it receive new genes and weights from a random neighbor.
Every time step, a few neurons have their genes and weights mutated.

## Usage
Requires the LÖVE2D engine. Run with:

        love ./feine

**spacebar** - Pause simulation.
**a** - Advance paused simulation by one step.
**left mouse click** - Write information about the pointed cell into the terminal (activation, genes, weights).
**shift + left mouse click** - Copy the pointed cell onto the whole grid.
**right mouse click** - Randomize the pointed cell.
**arrow keys** - Shift the grid.

View modes:
**0** - Hide grid to run a little bit faster. Also shows current generation.
**1** - Default view. Shows activation with slight indication of genes.
**2** - Shows error.
**3** - Shows learning rate.
**4** - Shows gene 1.
**5** - Shows gene 2.
**6** - Shows all genes.

## License
Affero GPL
