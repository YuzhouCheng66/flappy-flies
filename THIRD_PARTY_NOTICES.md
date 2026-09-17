# Data and implementation credits

## MaleCNS v1.0

Connectome and soma-coordinate data: FlyEM / HHMI Janelia Research Campus,
University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research.
[Official dataset](https://male-cns.janelia.org/download/),
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

Changes: retain `status == Traced` neurons; aggregate directed synapse counts
into 25,563,197 neuron-pair edges; row-normalize the frozen rate-model graph;
encode losslessly as delta/gzip chunks; subsample measured somata and actual
edges for display. The 165,122 retained neurons are not the headline census.
The graph's source and per-block hashes are in `graph/manifest.json`.
Node activity is synthetic model state, not measured biological recordings.
No endorsement by the data creators is implied.

## Flyhard

The original frozen-reservoir pipeline was adapted from
[Flyhard](https://github.com/MarkUnthank/flyhard), audited revision
`328906f4a0e62c8f9fc18805cf6edae6989b82a5`. Browser WGSL kernels are a new
implementation of the same row-normalized leaky-tanh recurrence.

MIT License

Copyright (c) 2026 Mark Unthank

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

These third-party terms do not relicense unrelated project code or weights.
