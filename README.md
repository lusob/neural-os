# NeuralOS

**moving windows without programming an event system — just a neural network guessing pixels from mouse actions**

[**Live demo →**](https://lusob.github.io/neural-os/) &nbsp;|&nbsp; [**Colab →**](https://colab.research.google.com/drive/1wrc67GjQErvWOsWSwoWndiKf-Wx-EZpz) &nbsp;|&nbsp; [**Neural Computers paper (Meta, 2026) →**](https://arxiv.org/abs/2604.06425)

![demo](neural_os.gif)

## How it works

The demo runs two tiny neural networks in your browser via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). No server, no physics engine, no event system.

### The model: SplitGenie

A two-headed MLP (~39KB) with separate hemispheres for moving and resizing:

```
Input: [dist_to_header_x, dist_to_header_y, dist_to_grip_x, dist_to_grip_y, click]  →  5 floats

  MOVE hemisphere:   Linear(3→64) → ReLU → Linear(64→64) → ReLU → Linear(64→2) → Tanh
                     input: [dist_header_x, dist_header_y, click]
                     output: [vel_x, vel_y]

  RESIZE hemisphere: Linear(3→64) → ReLU → Linear(64→64) → ReLU → Linear(64→2) → Tanh
                     input: [dist_grip_x, dist_grip_y, click]
                     output: [delta_w, delta_h]
```

The two heads share nothing except the click signal, so the model can't confuse dragging with resizing.

### Training data

40,000 synthetic examples generated analytically:
- Move fires when cursor is within `HEADER_TOLERANCE=0.25` of the titlebar center
- Resize fires when cursor is within `GRIP_TOLERANCE=0.15` of the corner grip
- Resize takes priority over move when both zones overlap
- Loss: MSE, trained with Adam for 10 epochs

No real interaction recorded — the behavior is learned purely from the geometry of the zones.

### What the side panel shows

- **DIST HEADER / DIST GRIP**: radar showing cursor distance to each interaction zone
- **Neural Activity**: activations of the last hidden layer of each hemisphere (green = move, orange = resize)
- **Motor Output**: raw network output — velocity and resize deltas before being applied to window state

### The interesting part

There's no `if/else` for "are we dragging or resizing". The network learned the decision boundary from examples. You can feel it near the edges — when the cursor is between the titlebar and the grip corner the network's uncertainty is visible as the window hesitates between modes.

## Run it yourself

Open the [Colab notebook](https://colab.research.google.com/drive/1wrc67GjQErvWOsWSwoWndiKf-Wx-EZpz) to retrain the model from scratch. Takes ~2 minutes on a free GPU.

## Related

Meta AI published [Neural Computers](https://arxiv.org/abs/2604.06425) (Zhuge et al., 2026) — the same idea scaled up: a video model that predicts full screen frames conditioned on pixels + instructions + user actions, for both CLI and GUI. Their open problems ("challenges remain with routine reuse, controlled updates, and symbolic stability") are the same walls this experiment hits.
