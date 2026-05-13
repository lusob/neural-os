# NeuralOS

**moving windows without programming an event system, just a neural network guessing pixels from mouse actions**

[live demo](https://lusob.github.io/neural-os/) · [colab](https://colab.research.google.com/drive/1wrc67GjQErvWOsWSwoWndiKf-Wx-EZpz) · [Neural Computers paper (Meta, 2026)](https://arxiv.org/abs/2604.06425)

![demo](neural_os.gif)

## two approaches

this repo has two experiments that tackle the same problem differently.

### approach 1: pixel prediction (colab)

the first approach is the more radical one. a U-Net takes the last 2 frames of the screen as input (one-hot encoded by color class) plus a mouse vector (dx, dy, click), and predicts the next frame pixel by pixel. no renderer, no window state, no coordinates anywhere. the network is the window manager.

```
stacked frames [B, 8, 128, 128] + motion [B, 3]  ->  next frame logits [B, 4, 128, 128]
```

it works, the window moves and the cursor follows. but it drifts over time because theres no explicit position stored anywhere, the model infers where the window is from what it sees. open the [colab](https://colab.research.google.com/drive/1wrc67GjQErvWOsWSwoWndiKf-Wx-EZpz) to train it from scratch and see the autoregressive gif it generates.

### approach 2: learned behavior + js renderer (live demo)

the second approach is a middle ground. the renderer is still deterministic (rectangles drawn in canvas), but the *behavior* of the window is learned. a two-headed MLP called SplitGenie takes distances from the cursor to the titlebar and the resize grip, and outputs velocity and resize deltas. the js renderer applies those deltas every frame.

this is what runs in the [live demo](https://lusob.github.io/neural-os/). the model is 39KB and loads instantly.

```
input: [dist_to_header_x, dist_to_header_y, dist_to_grip_x, dist_to_grip_y, click]  ->  5 floats

  MOVE hemisphere:   Linear(3->64) -> ReLU -> Linear(64->64) -> ReLU -> Linear(64->2) -> Tanh
                     input: [dist_header_x, dist_header_y, click]
                     output: [vel_x, vel_y]

  RESIZE hemisphere: Linear(3->64) -> ReLU -> Linear(64->64) -> ReLU -> Linear(64->2) -> Tanh
                     input: [dist_grip_x, dist_grip_y, click]
                     output: [delta_w, delta_h]
```

the two heads share nothing except the click signal, so the model cant confuse dragging with resizing. theres no if/else for that anywhere, the network learned the decision boundary from 40k synthetic examples.

### what the side panel shows (live demo)

- **DIST HEADER / DIST GRIP**: radar showing cursor distance to each interaction zone
- **Neural Activity**: activations of the last hidden layer of each hemisphere (green = move, orange = resize)
- **Motor Output**: raw network output, velocity and resize deltas before being applied to window state

### the interesting part

you can feel the network's learned space when you interact with it. drag near the titlebar and it moves, get close to the corner and it switches to resize. sometimes it gets confused near the edges, which is honestly more interesting than if it just worked perfectly, you can sense the probability mass shifting.

## run it yourself

open the [colab notebook](https://colab.research.google.com/drive/1wrc67GjQErvWOsWSwoWndiKf-Wx-EZpz) to retrain both models from scratch. takes ~2 minutes on a free GPU.

## related

Meta AI published [Neural Computers](https://arxiv.org/abs/2604.06425) (Zhuge et al., 2026), same idea scaled up: a video model that predicts full screen frames conditioned on pixels + instructions + user actions, for both CLI and GUI. their open problems ("challenges remain with routine reuse, controlled updates, and symbolic stability") are the same walls the pixel approach hits.
