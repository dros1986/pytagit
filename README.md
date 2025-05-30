### PyTagit

**PyTagit** is a human-in-the-loop tool for large-scale image classification.  
Install and launch it with:

```bash
# install
pip install pytagit

# run the program
pytagit
```

If you use PyTagit, please cite us:

```
# citation
```

---

### Features

At startup, all images are unclassified. You can assign them via drag-and-drop:

![Main window](https://raw.githubusercontent.com/dros1986/pytagit/refs/heads/main/res/main_window.png?token=GHSAT0AAAAAAC4XJIVUPCZIQQSYBBWK2BPS2BZUXMA)

Start by assigning a few examples per class. Then, apply:

- **Random Forest** or **k-NN** to classify the rest.
- Visit each class and click to mark correct predictions. Once clicked, the border will become red.
- Repeat the process to reclassify using the verified samples.

For accelerated labeling, use:

#### Interactive t-SNE

Draw a decision boundary directly on a 2D feature map to assign multiple samples:

![t-SNE](https://raw.githubusercontent.com/dros1986/pytagit/refs/heads/main/res/interactive_tsne.png?token=GHSAT0AAAAAAC4XJIVUAQQF7YBEKZGQBO522BZUYVQ)

#### Out-of-Distribution Detection

Useful for quality control scenarios: find samples close to a class using feature-based OOD:

![OOD](https://raw.githubusercontent.com/dros1986/pytagit/refs/heads/main/res/visual_ood.png?token=GHSAT0AAAAAAC4XJIVVS7QISXNSAZEEL2M22BZUZEQ)

To classify all samples, use Random Forest with a confidence threshold of 0.
