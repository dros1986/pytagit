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

![Main window](https://github.com/dros1986/pytagit/blob/2e73f12c4028bac1bd9827d2a261a528c3b3ea55/res/main_window.png)

Start by assigning a few examples per class. Then, apply:

- **Random Forest** or **k-NN** to classify the rest.
- Visit each class and click to mark correct predictions. Once clicked, the border will become red.
- Repeat the process to reclassify using the verified samples.

For accelerated labeling, use:

#### Interactive t-SNE

Draw a decision boundary directly on a 2D feature map to assign multiple samples:

![t-SNE](https://github.com/dros1986/pytagit/blob/247fdce9967567caa6b15f5fe97c88c9e9848dab/res/interactive_tsne.png)

#### Out-of-Distribution Detection

Useful for quality control scenarios: find samples close to a class using feature-based OOD:

![OOD](https://github.com/dros1986/pytagit/blob/247fdce9967567caa6b15f5fe97c88c9e9848dab/res/visual_ood.png)

To classify all samples, use Random Forest with a confidence threshold of 0.
