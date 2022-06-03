# Tumor_Classification

Multiclass classification for brain tumors!

**Data**

The data used for this mini project was taken from the Kaggle Brain Tumor dataset, which can be found here https://www.kaggle.com/datasets/denizkavi1/brain-tumor.

This dataset contains images for 3 classes of brain tumor:

* meningioma(1)
 * ![423](https://user-images.githubusercontent.com/60507154/171852556-3af61227-d0c3-46f9-8a68-19f718ad3a24.png)

* glioma(2)
 * ![2](https://user-images.githubusercontent.com/60507154/171852714-40e30fb9-00bf-4508-b52e-60be0decc2ac.png)
 
* pituitary tumor(3)
 * ![922](https://user-images.githubusercontent.com/60507154/171852767-2f3eb37d-0b11-43e3-852c-b759e33cf44c.png)

The original dataset was shared by Jun Cheng: Cheng, Jun (2017): brain tumor dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.1512427.v5

To have the data organized exactly as this project did:
1. Download the data
2. Extract the data in a "data/" folder

Notes:
* The model was trained in Google Colab using the GPU available.
  * the weights were used in vscode to make the predictions, hence the absence of the training iterations.
* All used functions in the "Training___Evaluation" notebook are defined in the "utils.py" script

