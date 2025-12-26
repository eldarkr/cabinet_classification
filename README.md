# Cabinet Classification

Briefly, a cabinet-type classifier using ResNet18 with some simple techniques for class imbalance handling.
### Category Distribution
| Category         | Percentage |
|------------------|------------|
| `lc:bcabo`       | 47.8%      |
| `lc:wcabo`       | 21.1%      |
| `lc:muscabinso`  | 19.8%      |
| `lc:wcabcub`     | 9.5%       |
| `lc:bcabocub`    | 1.7%       |

## Project Structure
- `artifacts/models/` — Saved models and training artifacts
- `configs/config.yaml` — Main configuration file
- `dataset/` — Data and annotations
- `notebooks/` — Jupyter notebooks for data exploration and training
- `scripts/` — Python scripts for training, inference, and data preparation
- `src/` — Source code (data loaders, models, trainer, utils)

## How to setup and run

### Clone repository:
   ```bash
   git clone 
   cd classification
   ```

### Make sure that you have installed and unzipped dataset in `dataset/annotated_pdfs_and_data`!
You should have structure like that:
```
...
configs/
dataset/annotated_pdfs_and_data/
├── simple_annotation_statistics.csv
├── simple_categories.json
├── Cullman High School/
├── Grand Ridge Pre-K-8 Phase 2/
├── ID 220/
├── ID 221/
├── ...
```

### How to run:
1. **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare data:**
   ```bash
   python -m scripts.prepare_data
   ```
4. **Train the model:**
   ```bash
   python -m scripts.train
   ```
5. **Run inference:**
   ```bash
   python -m scripts.infer
   ```

### Experiments
You can experiment with different hyperparameters and configuration options using OmegaConf. 

To override any parameter from the config file, simply pass it as a CLI argument. For example:
  ```bash
  python -m scripts.train training.fine_tune.strategy=frozen model.optimizer.lr=0.0001 training.num_epochs=20

  # Infer
  python -m scripts.infer inference.image_path=dataset/examples
  ```

You can override any nested parameter in the config this way. This makes it easy to run multiple experiments with different settings without editing the config file each time.


## Results
*You can see results in [notebooks/result_analysis.ipynb](https://github.com/eldarkr/cabinet_classification/blob/main/notebooks/result_analysis.ipynb).*

**Per-class metrics:**

| Category         | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| `lc:bcabo`         | 1.00      | 1.00   | 1.00     | 430     |
| `lc:bcabocub`      | 1.00      | 1.00   | 1.00     | 5       |
| `lc:wcabo`         | 0.99      | 1.00   | 0.99     | 81      |
| `lc:wcabcub`       | 1.00      | 1.00   | 1.00     | 27      |
| `lc:muscabinso`    | 1.00      | 0.99   | 0.99     | 90      |


**Confusion Matrix:**  
<img width="520" height="498" alt="image" src="https://github.com/user-attachments/assets/91aa3df1-9730-4e22-bf4e-a1f255cd5a15" />

The `ResNet18` shows excellent accuracy and performance across all categories on the validation set. As we have a low number of samples in two categories, I would not trust these results completely.

## Approach

- Most of the time was spent organizing the project architecture. My main goal was to implement a structure that allows for simple experiment management. I considered using Hydra, but ultimately decided to use OmegaConf for simplicity. To improve my understanding of project structuring, I reviewed several GitHub repositories (such as Ultralytics) for inspiration.
- I decided not to use Docker for simplicity and to make GPU usage easier, especially with MPS (Apple Silicon).
- To handle class imbalance, I tested two simple strategies:
   - Adding class weights to the loss function
   - Using a Weighted Random Sampler — this worked better for me
   - Combining both is not recommended, as it leads to double compensation.
   - When using the Weighted Random Sampler, it is better to apply gradient clipping for stability (to avoid spikes).
- I tried using data transformations but did not observe significant improvements. This is probably due to the dataset's geometry; most basic transformations are not useful and do not provide the model with additional geometric understanding.

## Future Enhancements

- Deep dive into data
   - Transition from rasterized PNGs to direct extraction from PDFs. This will preserve line sharpness and eliminate compression artifacts critical for geometric analysis.
   - Noise reduction via morphological operations (OpenCV)
   - Research synthetic data generation methods for advanced oversampling.
   - Analyse class separability for better understanding how much classes overlap. For example, analyse latent space using embeddings and clustering algorithms or to use tool like Grad-CAM
- Use cross-validation for more robust assesment of generalization.
- Try to find models that have better understanding of shape. CNN based models like ResNet have texture bias.
- Implement factory pattern for better experiments managment experience.
- Use some MLOps tools for logging and etc.
