# Final Experiment Design

## 1. Datasets

*   **UCR Time Series Classification Archive:**
    *   **Description:** A widely recognized repository containing a diverse collection of univariate and multivariate time series datasets, primarily used for classification tasks. These datasets span various domains such as medical (e.g., ECG, EEG), sensor readings (e.g., HAR, Wafer), and more, offering different lengths and complexities.
    *   **Link:** [https://www.cs.ucr.edu/~eamonn/time_series_data_2018/](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
    *   **Citation:** Not directly cited in the provided literature, but referenced as a source for "Benchmark Time Series Datasets" in the experiment plan.
*   **UEA & UCR Time Series Classification Repository:**
    *   **Description:** An expanded and updated version of the UCR archive, providing additional and more recent time series datasets, including a significant number of multivariate datasets suitable for both classification and potentially forecasting tasks.
    *   **Link:** [https://timeseriesclassification.com/](https://timeseriesclassification.com/)
    *   **Citation:** Not directly cited in the provided literature, but referenced as a source for "Benchmark Time Series Datasets" in the experiment plan.

**Justification:** The experiment plan explicitly calls for "Benchmark Time Series Datasets (e.g., from UCR Archive, UEA Archive)" to ensure comprehensive evaluation across diverse time series characteristics. While the provided literature [2] mentions "five real-world datasets" and [3] mentions "several typical time series datasets," neither provides specific names or direct links to these datasets. Therefore, the UCR and UEA archives are selected as verifiable, official sources from which specific datasets (e.g., ECG, EEG, HAR, Wafer for classification; Traffic, Electricity Consumption, Weather, Stock prices for forecasting, as mentioned in the experiment plan) would be chosen to match the research goals and allow for both quantitative and qualitative assessment by domain experts.

## 2. Tools & Instruments

*   **Deep Learning Frameworks:** Python-based libraries for building, training, and evaluating deep learning models.
    *   **Examples:** TensorFlow, PyTorch.
*   **Deep Learning Models for Time Series:** Implementations of various deep learning architectures tailored for time series data.
    *   **Examples:** Convolutional Neural Networks (CNNs) (e.g., FCN, ResNet for time series), Recurrent Neural Networks (RNNs) (e.g., LSTMs, GRUs), Transformer-based models (e.g., Self-Attention Encoder).
*   **Saliency/Explainability Methods:** Implementations of diverse saliency techniques, including gradient-based, perturbation-based, and time series-specific approaches.
    *   **Examples:**
        *   *Gradient-based:* Integrated Gradients, Grad-CAM (adapted for time series), DEMUX [1].
        *   *Perturbation-based:* LIME (adapted for time series), SHAP (adapted for time series), Series Saliency [3].
        *   *Baseline/Control:* Random attribution.
*   **Data Preprocessing and Analysis Libraries:** Python libraries for data manipulation, numerical operations, and statistical analysis.
    *   **Examples:** Pandas, NumPy, SciPy, Scikit-learn.
*   **Visualization Libraries:** Libraries for generating plots, heatmaps, and interactive visualizations of time series data and saliency maps.
    *   **Examples:** Matplotlib, Seaborn, Plotly.
*   **Computational Resources:** High-performance computing resources, specifically Graphics Processing Units (GPUs), for efficient training of deep learning models and the computationally intensive generation of saliency maps.
*   **Human Experts/Domain Specialists:** Individuals with specialized knowledge in the domains of the chosen time series datasets (e.g., medical doctors for ECG data, financial analysts for stock data) to conduct qualitative assessments of saliency explanations.

## 3. Variables to Measure

**A. Independent Variables:**

*   **Deep Learning Model Architecture:** The specific type of deep learning model employed (e.g., CNN, LSTM, Transformer). This variable will be systematically varied to assess how saliency methods perform across different model complexities and inductive biases.
*   **Saliency Explanation Method:** The particular saliency technique applied (e.g., Integrated Gradients, LIME, DEMUX [1], Series Saliency [3], Random attribution). This is the primary variable of interest, as its impact on explanation quality will be evaluated.
*   **Time Series Dataset:** The specific benchmark dataset used for training and evaluation. Different datasets introduce varying characteristics (e.g., univariate/multivariate, length, noise levels, domain) that may influence saliency method performance.
*   **Reference Input for Perturbation Methods (for Series Saliency and similar):** The method used to obtain the reference "series image" `^X` (e.g., constant, Gaussian noise `ε_1 ~ N(μ, σ_1^2)`, or Gaussian blur `g_σ_2^2`) [3]. This parameter affects how perturbations are introduced and thus the resulting saliency.
*   **Regularization Parameters (for Series Saliency):** `λ_1` (mask size penalty) and `λ_2` (mask smoothness penalty) [3]. These parameters control the characteristics of the generated saliency mask, influencing its sparsity and coherence.
*   **Window Size (`w`) and Forecasting Horizon (`τ`) (if applicable):** Parameters defining the input window length for models and the prediction horizon for forecasting tasks [2, 3]. These affect the scope of the explanation.

**B. Dependent Variables:**

*   **Model Performance:**
    *   **Classification Tasks:** Accuracy, F1-score, Area Under the Receiver Operating Characteristic Curve (AUC). These metrics quantify the predictive capability of the deep learning models.
    *   **Forecasting Tasks:** Mean Squared Error (MSE), Mean Absolute Error (MAE) [3]. These metrics quantify the accuracy of time series forecasts.
*   **Saliency Map Characteristics (Quantitative Metrics):**
    *   **Fidelity/Faithfulness:** How accurately the saliency map reflects the model's actual decision-making process.
    *   **Consistency:** The stability and reproducibility of saliency explanations across adjacent or overlapping time windows [2].
    *   **Robustness:** The stability of saliency explanations when subjected to small, irrelevant perturbations or feature swaps in the input data [2].
    *   **Sparsity/Complexity:** The degree to which the saliency map highlights only a minimal, yet sufficient, number of influential temporal segments or features.
    *   **Smoothness:** The temporal and feature-wise coherence and continuity of saliency scores, making the explanations visually interpretable.
*   **Human Understandability and Usefulness (Qualitative Assessment):**
    *   **Clarity/Interpretability:** How easily a human expert can comprehend the explanation provided by the saliency visualization.
    *   **Alignment with Domain Knowledge:** The extent to which the highlighted temporal segments or features correspond to known domain-specific patterns or causal factors.
    *   **Trustworthiness:** The degree to which the explanation increases an expert's confidence and trust in the model's prediction.
    *   **Actionability:** The potential for the explanation to help identify model flaws, suggest improvements to the data, or lead to new domain insights.

## 4. Experimental Procedures (step-by-step)

### Phase 1: Setup and Data Preparation

1.  **Step 1.1: Dataset Selection and Acquisition.**
    *   Select 3-5 diverse benchmark time series datasets from the UCR/UEA archives or other reputable sources. Prioritize datasets with varying characteristics (e.g., univariate/multivariate, different lengths, noise levels) and from different application domains (e.g., medical, sensor, financial).
    *   Ensure that for each selected dataset, deep learning models are known to achieve competitive performance, as explanations for poorly performing models are less meaningful.
    *   Acquire the raw data files for all selected datasets.
2.  **Step 1.2: Data Preprocessing.**
    *   For each selected dataset:
        *   **Normalization/Scaling:** Apply a consistent scaling method (e.g., Z-score standardization or Min-Max scaling to [0, 1]) to all time series features. This ensures numerical stability during model training and comparability across different datasets and models.
        *   **Windowing (if applicable):** For models requiring fixed-length inputs (e.g., CNNs, Transformers) or for evaluating consistency across time windows [2], segment the time series into overlapping or non-overlapping windows of a predefined length `w`. The choice of `w` will be consistent for a given dataset and task.
        *   **Train-Validation-Test Split:** Divide each dataset into training, validation, and test sets (e.g., 70% training, 15% validation, 15% test). For forecasting tasks, ensure that the temporal order is strictly maintained during splitting to prevent data leakage.
[CODE_NEEDED: Load and preprocess time series datasets, including normalization, windowing (if applicable), and train-validation-test splitting.]

### Phase 2: Deep Learning Model Training

1.  **Step 2.1: Model Architecture Selection.**
    *   Select 2-3 diverse deep learning model architectures commonly used for time series tasks. This diversity ensures that saliency methods are evaluated across different underlying model complexities and mechanisms.
    *   **Examples:** A Convolutional Neural Network (CNN) (e.g., a Time-CNN or ResNet variant), a Recurrent Neural Network (RNN) (e.g., LSTM or GRU), and a Transformer-based model (e.g., a Self-Attention Encoder).
2.  **Step 2.2: Model Training and Hyperparameter Tuning.**
    *   For each selected dataset and model architecture combination:
        *   Initialize model weights using standard practices (e.g., Xavier or Kaiming initialization).
        *   Train the model on the designated training set. Use the validation set for hyperparameter tuning (e.g., learning rate, batch size, number of layers, hidden units, dropout rates) and for early stopping to prevent overfitting.
        *   Employ consistent optimizers (e.g., Adam, RMSprop) and loss functions appropriate for the task (e.g., Categorical Cross-Entropy for classification, Mean Squared Error for forecasting).
        *   Monitor model performance on the validation set. Only models achieving competitive performance (e.g., >80% accuracy for classification, low MSE for forecasting relative to baselines) will be used for saliency evaluation, ensuring that explanations are generated for capable models.
        *   Save the trained model weights and configuration for later use.
[CODE_NEEDED: Implement and train selected deep learning models (e.g., CNN, LSTM, Transformer) for time series classification or forecasting tasks, including hyperparameter tuning and saving model states.]

### Phase 3: Saliency Map Generation

1.  **Step 3.1: Saliency Method Selection.**
    *   Select 4-5 diverse saliency/explainability methods. This selection should include a mix of gradient-based and perturbation-based approaches, as well as at least one method specifically designed or adapted for time series data.
    *   **Examples:** Integrated Gradients, Grad-CAM (adapted for time series), DEMUX [1], LIME (adapted for time series), SHAP (adapted for time series), Series Saliency [3].
    *   **Control Baseline:** Include a "Random Attribution" method where importance scores are assigned randomly, to serve as a lower bound for quantitative metrics.
2.  **Step 3.2: Saliency Map Computation.**
    *   For each trained deep learning model and selected saliency method:
        *   Select a representative subset of test samples (e.g., 100-500 samples per class for classification, or a diverse range of forecasting scenarios covering different prediction outcomes). Include both correctly and incorrectly predicted samples to analyze explanations for various model behaviors.
        *   Generate saliency maps for each selected sample. A saliency map `S` will typically be a matrix of dimensions `N` (number of features) by `d` (window length), where `S_n,t` represents the importance score of feature `n` at time step `t`.
        *   **For perturbation-based methods (e.g., LIME, SHAP, Series Saliency [3]):**
            *   Define consistent perturbation strategies. For Series Saliency [3], specify the method for obtaining the reference "series image" `^X` (e.g., constant, Gaussian noise `ε_1 ~ N(μ, σ_1^2)`, or Gaussian blur `g_σ_2^2`).
            *   Set regularization parameters (`λ_1` for mask size penalty and `λ_2` for mask smoothness penalty [3]) consistently across methods or tune them to achieve visually interpretable and sparse masks.
        *   **For consistency evaluation [2]:** Generate saliency maps for multiple overlapping time windows (`w_d_s` and `w_d_s_bar`) for a given time series sample, as defined in [2], to allow for comparison of importance scores in shared temporal segments.
        *   Store all generated saliency maps, along with the corresponding input data, model predictions, and ground truth labels.
[CODE_NEEDED: Implement and apply selected saliency methods (e.g., Integrated Gradients, LIME, DEMUX, Series Saliency, Random Attribution) to generate saliency maps for a representative set of model predictions on the test data.]

### Phase 4: Quantitative Evaluation of Saliency Maps

1.  **Step 4.1: Model Performance Measurement.**
    *   Evaluate the final performance of each trained deep learning model on the test set using the chosen metrics (Accuracy, F1-score, AUC for classification; MSE, MAE for forecasting). Record these scores to contextualize the saliency evaluations.
2.  **Step 4.2: Fidelity/Faithfulness Evaluation.**
    *   For each saliency method and model combination:
        *   **Deletion/Insertion Curves:** Systematically perturb (delete or insert) features/time steps in the input time series based on their saliency scores. For deletion, remove features from most to least important and observe the drop in model confidence/output. For insertion, add features from most to least important to a baseline and observe the increase in confidence.
        *   **Area Over the Perturbation Curve (AOPC):** Calculate the area under the deletion/insertion curves. A higher AOPC for deletion (faster drop) and a lower AOPC for insertion (slower rise) indicates better fidelity.
        *   **Sensitivity-n:** Measure the change in the model's output when the top-n salient features are perturbed or masked.
        *   Compare these metrics against the random saliency baseline; effective saliency methods should show significantly better fidelity.
3.  **Step 4.3: Consistency Evaluation [2].**
    *   For each saliency method and model combination:
        *   Utilize the saliency maps generated for overlapping time windows (`w_d_s` and `w_d_s_bar`) from Step 3.2.
        *   Calculate consistency metrics for the overlapping segments:
            *   **Correlation Coefficient:** Compute the Pearson or Spearman correlation between the importance scores of the shared temporal segments in `S_ds` and `S_d_s_bar`.
            *   **Divergence Measures:** Employ metrics such as Kullback-Leibler (KL) divergence or Jensen-Shannon (JS) divergence to quantify the difference in the distribution of importance scores within the overlapping regions.
        *   Average these consistency metrics over all evaluated overlapping windows and samples.
4.  **Step 4.4: Robustness Evaluation [2].**
    *   For each saliency method and model combination:
        *   Select a subset of test samples.
        *   **Feature Swaps/Perturbations:** For multivariate time series, introduce small, irrelevant perturbations or perform feature column swaps (e.g., swap two non-critical feature columns) as conceptually described in [2].
        *   Generate saliency maps for both the original and the subtly perturbed/swapped inputs.
        *   Calculate robustness metrics:
            *   **Correlation/Distance:** Measure the correlation (e.g., Pearson) or Euclidean distance between the saliency maps of the original and perturbed inputs. The expectation is that important areas in the saliency map should remain salient in the corresponding (potentially swapped) areas.
            *   **Stability Index:** Quantify the percentage of top-k salient features that remain in the top-k after the perturbation, indicating the stability of the most important attributions.
        *   Average these robustness metrics over multiple perturbations and samples.
5.  **Step 4.5: Sparsity and Smoothness Evaluation [3].**
    *   For each saliency method and model combination:
        *   **Sparsity:** Calculate the L0-norm (number of non-zero elements) or L1-norm of the saliency maps (or the mask `M` for Series Saliency [3]) to quantify the number of highlighted features/time steps. Lower values indicate sparser explanations.
        *   **Smoothness:** Calculate the Total Variation (TV) norm of the saliency map, or the penalty function `r = Σ_{(t,i)}(m_{t,i} - m_{t,i+1})^2 + Σ_{(t,i)}(m_{t,i} - m_{t+1,i})^2` from [3] for mask smoothness. Lower values indicate smoother explanations.
        *   Average these metrics across all generated saliency maps.
[CODE_NEEDED: Implement quantitative evaluation metrics for saliency maps, including fidelity (AOPC, deletion/insertion curves, sensitivity-n), consistency (correlation, divergence for overlapping windows), robustness (correlation/distance after feature swaps/perturbations), sparsity (L0/L1 norm), and smoothness (Total Variation norm or Series Saliency's 'r' penalty).]

### Phase 5: Qualitative Evaluation (Human Expert Assessment)

1.  **Step 5.1: Expert Selection and Briefing.**
    *   Recruit 5-10 domain experts whose expertise aligns with the chosen time series datasets (e.g., cardiologists for ECG data, industrial engineers for sensor data).
    *   Conduct an initial briefing session to explain the research goal, the concept of saliency maps, the deep learning model's task, and the specific evaluation questions. Ensure experts understand the data and the model's predictions.
2.  **Step 5.2: User Study Design and Execution.**
    *   Develop a structured user study interface (e.g., a web application). For each selected test sample:
        *   Present the model's prediction (e.g., classification label and confidence, forecasted values).
        *   Present the raw time series input data.
        *   Present the saliency visualizations generated by different methods (e.g., heatmaps overlaid on the time series, separate feature importance plots). Randomize the order of presentation of saliency methods for each sample to mitigate order bias.
        *   Include a "No Explanation" baseline where only the prediction and raw data are shown, for comparison.
        *   Ask experts to answer a series of structured questions for each explanation:
            *   "Does this explanation help you understand *why* the model made this prediction?" (Likert scale: 1-5, from "Not at all" to "Completely")
            *   "Do the highlighted temporal segments/features align with your domain knowledge?" (Likert scale: 1-5, from "Strongly disagree" to "Strongly agree", or Yes/No/Partially with an open-ended justification field)
            *   "Does this explanation increase your trust in the model's prediction?" (Likert scale: 1-5, from "Decreases trust" to "Greatly increases trust")
            *   "Can this explanation help you identify potential flaws in the model or data, or provide new insights?" (Open-ended text response)
            *   "Which saliency method provides the most useful explanation for this specific prediction?" (Forced choice ranking or preference selection among the presented methods).
3.  **Step 5.3: Data Collection and Analysis.**
    *   Collect all Likert scale ratings, categorical responses, and open-ended feedback from the experts.
    *   Perform statistical analysis on the Likert scale ratings (e.g., calculate means, standard deviations, conduct ANOVA or non-parametric tests to compare methods).
    *   Qualitatively analyze the open-ended feedback by categorizing themes, identifying common strengths and weaknesses, and extracting novel insights provided by different saliency methods.
[CODE_NEEDED: Develop a user interface for presenting time series data, model predictions, and saliency maps to human experts, and for collecting their qualitative feedback via Likert scales and open-ended questions.]

### Phase 6: Analysis and Reporting

1.  **Step 6.1: Integrate Quantitative and Qualitative Results.**
    *   Synthesize the findings from the quantitative metrics (fidelity, consistency, robustness, sparsity, smoothness) with the qualitative expert assessments (understandability, alignment with domain knowledge, trustworthiness, actionability).
2.  **Step 6.2: Comparative Analysis.**
    *   Conduct a comprehensive comparative analysis of the performance of different saliency methods across various deep learning models and datasets. Identify which methods excel under specific conditions, for particular types of time series, or for generating specific types of explanations (e.g., highlighting temporal events vs. feature importance).
3.  **Step 6.3: Hypothesis Testing.**
    *   Statistically test the initial hypothesis: "Saliency-based visualizations can highlight the most influential temporal segments and features, providing accurate explanations of model behavior." This involves comparing the performance of saliency methods against baselines and analyzing the significance of expert feedback.
4.  **Step 6.4: Documentation and Reporting.**
    *   Document all experimental procedures, results, and analyses in detail.
    *   Generate clear and informative visualizations of saliency maps, quantitative metric comparisons (e.g., bar charts, line plots), and summaries of expert feedback.
    *   Prepare a comprehensive research report detailing the findings, discussing their implications for time series model transparency, outlining limitations of the study, and suggesting directions for future research.

## 5. Experimental Conditions and Controls

**A. Experimental Conditions:**

1.  **Model Training Standardization:**
    *   **Procedure:** For each selected deep learning model (CNN, LSTM, Transformer) and dataset, standardize the training process. This includes using consistent optimizers (e.g., Adam with default parameters), learning rate schedules (e.g., cosine annealing or step decay), batch sizes, and a fixed number of epochs or early stopping criteria based on validation performance.
    *   **Justification:** Ensures that differences in saliency method performance are attributable to the methods themselves, rather than variations in model training quality or convergence.
2.  **Saliency Map Generation Consistency:**
    *   **Procedure:** Apply each selected saliency method to the trained deep learning models for a representative and consistent set of predictions from the test set. This set should include predictions across different classes (for classification) or value ranges (for forecasting), and both correct and incorrect predictions.
    *   **Perturbation Strategies:** For perturbation-based methods (e.g., LIME, SHAP, Series Saliency [3]), define and consistently apply specific perturbation strategies and reference inputs (e.g., using Gaussian noise with a fixed standard deviation, a constant baseline, or Gaussian blur for `^X` as in [3]).
    *   **Windowing Parameters:** For methods involving time windows (e.g., for consistency evaluation [2]), define consistent window lengths (`d`) and sliding steps (`s`) across all relevant experiments.
    *   **Justification:** Ensures that saliency maps are generated under comparable conditions, allowing for fair comparison between methods.
3.  **Data Preprocessing Uniformity:**
    *   **Procedure:** Standardize all data preprocessing steps (e.g., normalization, scaling, windowing) across all datasets and models. Ensure consistent handling of multivariate features (e.g., feature-wise scaling).
    *   **Justification:** Eliminates data preprocessing as a confounding variable, ensuring that models and saliency methods operate on consistently prepared inputs.

**B. Controls:**

1.  **Random Saliency Baseline:**
    *   **Procedure:** Generate "saliency maps" by assigning random importance scores to temporal segments and features.
    *   **Justification:** This serves as a crucial lower bound for quantitative fidelity metrics. Any effective saliency method should demonstrate significantly better performance than random attribution, validating its ability to identify truly influential input components.
2.  **No Explanation Baseline (for Human Assessment):**
    *   **Procedure:** In the qualitative expert assessment phase, include scenarios where human experts are presented with only the model's prediction and the raw time series data, without any saliency explanation.
    *   **Justification:** This baseline allows for the direct measurement of the added value of saliency explanations in terms of human understanding, trust, and insight. It helps determine if saliency methods genuinely improve transparency beyond simply seeing the model's output.
3.  **Simpler Model Explanations (Optional Control):**
    *   **Procedure:** (If resources allow) Generate explanations from inherently interpretable simpler models (e.g., ARMA models for forecasting, Decision Trees for classification) trained on the same datasets.
    *   **Justification:** Provides a reference point for the complexity and interpretability of explanations from deep learning models. While the focus is on deep learning, this control can offer insights into the trade-offs between model complexity and explanation clarity.

## 6. Evaluation Metrics and Success Criteria

**A. Quantitative Metrics:**

1.  **Model Performance:**
    *   **Metrics:** Accuracy, F1-score, AUC (for classification); MSE, MAE [3] (for forecasting).
    *   **Success Criterion:** All trained deep learning models must achieve competitive performance (e.g., >80% accuracy for classification, MSE/MAE comparable to state-of-the-art for forecasting) on their respective tasks. This ensures that the explanations are generated for capable and relevant models.
2.  **Fidelity/Faithfulness:**
    *   **Metrics:** Area Over the Perturbation Curve (AOPC), deletion/insertion curves, sensitivity-n. These metrics quantify how well the saliency map reflects the model's actual decision-making by measuring changes in model output when salient features are perturbed.
    *   **Success Criterion:** Saliency methods should consistently achieve significantly higher AOPC (for deletion) or better deletion/insertion curve performance compared to the random saliency baseline. This indicates that they accurately highlight features that are genuinely influential to the model's prediction.
3.  **Consistency [2]:**
    *   **Metrics:** Pearson/Spearman correlation coefficients or Kullback-Leibler/Jensen-Shannon divergence between saliency maps generated for overlapping time windows.
    *   **Success Criterion:** Saliency methods should demonstrate high consistency scores (e.g., correlation coefficients > 0.7 or low divergence values) across overlapping windows. This indicates that the explanations are stable and reliable for similar input segments, a crucial aspect for trustworthiness.
4.  **Robustness [2]:**
    *   **Metrics:** Correlation or distance metrics (e.g., Euclidean distance) between saliency maps of original and subtly perturbed/feature-swapped inputs. A stability index (e.g., percentage of top-k salient features remaining in top-k).
    *   **Success Criterion:** Saliency methods should exhibit high robustness scores (e.g., correlation coefficients > 0.6 or low distance values) against small, irrelevant input perturbations or feature swaps. This ensures that explanations are not easily manipulated or sensitive to minor, non-meaningful input changes.
5.  **Sparsity/Complexity:**
    *   **Metrics:** L0-norm (count of non-zero elements) or L1-norm of the saliency map (or mask `M` from [3]).
    *   **Success Criterion:** Saliency maps should ideally be sparse (e.g., L0-norm indicating only 5-15% of features/time steps are highlighted), meaning they pinpoint only the most critical segments/features without sacrificing fidelity. This enhances interpretability by reducing cognitive load.
6.  **Smoothness:**
    *   **Metrics:** Total Variation (TV) norm of the saliency map, or the penalty function `r` from [3] for mask smoothness.
    *   **Success Criterion:** Saliency maps should exhibit temporal and feature-wise smoothness (e.g., low TV norm or `r` value), meaning importance scores change gradually rather than abruptly. This makes the visualizations easier to interpret and less noisy.

**B. Qualitative Metrics (Expert Assessment):**

1.  **Human Understandability and Usefulness Study:**
    *   **Method:** Structured interviews, surveys, or user studies with domain experts, presenting model predictions and corresponding saliency visualizations.
    *   **Questions to Experts:**
        *   "Does this explanation help you understand *why* the model made this prediction?" (Likert scale)
        *   "Do the highlighted temporal segments/features align with your domain knowledge?" (Likert scale/Yes/No/Partially, with justification)
        *   "Does this explanation increase your trust in the model's prediction?" (Likert scale)
        *   "Can this explanation help you identify potential flaws in the model or data, or provide new insights?" (Open-ended)
        *   "Which saliency method provides the most useful explanation for this task?" (Ranking/Preference)
    *   **Success Criterion:** Saliency methods that consistently receive high ratings (e.g., average Likert scores > 4 out of 5) for clarity, alignment with domain knowledge, trustworthiness, and actionability from human experts will be considered successful. The ultimate success will be the identification of saliency methods that significantly improve human comprehension and trust in deep learning time series models, as evidenced by expert feedback and preference over the "No Explanation" baseline.

## 7. References

[1] Class-Specific Explainability for Deep Time Series Classifiers ([https://arxiv.org/pdf/2210.05411v1](https://arxiv.org/pdf/2210.05411v1))
[2] On the Consistency and Robustness of Saliency Explanations for Time Series Classification ([https://arxiv.org/pdf/2309.01457v1](https://arxiv.org/pdf/2309.01457v1))
[3] Series Saliency: Temporal Interpretation for Multivariate Time Series Forecasting ([https://arxiv.org/pdf/2012.09324v1](https://arxiv.org/pdf/2012.09324v1))

---

### Summary List of [CODE_NEEDED] Tags:

*   **[CODE_NEEDED: Load and preprocess time series datasets, including normalization, windowing (if applicable), and train-validation-test splitting.]**
*   **[CODE_NEEDED: Implement and train selected deep learning models (e.g., CNN, LSTM, Transformer) for time series classification or forecasting tasks, including hyperparameter tuning and saving model states.]**
*   **[CODE_NEEDED: Implement and apply selected saliency methods (e.g., Integrated Gradients, LIME, DEMUX, Series Saliency, Random Attribution) to generate saliency maps for a representative set of model predictions on the test data.]**
*   **[CODE_NEEDED: Implement quantitative evaluation metrics for saliency maps, including fidelity (AOPC, deletion/insertion curves, sensitivity-n), consistency (correlation, divergence for overlapping windows), robustness (correlation/distance after feature swaps/perturbations), sparsity (L0/L1 norm), and smoothness (Total Variation norm or Series Saliency's 'r' penalty).]**
*   **[CODE_NEEDED: Develop a user interface for presenting time series data, model predictions, and saliency maps to human experts, and for collecting their qualitative feedback via Likert scales and open-ended questions.]**

## Generated Code

```python
```python
# --- Load and preprocess time series datasets, including normalization, windowing (if applicable), and train-validation-test splitting. ---
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Configuration
lookback = 10  # Number of past time steps to use as input
forecast_horizon = 1  # Number of future time steps to predict (single step for y)
train_ratio = 0.7
val_ratio = 0.15

# 1. Load data (dummy data for demonstration)
# Replace this with actual data loading (e.g., from CSV, database)
data = np.arange(1000).reshape(-1, 1) + np.random.rand(1000, 1) * 10

# 2. Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Windowing function
def create_sequences(data, lookback, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback + forecast_horizon - 1])
    return np.array(X), np.array(y)

# Apply windowing
X, y = create_sequences(scaled_data, lookback, forecast_horizon)

# 4. Train-validation-test splitting
total_samples = len(X)
train_size = int(total_samples * train_ratio)
val_size = int(total_samples * val_ratio)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size : train_size + val_size], y[train_size : train_size + val_size]
X_test, y_test = X[train_size + val_size :], y[train_size + val_size :]

# --- Implement and train selected deep learning models (e.g., CNN, LSTM, Transformer) for time series classification or forecasting tasks, including hyperparameter tuning and saving model states. ---
# To use this code, run: pip install keras_tuner
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# 1. Generate dummy time series data
sequence_length = 100
num_features = 1
num_samples = 1000
num_classes = 2

X = np.random.rand(num_samples, sequence_length, num_features).astype(np.float32)
y = np.random.randint(0, num_classes, num_samples).astype(np.int32)

# 2. Define a simple CNN model for time series classification
def build_cnn_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(sequence_length, num_features)))
    model.add(layers.Conv1D(
        filters=hp.Int('filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Int('kernel_size', min_value=3, max_value=7, step=2),
        activation='relu'
    ))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
        activation='relu'
    ))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. Hyperparameter tuning using Keras Tuner
tuner = kt.Hyperband(
    build_cnn_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='time_series_cnn'
)

# Split data for tuning
X_train_tune, X_val_tune = X[:800], X[800:]
y_train_tune, y_val_tune = y[:800], y[800:]

tuner.search(X_train_tune, y_train_tune, epochs=5, validation_data=(X_val_tune, y_val_tune))

# Get the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# 4. Train the best model
model.fit(X, y, epochs=10, validation_split=0.2)

# 5. Save the model state
model.save('time_series_cnn_model.h5')

# --- Implement and apply selected saliency methods (e.g., Integrated Gradients, LIME, DEMUX, Series Saliency, Random Attribution) to generate saliency maps for a representative set of model predictions on the test data. ---
# To use this code, run: pip install lime
# To use this code, run: pip install captum
import torch
import torch.nn as nn
import numpy as np

from captum.attr import IntegratedGradients
from lime.lime_tabular import LimeTabularExplainer

# Dummy Model
class DummyModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Dummy Test Data
input_dim = 10
num_classes = 3
num_test_samples = 100
test_data = torch.randn(num_test_samples, input_dim)

model = DummyModel(input_dim, num_classes)
model.eval()

# Helper for LIME: predict_proba function
def predict_proba_for_lime(input_data_np):
    with torch.no_grad():
        input_data_torch = torch.tensor(input_data_np, dtype=torch.float32)
        logits = model(input_data_torch)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities.numpy()

# Integrated Gradients
def apply_integrated_gradients(model, input_tensor, target_class_idx):
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor)
    attributions = ig.attribute(input_tensor.unsqueeze(0), baselines=baseline.unsqueeze(0), target=target_class_idx)
    return attributions.squeeze(0)

# LIME
def apply_lime(model, input_tensor, feature_names, class_names, training_data_np, target_class_idx):
    explainer = LimeTabularExplainer(
        training_data=training_data_np,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=input_tensor.numpy(),
        predict_fn=predict_proba_for_lime,
        num_features=input_dim,
        top_labels=1
    )
    saliency_map = np.zeros(input_dim)
    exp_list = explanation.as_list(label=target_class_idx)
    feature_name_to_idx = {name: i for i, name in enumerate(feature_names)}
    for feature_name, score in exp_list:
        if feature_name in feature_name_to_idx:
            saliency_map[feature_name_to_idx[feature_name]] = score
    return torch.tensor(saliency_map, dtype=torch.float32)

# DEMUX (placeholder)
def apply_demux_saliency(input_tensor):
    # Placeholder: returns random saliency
    return torch.randn_like(input_tensor)

# Series Saliency (placeholder)
def apply_series_saliency(input_tensor):
    # Placeholder: returns random saliency
    return torch.randn_like(input_tensor)

# Random Attribution
def apply_random_attribution(input_tensor):
    return torch.randn_like(input_tensor)

# Select representative samples
representative_indices = [0, 10, 20, 30, 40]
representative_samples = test_data[representative_indices]

saliency_maps = {
    'integrated_gradients': [],
    'lime': [],
    'demux': [],
    'series_saliency': [],
    'random_attribution': []
}

# Prepare data for LIME explainer
training_data_np = test_data.numpy()
feature_names = [f'feature_{i}' for i in range(input_dim)]
class_names = [f'class_{i}' for i in range(num_classes)]

for sample_input in representative_samples:
    with torch.no_grad():
        output = model(sample_input.unsqueeze(0))
        predicted_class = torch.argmax(output, dim=1).item()

    ig_map = apply_integrated_gradients(model, sample_input, predicted_class)
    saliency_maps['integrated_gradients'].append(ig_map)

    lime_map = apply_lime(model, sample_input, feature_names, class_names, training_data_np, predicted_class)
    saliency_maps['lime'].append(lime_map)

    demux_map = apply_demux_saliency(sample_input)
    saliency_maps['demux'].append(demux_map)

    series_map = apply_series_saliency(sample_input)
    saliency_maps['series_saliency'].append(series_map)

    random_map = apply_random_attribution(sample_input)
    saliency_maps['random_attribution'].append(random_map)

# --- Implement quantitative evaluation metrics for saliency maps, including fidelity (AOPC, deletion/insertion curves, sensitivity-n), consistency (correlation, divergence for overlapping windows), robustness (correlation/distance after feature swaps/perturbations), sparsity (L0/L1 norm), and smoothness (Total Variation norm or Series Saliency's 'r' penalty). ---
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.special import rel_entr

# Placeholder for a model prediction function
# In a real scenario, this would be your actual model's prediction method.
def _model_predict_placeholder(image, target_class=None):
    """
    Placeholder for a model's prediction function.
    Returns a scalar score for the target class.
    """
    # Simulate a score based on image properties
    if target_class is None:
        return np.mean(image) # Example: mean pixel value
    else:
        # Simulate class-specific score
        return np.mean(image) * (1 + target_class * 0.1) # Dummy score

def _perturb_image(image, mask, perturbation_value=0.0):
    """Perturbs an image by setting masked pixels to a specific value."""
    perturbed_image = image.copy()
    perturbed_image[mask] = perturbation_value
    return perturbed_image

def aopc(image, saliency_map, model_predict_fn, target_class, num_steps=100, perturbation_value=0.0):
    """
    Calculates Area Over Perturbation Curve (AOPC) for deletion.
    Lower AOPC indicates better fidelity for deletion.
    """
    flat_saliency = saliency_map.flatten()
    sorted_indices = np.argsort(flat_saliency)[::-1] # Highest saliency first
    original_score = model_predict_fn(image, target_class)
    scores = [original_score]

    for i in range(1, num_steps + 1):
        num_pixels_to_perturb = int(len(flat_saliency) * (i / num_steps))
        mask_indices = sorted_indices[:num_pixels_to_perturb]
        
        perturb_mask = np.zeros_like(saliency_map, dtype=bool)
        np.put(perturb_mask, mask_indices, True)
        
        perturbed_image = _perturb_image(image, perturb_mask, perturbation_value)
        scores.append(model_predict_fn(perturbed_image, target_class))
    
    # AOPC is the area between the original score and the perturbation curve
    # For deletion, we expect scores to drop, so we sum the differences.
    return np.mean(original_score - np.array(scores))

def deletion_insertion_curve(image, saliency_map, model_predict_fn, target_class, num_steps=100, perturbation_value=0.0, mode='deletion'):
    """
    Generates scores for deletion or insertion curves.
    mode: 'deletion' (remove high saliency) or 'insertion' (add high saliency)
    """
    flat_saliency = saliency_map.flatten()
    sorted_indices = np.argsort(flat_saliency) # Lowest saliency first
    if mode == 'deletion':
        sorted_indices = sorted_indices[::-1] # Highest saliency first for deletion

    original_score = model_predict_fn(image, target_class)
    scores = [original_score]
    
    base_image = np.full_like(image, perturbation_value) if mode == 'insertion' else image

    for i in range(1, num_steps + 1):
        num_pixels_to_perturb = int(len(flat_saliency) * (i / num_steps))
        current_indices = sorted_indices[:num_pixels_to_perturb]
        
        perturb_mask = np.zeros_like(saliency_map, dtype=bool)
        np.put(perturb_mask, current_indices, True)
        
        if mode == 'deletion':
            perturbed_image = _perturb_image(image, perturb_mask, perturbation_value)
        else: # insertion
            # Start with base_image and insert original pixels
            perturbed_image = base_image.copy()
            perturbed_image[perturb_mask] = image[perturb_mask]

        scores.append(model_predict_fn(perturbed_image, target_class))
    
    return np.array(scores)

def sensitivity_n(image, saliency_map, model_predict_fn, target_class, n_pixels, perturbation_value=0.0):
    """
    Measures the change in model output when the top-n salient features are perturbed.
    """
    flat_saliency = saliency_map.flatten()
    sorted_indices = np.argsort(flat_saliency)[::-1] # Highest saliency first
    
    original_score = model_predict_fn(image, target_class)
    
    mask_indices = sorted_indices[:n_pixels]
    perturb_mask = np.zeros_like(saliency_map, dtype=bool)
    np.put(perturb_mask, mask_indices, True)
    
    perturbed_image = _perturb_image(image, perturb_mask, perturbation_value)
    perturbed_score = model_predict_fn(perturbed_image, target_class)
    
    return np.abs(original_score - perturbed_score)

def saliency_correlation(saliency_map1, saliency_map2, method='pearson'):
    """
    Calculates correlation between two saliency maps.
    method: 'pearson' or 'spearman'
    """
    flat_saliency1 = saliency_map1.flatten()
    flat_saliency2 = saliency_map2.flatten()
    
    if method == 'pearson':
        return pearsonr(flat_saliency1, flat_saliency2)[0]
    elif method == 'spearman':
        return spearmanr(flat_saliency1, flat_saliency2)[0]
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")

def saliency_kl_divergence(saliency_map1, saliency_map2):
    """
    Calculates KL divergence between two saliency maps.
    Saliency maps are normalized to sum to 1 to represent probability distributions.
    """
    # Ensure non-negative and normalize to sum to 1
    p = np.maximum(saliency_map1, 1e-10).flatten()
    q = np.maximum(saliency_map2, 1e-10).flatten()
    
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(rel_entr(p, q))

def robustness_metric(image, saliency_map_fn, perturbation_fn, target_class, num_perturbations=10, correlation_method='pearson'):
    """
    Measures robustness by perturbing the input and checking saliency map correlation.
    saliency_map_fn: A function that takes (image, target_class) and returns a saliency map.
    perturbation_fn: A function that takes (image) and returns a slightly perturbed image.
    """
    original_saliency = saliency_map_fn(image, target_class)
    correlations = []

    for _ in range(num_perturbations):
        perturbed_image = perturbation_fn(image)
        perturbed_saliency = saliency_map_fn(perturbed_image, target_class)
        correlations.append(saliency_correlation(original_saliency, perturbed_saliency, correlation_method))
    
    return np.mean(correlations)

def sparsity_l0(saliency_map, threshold=1e-6):
    """Calculates L0 norm (number of non-zero elements) of a saliency map."""
    return np.sum(np.abs(saliency_map) > threshold)

def sparsity_l1(saliency_map):
    """Calculates L1 norm (sum of absolute values) of a saliency map."""
    return np.sum(np.abs(saliency_map))

def smoothness_total_variation(saliency_map):
    """
    Calculates the Total Variation (TV) norm of a saliency map.
    Assumes 2D saliency map.
    """
    if saliency_map.ndim != 2:
        raise ValueError("Total Variation norm expects a 2D saliency map.")
    
    # Horizontal differences
    tv_h = np.sum(np.abs(saliency_map[:, 1:] - saliency_map[:, :-1]))
    # Vertical differences
    tv_v = np.sum(np.abs(saliency_map[1:, :] - saliency_map[:-1, :]))
    
    return tv_h + tv_v

# Example Usage (requires dummy data and functions)
if __name__ == '__main__':
    # Dummy data
    dummy_image = np.random.rand(32, 32)
    dummy_saliency_map = np.random.rand(32, 32)
    dummy_saliency_map_2 = np.random.rand(32, 32)
    dummy_target_class = 0

    # Dummy saliency map function for robustness
    def _dummy_saliency_fn(img, target_cls):
        # In a real scenario, this would be your actual saliency method
        return np.random.rand(*img.shape) # Returns a random map for demonstration

    # Dummy perturbation function for robustness
    def _dummy_perturbation_fn(img):
        return img + np.random.normal(0, 0.01, img.shape) # Add small Gaussian noise

    print("--- Fidelity Metrics ---")
    aopc_score = aopc(dummy_image, dummy_saliency_map, _model_predict_placeholder, dummy_target_class)
    print(f"AOPC (deletion): {aopc_score:.4f}")

    deletion_scores = deletion_insertion_curve(dummy_image, dummy_saliency_map, _model_predict_placeholder, dummy_target_class, mode='deletion')
    print(f"Deletion curve (first 5 scores): {deletion_scores[:5]}")

    insertion_scores = deletion_insertion_curve(dummy_image, dummy_saliency_map, _model_predict_placeholder, dummy_target_class, mode='insertion')
    print(f"Insertion curve (first 5 scores): {insertion_scores[:5]}")

    sens_n = sensitivity_n(dummy_image, dummy_saliency_map, _model_predict_placeholder, dummy_target_class, n_pixels=10)
    print(f"Sensitivity-N (top 10 pixels): {sens_n:.4f}")

    print("\n--- Consistency Metrics ---")
    pearson_corr = saliency_correlation(dummy_saliency_map, dummy_saliency_map_2, method='pearson')
    print(f"Pearson Correlation: {pearson_corr:.4f}")

    spearman_corr = saliency_correlation(dummy_saliency_map, dummy_saliency_map_2, method='spearman')
    print(f"Spearman Correlation: {spearman_corr:.4f}")

    kl_div = saliency_kl_divergence(dummy_saliency_map, dummy_saliency_map_2)
    print(f"KL Divergence: {kl_div:.4f}")

    print("\n--- Robustness Metrics ---")
    robustness_score = robustness_metric(dummy_image, _dummy_saliency_fn, _dummy_perturbation_fn, dummy_target_class)
    print(f"Robustness (avg Pearson corr after perturbation): {robustness_score:.4f}")

    print("\n--- Sparsity Metrics ---")
    l0_norm = sparsity_l0(dummy_saliency_map)
    print(f"L0 Norm (non-zero elements): {l0_norm}")

    l1_norm = sparsity_l1(dummy_saliency_map)
    print(f"L1 Norm (sum of absolute values): {l1_norm:.4f}")

    print("\n--- Smoothness Metrics ---")
    tv_norm = smoothness_total_variation(dummy_saliency_map)
    print(f"Total Variation Norm: {tv_norm:.4f}")

# --- Develop a user interface for presenting time series data, model predictions, and saliency maps to human experts, and for collecting their qualitative feedback via Likert scales and open-ended questions. ---
# To use this code, run: pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dummy data generation
time_index = pd.date_range(start='2023-01-01', periods=100, freq='H')
time_series_data = pd.DataFrame({
    'value': np.random.rand(100).cumsum() + 50,
    'prediction': np.random.rand(100).cumsum() + 48 + np.sin(np.linspace(0, 10, 100)) * 5
}, index=time_index)
saliency_map_data = pd.DataFrame({
    'saliency': np.random.rand(100) * 10
}, index=time_index)

st.title("Time Series Data Analysis and Feedback")

st.header("Time Series Data and Predictions")
st.line_chart(time_series_data)

st.header("Saliency Map")
st.bar_chart(saliency_map_data)

st.header("Qualitative Feedback")

with st.form("feedback_form"):
    st.subheader("Likert Scale Questions")
    likert_q1 = st.radio(
        "How well does the model prediction align with the time series data?",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x} - {'Strongly Disagree' if x==1 else 'Disagree' if x==2 else 'Neutral' if x==3 else 'Agree' if x==4 else 'Strongly Agree'}",
        index=2
    )
    likert_q2 = st.radio(
        "How helpful is the saliency map in understanding the prediction?",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x} - {'Not Helpful' if x==1 else 'Slightly Helpful' if x==2 else 'Neutral' if x==3 else 'Helpful' if x==4 else 'Very Helpful'}",
        index=2
    )

    st.subheader("Open-ended Questions")
    open_q1 = st.text_area("What are your overall observations about this data and prediction?")
    open_q2 = st.text_area("Do you have any suggestions for improving the model or its explanations?")

    submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        # In a real application, this data would be saved to a database or file
        st.success("Feedback submitted!")

# --- Tags: ---
tags = []

# --- Load and preprocess time series datasets, including normalization, windowing (if applicable), and train-validation-test splitting. ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Load data (example: dummy data)
data = pd.DataFrame(np.random.rand(100, 1), columns=['value'])

# 2. Normalize data
scaler = MinMaxScaler()
data['value_normalized'] = scaler.fit_transform(data[['value']])

# 3. Windowing function
def create_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 10
X, y = create_windows(data['value_normalized'].values, window_size)

# 4. Train-validation-test splitting
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

total_samples = len(X)
train_split = int(total_samples * train_ratio)
val_split = int(total_samples * (train_ratio + val_ratio))

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

# --- Implement and train selected deep learning models (e.g., CNN, LSTM, Transformer) for time series classification or forecasting tasks, including hyperparameter tuning and saving model states. ---
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Generate synthetic time series data
num_samples = 1000
timesteps = 50
num_features = 1
num_classes = 3

X = np.random.rand(num_samples, timesteps, num_features).astype(np.float32)
y = np.random.randint(0, num_classes, num_samples).astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define a CNN model for time series classification
def build_cnn_model(learning_rate, num_filters):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=num_filters, kernel_size=5, activation='relu', input_shape=(timesteps, num_features)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Hyperparameter tuning (example with learning rate and num_filters)
best_accuracy = -1
best_model = None
best_hyperparams = {}

learning_rates = [0.01, 0.001]
num_filters_options = [32, 64]

for lr in learning_rates:
    for filters in num_filters_options:
        model = build_cnn_model(lr, filters)
        history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
        val_accuracy = history.history['val_accuracy'][-1]

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            best_hyperparams = {'learning_rate': lr, 'num_filters': filters}

# 4. Train the best model (if not already trained sufficiently in tuning)
# For this minimal example, the best_model is already trained.
# If more epochs were needed, it would be trained here.
# best_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# 5. Save the best model's state
best_model.save('best_time_series_cnn_model.h5')

# --- Implement and apply selected saliency methods (e.g., Integrated Gradients, LIME, DEMUX, Series Saliency, Random Attribution) to generate saliency maps for a representative set of model predictions on the test data. ---
# To use this code, run: pip install lime
# To use this code, run: pip install captum
import torch
import torch.nn as nn
import numpy as np

# Captum for Integrated Gradients
from captum.attr import IntegratedGradients

# LIME for LIME
import lime
import lime.lime_tabular

# 1. Define a dummy model
class DummyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc(x))

# 2. Generate dummy test data
input_dim = 10
output_dim = 3
num_samples = 100
test_data = torch.randn(num_samples, input_dim)
test_labels = torch.randint(0, output_dim, (num_samples,))

# Instantiate model
model = DummyModel(input_dim, output_dim)
model.eval()

# 3. Select a representative set of model predictions
num_samples_to_explain = 3
representative_inputs = test_data[:num_samples_to_explain]
representative_labels = test_labels[:num_samples_to_explain]

saliency_maps = {}

# --- Integrated Gradients ---
ig = IntegratedGradients(model)
ig_attributions = []
for i in range(num_samples_to_explain):
    input_tensor = representative_inputs[i].unsqueeze(0)
    target_label = representative_labels[i].item()
    attributions = ig.attribute(input_tensor, target=target_label)
    ig_attributions.append(attributions.squeeze(0).detach().cpu().numpy())
saliency_maps['Integrated_Gradients'] = ig_attributions

# --- LIME ---
def predict_fn_lime(input_data_np):
    input_tensor = torch.tensor(input_data_np, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs.cpu().numpy()

feature_names = [f'feature_{i}' for i in range(input_dim)]
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=test_data.numpy(),
    feature_names=feature_names,
    class_names=[f'class_{i}' for i in range(output_dim)],
    mode='classification'
)

lime_attributions = []
for i in range(num_samples_to_explain):
    input_np = representative_inputs[i].cpu().numpy()
    target_label = representative_labels[i].item()
    explanation = explainer.explain_instance(
        data_row=input_np,
        predict_fn=predict_fn_lime,
        num_features=input_dim,
        top_labels=1
    )
    attr_dict = {f: w for f, w in explanation.as_list()}
    attribution_array = np.array([attr_dict.get(f, 0.0) for f in feature_names])
    lime_attributions.append(attribution_array)
saliency_maps['LIME'] = lime_attributions

# --- Random Attribution ---
random_attributions = []
for i in range(num_samples_to_explain):
    random_attr = np.random.rand(input_dim)
    random_attributions.append(random_attr)
saliency_maps['Random_Attribution'] = random_attributions

# --- Implement quantitative evaluation metrics for saliency maps, including fidelity (AOPC, deletion/insertion curves, sensitivity-n), consistency (correlation, divergence for overlapping windows), robustness (correlation/distance after feature swaps/perturbations), sparsity (L0/L1 norm), and smoothness (Total Variation norm or Series Saliency's 'r' penalty). ---
import numpy as np
from scipy.stats import pearsonr
from scipy.special import rel_entr # For KL divergence

# --- Helper functions ---
def _get_model_output_for_target_class(model, input_batch, target_class):
    # Assumes model returns logits/probabilities for a batch
    outputs = model(input_batch)
    return outputs[:, target_class]

def _perturb_image_with_mask(image, mask, baseline_value=0.0):
    # mask: boolean array, True where pixels should be perturbed
    perturbed_image = np.copy(image)
    perturbed_image[mask] = baseline_value
    return perturbed_image

# --- Fidelity Metrics ---

def deletion_insertion_curve(model, input_tensor, target_class, saliency_map, mode='deletion', num_steps=100, baseline_value=0.0):
    """
    Generates deletion or insertion curve.
    mode='deletion': gradually remove most salient pixels.
    mode='insertion': gradually add most salient pixels to a baseline image.
    Returns a list of model outputs at each step.
    """
    original_output = _get_model_output_for_target_class(model, np.expand_dims(input_tensor, 0), target_class)[0]
    flat_saliency = saliency_map.flatten()

    if mode == 'deletion':
        sorted_indices = np.argsort(flat_saliency)[::-1] # Most salient first
    elif mode == 'insertion':
        sorted_indices = np.argsort(flat_saliency) # Least salient first (to add most salient last)
    else:
        raise ValueError("Mode must be 'deletion' or 'insertion'")

    num_total_pixels = len(sorted_indices)
    step_indices = np.unique(np.linspace(0, num_total_pixels, num_steps, endpoint=True, dtype=int))
    if step_indices[-1] != num_total_pixels:
        step_indices = np.append(step_indices, num_total_pixels)

    outputs_at_steps = []
    
    for k in step_indices:
        if k == 0:
            current_image = input_tensor if mode == 'deletion' else np.full_like(input_tensor, baseline_value)
        else:
            indices_to_change = sorted_indices[:k]
            rows, cols = np.unravel_index(indices_to_change, saliency_map.shape)
            
            if mode == 'deletion':
                mask = np.zeros_like(input_tensor, dtype=bool)
                mask[rows, cols] = True
                current_image = _perturb_image_with_mask(input_tensor, mask, baseline_value)
            elif mode == 'insertion':
                current_image = np.full_like(input_tensor, baseline_value)
                mask = np.zeros_like(input_tensor, dtype=bool)
                mask[rows, cols] = True
                current_image[mask] = input_tensor[mask]

        output = _get_model_output_for_target_class(model, np.expand_dims(current_image, 0), target_class)[0]
        outputs_at_steps.append(output)

    return np.array(outputs_at_steps)

def aopc(model, input_tensor, target_class, saliency_map, num_steps=100, baseline_value=0.0):
    """
    Area Over Perturbation Curve (AOPC) - typically for deletion.
    Calculates the area under the curve of (original_output - current_output) vs. fraction of pixels perturbed.
    """
    original_output = _get_model_output_for_target_class(model, np.expand_dims(input_tensor, 0), target_class)[0]
    
    curve_outputs = deletion_insertion_curve(model, input_tensor, target_class, saliency_map, 
                                             mode='deletion', num_steps=num_steps, baseline_value=baseline_value)
    
    probability_drop = original_output - curve_outputs
    
    num_total_pixels = saliency_map.size
    step_indices = np.unique(np.linspace(0, num_total_pixels, num_steps, endpoint=True, dtype=int))
    if step_indices[-1] != num_total_pixels:
        step_indices = np.append(step_indices, num_total_pixels)
    
    x_axis = step_indices / num_total_pixels
    
    return np.trapz(probability_drop, x=x_axis)

def sensitivity_n(model, input_tensor, target_class, saliency_map, n, baseline_value=0.0):
    """
    Measures the change in model output when the 'n' most salient features are perturbed.
    """
    original_output = _get_model_output_for_target_class(model, np.expand_dims(input_tensor, 0), target_class)[0]

    flat_saliency = saliency_map.flatten()
    sorted_indices = np.argsort(flat_saliency)[::-1] # Most salient first

    n_actual = min(n, len(sorted_indices))
    if n_actual == 0:
        return 0.0

    indices_to_perturb = sorted_indices[:n_actual]
    mask = np.zeros_like(input_tensor, dtype=bool)
    rows, cols = np.unravel_index(indices_to_perturb, saliency_map.shape)
    mask[rows, cols] = True

    perturbed_input = _perturb_image_with_mask(input_tensor, mask, baseline_value)
    perturbed_output = _get_model_output_for_target_class(model, np.expand_dims(perturbed_input, 0), target_class)[0]

    return np.abs(original_output - perturbed_output)

# --- Consistency Metrics ---

def saliency_correlation(saliency_map1, saliency_map2):
    """
    Calculates Pearson correlation coefficient between two saliency maps.
    """
    flat_map1 = saliency_map1.flatten()
    flat_map2 = saliency_map2.flatten()
    if np.std(flat_map1) == 0 or np.std(flat_map2) == 0:
        return 0.0
    corr, _ = pearsonr(flat_map1, flat_map2)
    return corr

def saliency_divergence_windows(saliency_map1, saliency_map2, window_size):
    """
    Calculates average KL divergence between overlapping windows of two saliency maps.
    Saliency maps are normalized to sum to 1 within each window.
    """
    h, w = saliency_map1.shape
    if saliency_map1.shape != saliency_map2.shape:
        raise ValueError("Saliency maps must have the same shape.")

    total_divergence = 0.0
    num_windows = 0

    for r in range(0, h - window_size + 1):
        for c in range(0, w - window_size + 1):
            window1 = saliency_map1[r:r+window_size, c:c+window_size].flatten()
            window2 = saliency_map2[r:r+window_size, c:c+window_size].flatten()

            epsilon = 1e-10
            p = window1 / (np.sum(window1) + epsilon)
            q = window2 / (np.sum(window2) + epsilon)

            q[q == 0] = epsilon

            total_divergence += np.sum(rel_entr(p, q))
            num_windows += 1

    return total_divergence / num_windows if num_windows > 0 else 0.0

# --- Robustness Metrics ---

def robustness_correlation(original_saliency_map, perturbed_saliency_maps):
    """
    Measures robustness by correlating original saliency map with a list of maps generated from perturbed inputs.
    perturbed_saliency_maps: A list of saliency maps generated from slightly perturbed versions of the original input.
    """
    correlations = []
    for p_map in perturbed_saliency_maps:
        corr = saliency_correlation(original_saliency_map, p_map)
        correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0

# --- Sparsity Metrics ---

def l0_norm(saliency_map, threshold=1e-6):
    """
    Calculates the L0 norm (number of non-zero elements) of a saliency map.
    """
    return np.sum(np.abs(saliency_map) > threshold)

def l1_norm(saliency_map):
    """
    Calculates the L1 norm (sum of absolute values) of a saliency map.
    """
    return np.sum(np.abs(saliency_map))

# --- Smoothness Metrics ---

def total_variation_norm(saliency_map):
    """
    Calculates the Total Variation (TV) norm of a saliency map.
    Sum of absolute differences between adjacent pixels.
    """
    if saliency_map.ndim < 2:
        return 0.0

    tv_h = np.sum(np.abs(saliency_map[:, 1:] - saliency_map[:, :-1]))
    tv_w = np.sum(np.abs(saliency_map[1:, :] - saliency_map[:-1, :]))
    return tv_h + tv_w

# --- Develop a user interface for presenting time series data, model predictions, and saliency maps to human experts, and for collecting their qualitative feedback via Likert scales and open-ended questions. ---
# To use this code, run: pip install streamlit
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dummy data generation
time_index = pd.date_range(start='2023-01-01', periods=100, freq='H')
time_series_data = np.random.randn(100).cumsum()
model_predictions = time_series_data * 0.9 + np.random.randn(100) * 0.5
saliency_map_data = np.abs(np.random.randn(100))

st.title("Time Series Analysis and Expert Feedback")

st.header("Data Visualization")

# Time series and predictions plot
fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
ax_ts.plot(time_index, time_series_data, label="Actual Data")
ax_ts.plot(time_index, model_predictions, label="Model Predictions", linestyle='--')
ax_ts.set_title("Time Series Data and Model Predictions")
ax_ts.legend()
st.pyplot(fig_ts)

# Saliency map plot
fig_saliency, ax_saliency = plt.subplots(figsize=(10, 2))
ax_saliency.bar(time_index, saliency_map_data, width=0.01, color='red', alpha=0.6)
ax_saliency.set_title("Saliency Map")
st.pyplot(fig_saliency)

st.header("Qualitative Feedback")

st.subheader("Likert Scale Questions")
st.radio("How confident are you in the model's predictions?", [1, 2, 3, 4, 5], index=2, key="likert_confidence")
st.radio("How well does the saliency map explain the predictions?", [1, 2, 3, 4, 5], index=2, key="likert_saliency_explanation")

st.subheader("Open-ended Questions")
st.text_area("What are your overall observations about the data and predictions?", key="open_ended_observations")
st.text_area("Do you have any specific concerns or suggestions?", key="open_ended_concerns")

st.button("Submit Feedback")
```
```
