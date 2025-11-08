

## Generated Artifacts
- artifacts/best_model_mlp.joblib
- artifacts/confusion_matrix.png
- artifacts/roc_curve.png
- artifacts/feature_importance.png

Dockerfile and requirements.txt included. Use `docker build -t readmission-app .` to build.


## Generated artifacts (MLP run)
- artifacts/best_model_mlp.joblib
- artifacts/confusion_matrix.png
- artifacts/roc_curve.png
- artifacts/feature_importance.png

To build Docker image: `docker build -t readmission-app .`
To run: `docker run -p 8501:8501 readmission-app`


## Final artifacts (created)
- artifacts/best_model_mlp.joblib
- artifacts/confusion_matrix.png
- artifacts/roc_curve.png
- artifacts/feature_importance.png
- artifacts/classification_report.txt
- artifacts/shap_error.txt (see if SHAP failed due to environment limitations)

To recreate locally: run `main.ipynb` and `main_explainability.ipynb`. To run the Streamlit app: `streamlit run streamlit_app.py`.
