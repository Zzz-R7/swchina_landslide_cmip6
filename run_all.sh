#!/bin/bash
echo "Starting Landslide Susceptibility Pipeline..."

python src/00_preprocess_environment.py
python src/01_preprocess_climate.py
python src/02_vif_multicollinearity.py
python src/03_01hpbsf_sampling_iv.py
python src/03_02hpbsf_sampling.py
python src/04_train_models.py
python src/05_spatial_cv_evaluate.py
python src/06_01shap_analysis.py
python src/06_02shap_analysis_fig.py
python src/06_03shap_analysis_fig.py
python src/07_future_projection.py
python src/08_uncertainty_decomposition.py
python src/09_01transition_centroid.py
python src/09_02transition_centroid.py
python src/10_moran_lisa.py
python src/11_01robust_hotspots.py
python src/11_02robust_hotspots_fig.py
python src/11_03robust_hotspots_fig.py

echo "Pipeline completed successfully!"