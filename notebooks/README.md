# Team walkthrough

Open `catalogue_quality_walkthrough.ipynb` in JupyterLab or VS Code. It contains executed outputs, so you can read the report without retraining. `catalogue_quality_walkthrough.html` is the browser-readable copy for sharing with teammates.

The notebook explains data checks, extracting product facts from text, category surprisal, expected prices and weights, nested seller validation, CatBoost, results, feature importance, missing-description errors and the submission. Core modelling functions are defined in the notebook itself.

To rerun it, keep the project's `data/` folder in place. From the project directory:

```powershell
python -m pip install -r requirements-notebook.txt
python -m jupyterlab notebooks/catalogue_quality_walkthrough.ipynb
```

Choose **Restart Kernel and Run All Cells**. It fits five outer folds with three inner seller folds on CPU, matching the working model's settings. Notebook outputs are refreshed under `outputs/notebook_walkthrough/`; original data and production model bundles are preserved.

To rebuild, execute, and export the notebook from its maintained source:

```powershell
python analysis/build_team_notebook.py --execute --html
```

The notebook can also run with the data and notebook dependencies alone; it does not import the project's model package. The two historical model-comparison rows are shown only if `analysis/research_seed_42.json` is available. All main training results and charts come from executing the notebook.
