"""YZTA 2026 Datathon — Pipeline CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import typer
from loguru import logger
from rich.console import Console

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import settings
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
from src.utils.logger import setup_custom_logger

setup_custom_logger()

app = typer.Typer(name="datathon-core", help="YZTA 2026 Datathon Pipeline", add_completion=False)
console = Console()


def _load_processed() -> pl.DataFrame:
    path = settings.DATA_DIR / "processed_data.parquet"
    if not path.exists():
        logger.error("Run 'preprocess' first.")
        raise typer.Exit(1)
    return pl.read_parquet(path)


def _get_features(df: pl.DataFrame) -> list[str]:
    ignore = {settings.ID_COL, settings.TARGET_COL, "is_train", "age_decade"}
    features = [c for c in df.columns if c not in ignore]
    drop_file = settings.DATA_DIR / "dropped_features.json"
    if drop_file.exists():
        try:
            with open(drop_file) as f:
                dropped = json.load(f)
            features = [f for f in features if f not in dropped]
            logger.info(f"Feature pruning active: {len(dropped)} dropped, {len(features)} remaining")
        except Exception:
            pass
    return features


def _detect_gpu() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


@app.command()
def info() -> None:
    """System, hardware and strategy summary."""
    console.rule("[bold blue]INFO[/bold blue]")
    gpu = _detect_gpu()
    logger.info(f"Device: {'GPU' if gpu else 'CPU'}")
    logger.info(f"Seeds: {settings.SEEDS}")
    logger.info(f"Folds: {settings.N_FOLDS}")
    logger.info("Strategy: multi-model × multi-target × multi-seed → Caruana FS → L2 Ridge/LGB → L3 blend")


@app.command()
def check() -> None:
    """Ruff + mypy code quality check."""
    console.rule("[bold yellow]CHECK[/bold yellow]")
    try:
        subprocess.run(["ruff", "check", "src/"], check=True)
        subprocess.run(["mypy", "src/"], check=True)
        logger.success("Code quality OK.")
    except subprocess.CalledProcessError:
        logger.error("Quality checks failed.")


@app.command()
def test() -> None:
    """Run unit tests with coverage."""
    console.rule("[bold magenta]TEST[/bold magenta]")
    subprocess.run(["pytest", "--cov=src", "tests/"], check=True)


@app.command()
def preprocess() -> None:
    """Load data, run feature engineering, save parquet."""
    console.rule("[bold cyan]PREPROCESS[/bold cyan]")
    try:
        loader = DataLoader()
        raw_df = loader.load_combined()
        logger.info(f"Raw data: {raw_df.shape[0]} rows × {raw_df.shape[1]} cols")
        processed_df = FeatureEngineer.fit_transform(raw_df)
        out_path = settings.DATA_DIR / "processed_data.parquet"
        processed_df.write_parquet(out_path)
        logger.success(f"Done → {out_path}")
    except Exception as e:
        logger.exception(f"Preprocess error: {e}")


@app.command()
def eda() -> None:
    """Exploratory data analysis — 6 chart types."""
    console.rule("[bold blue]EDA[/bold blue]")
    from src.features.eda import EDAReport
    df = _load_processed()
    report = EDAReport()
    report.run_full_eda(df)


@app.command()
def adversarial() -> None:
    """Train vs test distribution shift check."""
    console.rule("[bold red]ADVERSARIAL VALIDATION[/bold red]")
    from src.features.adversarial import AdversarialValidator
    df = _load_processed()
    train_df = df.filter(pl.col("is_train") == 1).to_pandas()
    test_df = df.filter(pl.col("is_train") == 0).to_pandas()
    features = _get_features(df)
    cat_features = [c for c in settings.CAT_FEATURES if c in features]
    adv_auc, test_likeness = AdversarialValidator.run(train_df, test_df, features, cat_features=cat_features)
    np.save(settings.DATA_DIR / "test_likeness.npy", test_likeness)
    logger.info(f"Test-likeness saved (mean={test_likeness.mean():.3f})")


@app.command()
def hpo(
    n_trials: int = typer.Option(settings.HPO_TRIALS, "--trials", "-n"),
    model: str = typer.Option("catboost", "--model", "-m"),
) -> None:
    """Optuna hyperparameter optimization."""
    console.rule(f"[bold yellow]HPO — {model.upper()}[/bold yellow]")
    from src.models.hpo import run_optuna_hpo
    df = _load_processed()
    train_df = df.filter(pl.col("is_train") == 1)
    train_pd = train_df.to_pandas()
    features = _get_features(df)
    cat_features = [c for c in (settings.CAT_FEATURES + settings.CAT_INTERACTION_FEATURES) if c in features]
    y_train = train_df.get_column(settings.TARGET_COL).to_numpy().astype(np.float32)
    y_low, y_high = np.percentile(y_train, settings.WINSOR_LIMITS)
    y_train_winsor = np.clip(y_train, y_low, y_high)
    weights = FeatureEngineer.calculate_sample_weights(y_train)
    from src.models.validator import CrossValidator
    y_bins = CrossValidator.bin_target(y_train)
    use_gpu = _detect_gpu()
    best_params = run_optuna_hpo(
        train_pd[features], y_train_winsor, y_train, y_bins,
        cat_features=cat_features, sample_weights=weights,
        n_trials=n_trials, use_gpu=use_gpu, model_name=model,
    )
    logger.success(f"Best params: {best_params}")


_DEFAULT_MODELS = ["catboost", "lightgbm", "xgboost", "mlp", "ridge"]
_DEFAULT_TARGETS = ["A_direct"]


@app.command(name="engine")
def engine(
    models: list[str] = typer.Option(_DEFAULT_MODELS, "--model", "-m"),
    targets: list[str] = typer.Option(_DEFAULT_TARGETS, "--target", "-t"),
) -> None:
    """Multi-model × multi-target × multi-seed OOF training."""
    console.rule("[bold green]ENGINE[/bold green]")
    from src.models.engine import ModelEngine
    from src.models.target_strategy import TargetStrategy
    from src.models.validator import CrossValidator

    df = _load_processed()
    train_df = df.filter(pl.col("is_train") == 1)
    test_df = df.filter(pl.col("is_train") == 0)
    features = _get_features(df)
    cat_features = [c for c in (settings.CAT_FEATURES + settings.CAT_INTERACTION_FEATURES) if c in features]

    train_pd = train_df.to_pandas()
    test_pd = test_df.to_pandas()
    y_train = train_df.get_column(settings.TARGET_COL).to_numpy().astype(np.float32)

    y_low, y_high = np.percentile(y_train, settings.WINSOR_LIMITS)
    y_train_winsor = np.clip(y_train, y_low, y_high)
    n_clipped = int(((y_train < y_low) | (y_train > y_high)).sum())
    logger.info(f"Winsorization: {n_clipped} samples clipped")

    weights = FeatureEngineer.calculate_sample_weights(y_train)
    target_strat = TargetStrategy(train_pd, y_train_winsor)
    target_strat.summary()
    y_bins = CrossValidator.bin_target(y_train)
    data_dict = ModelEngine.prepare_data(train_pd, test_pd, features, cat_features)

    cb_params_path = settings.MODELS_DIR / "best_cb_params.json"
    if cb_params_path.exists():
        with open(cb_params_path) as f:
            best_cb = json.load(f)
        logger.info(f"Loaded HPO params: {cb_params_path}")
    else:
        from src.models.hpo import DEFAULT_CB_PARAMS
        best_cb = DEFAULT_CB_PARAMS.copy()
        logger.warning("HPO params not found — using defaults")

    use_gpu = _detect_gpu()
    engine_model = ModelEngine(best_cb_params=best_cb, use_gpu=use_gpu)

    all_oof, all_test = engine_model.run_full_training(
        data_dict=data_dict,
        y_train_orig=y_train,
        y_bins=y_bins,
        target_strategy=target_strat,
        target_names=targets,
        model_names=models,
        train_df_pd=train_pd,
        test_df_pd=test_pd,
        sample_weights=weights,
    )

    for key, oof in all_oof.items():
        np.save(settings.DATA_DIR / f"oof_{key}.npy", oof)
    for key, tst in all_test.items():
        np.save(settings.DATA_DIR / f"test_{key}.npy", tst)
    logger.success(f"{len(all_oof)} models trained. OOF + test predictions saved.")

    tree_keys = [k for k in all_test if any(m in k for m in ["catboost", "lightgbm", "xgboost"])]
    if len(tree_keys) >= 2:
        test_tree_matrix = np.stack([all_test[k] for k in tree_keys], axis=1)
        test_std = test_tree_matrix.std(axis=1)
        confidence_threshold = max(0.05, float(np.percentile(test_std, 25)))
        confident_mask = test_std <= confidence_threshold
        pseudo_count = confident_mask.sum()
        logger.info(f"Pseudo-label confidence threshold (std): {confidence_threshold:.4f}")
        logger.info(f"High-confidence samples: {pseudo_count} / {len(test_pd)}")

        if pseudo_count > 0:
            all_test_matrix = np.stack(list(all_test.values()), axis=1)
            pseudo_y = all_test_matrix.mean(axis=1)[confident_mask]
            pseudo_y_full = np.concatenate([y_train, pseudo_y])
            pseudo_weights = np.concatenate([weights, 0.5 * np.ones(pseudo_count)])
            pseudo_train_pd = pd.concat([train_pd, test_pd.iloc[confident_mask]], axis=0).reset_index(drop=True)

            data_dict_r2 = ModelEngine.prepare_data(pseudo_train_pd, test_pd, features, cat_features)
            target_strat_r2 = TargetStrategy(pseudo_train_pd, pseudo_y_full)
            y_bins_r2 = CrossValidator.bin_target(pseudo_y_full)

            all_oof_r2, all_test_r2 = engine_model.run_full_training(
                data_dict=data_dict_r2,
                y_train_orig=pseudo_y_full,
                y_bins=y_bins_r2,
                target_strategy=target_strat_r2,
                target_names=targets,
                model_names=models,
                train_df_pd=pseudo_train_pd,
                test_df_pd=test_pd,
                sample_weights=pseudo_weights,
            )

            for key in all_oof_r2:
                orig_oof = all_oof_r2[key][:len(y_train)]
                np.save(settings.DATA_DIR / f"oof_{key}.npy", orig_oof)
                np.save(settings.DATA_DIR / f"test_{key}.npy", all_test_r2[key])

            logger.success("Round 2 (pseudo-label) complete.")
        else:
            logger.warning("No confident pseudo-labels found. Keeping round 1 results.")
    else:
        logger.warning("Not enough tree models for pseudo-labeling.")


@app.command()
def stack() -> None:
    """Diversity audit + Caruana FS + L2 meta + L3 blend."""
    console.rule("[bold gold1]STACK[/bold gold1]")
    from sklearn.metrics import mean_squared_error as mse

    from src.models.ensemble import EnsembleEngine
    from src.models.validator import CrossValidator

    df = _load_processed()
    y_train = df.filter(pl.col("is_train") == 1).get_column(settings.TARGET_COL).to_numpy().astype(np.float32)

    oof_files = sorted(settings.DATA_DIR.glob("oof_*.npy"))
    test_files = sorted(settings.DATA_DIR.glob("test_*.npy"))
    if not oof_files:
        logger.error("No OOF files found. Run engine first.")
        raise typer.Exit(1)

    model_names = [f.stem.replace("oof_", "") for f in oof_files]
    oof_matrix = np.stack([np.load(f) for f in oof_files], axis=1)
    test_matrix = np.stack([np.load(f) for f in test_files], axis=1)

    single_rmse: dict[str, float] = {}
    for i, name in enumerate(model_names):
        r = float(np.sqrt(mse(y_train, oof_matrix[:, i])))
        single_rmse[name] = r
        logger.info(f"  {name:35s}: {r:.5f}")

    best_single = min(single_rmse, key=single_rmse.get)  # type: ignore[arg-type]
    logger.info(f"Best single model: {best_single} = {single_rmse[best_single]:.5f}")
    logger.info(f"Simple average RMSE: {np.sqrt(mse(y_train, oof_matrix.mean(axis=1))):.5f}")

    _, caruana_w, _ = EnsembleEngine.caruana_forward_selection(oof_matrix, y_train)
    caruana_oof = oof_matrix @ caruana_w
    caruana_test = test_matrix @ caruana_w
    logger.info(f"Caruana OOF RMSE: {np.sqrt(mse(y_train, caruana_oof)):.5f}")

    L2_train = EnsembleEngine.build_meta_features(oof_matrix)
    L2_test = EnsembleEngine.build_meta_features(test_matrix)
    y_bins = CrossValidator.bin_target(y_train)

    ridge_oof, ridge_test = EnsembleEngine.l2_ridge_meta(L2_train, L2_test, y_train, y_bins)
    lgb_oof, lgb_test = EnsembleEngine.l2_lgb_meta(L2_train, L2_test, y_train, y_bins)

    candidates_oof = {"caruana": caruana_oof, "ridge_meta": ridge_oof, "lgb_meta": lgb_oof}
    candidates_test = {"caruana": caruana_test, "ridge_meta": ridge_test, "lgb_meta": lgb_test}

    final_oof, final_test, l3_weights = EnsembleEngine.l3_blend(candidates_oof, candidates_test, y_train)

    np.save(settings.DATA_DIR / "final_oof.npy", final_oof)
    np.save(settings.DATA_DIR / "final_test.npy", final_test)
    np.save(settings.DATA_DIR / "caruana_w.npy", caruana_w)
    with open(settings.DATA_DIR / "l3_weights.json", "w") as f:
        json.dump(l3_weights, f, indent=2)

    final_rmse = float(np.sqrt(mse(y_train, final_oof)))
    logger.info(f"  Best single:   {single_rmse[best_single]:.5f}")
    logger.info(f"  Caruana FS:    {np.sqrt(mse(y_train, caruana_oof)):.5f}")
    logger.info(f"  L2 Ridge:      {np.sqrt(mse(y_train, ridge_oof)):.5f}")
    logger.info(f"  L2 LightGBM:   {np.sqrt(mse(y_train, lgb_oof)):.5f}")
    logger.info(f"  L3 final:      {final_rmse:.5f}")
    logger.success(f"Improvement vs best single: {single_rmse[best_single] - final_rmse:.5f}")


@app.command()
def optimize_stack() -> None:
    """Optuna L2/L3 stack optimization."""
    console.rule("[bold gold1]OPTIMIZE STACK[/bold gold1]")
    import lightgbm as lgb
    import optuna
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error as mse

    df = _load_processed()
    y_train = df.filter(pl.col("is_train") == 1).get_column(settings.TARGET_COL).to_numpy().astype(np.float32)

    oof_files = sorted(settings.DATA_DIR.glob("oof_*.npy"))
    if not oof_files:
        logger.error("No OOF files found.")
        raise typer.Exit(1)

    oof_matrix = np.stack([np.load(f) for f in oof_files], axis=1)
    caruana_w_path = settings.DATA_DIR / "caruana_w.npy"
    if not caruana_w_path.exists():
        logger.error("caruana_w.npy not found. Run stack first.")
        raise typer.Exit(1)
    caruana_w = np.load(caruana_w_path)
    caruana_oof = oof_matrix @ caruana_w

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("ridge_alpha", 0.1, 100.0, log=True)
        ridge_oof = Ridge(alpha=alpha, random_state=42).fit(oof_matrix, y_train).predict(oof_matrix)

        lgb_params = {
            "n_estimators": trial.suggest_int("n_est", 50, 300),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 15),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0),
            "random_state": 42, "verbose": -1, "n_jobs": -1,
        }
        lgb_oof = lgb.LGBMRegressor(**lgb_params).fit(oof_matrix, y_train).predict(oof_matrix)

        w_c = trial.suggest_float("w_caruana", 0.0, 1.0)
        w_r = trial.suggest_float("w_ridge", 0.0, 1.0)
        w_l = trial.suggest_float("w_lgb", 0.0, 1.0)
        total = w_c + w_r + w_l
        if total == 0:
            return 999.0
        blend = (w_c * caruana_oof + w_r * ridge_oof + w_l * lgb_oof) / total
        return float(np.sqrt(mse(y_train, blend)))

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="minimize")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=200)

    logger.success(f"Best RMSE: {study.best_value:.5f}")
    logger.info(f"Best params: {study.best_params}")

    best = study.best_params
    test_files = [settings.DATA_DIR / f.name.replace("oof_", "test_") for f in oof_files]
    test_matrix = np.stack([np.load(f) for f in test_files], axis=1)

    ridge_test = Ridge(alpha=best["ridge_alpha"], random_state=42).fit(oof_matrix, y_train).predict(test_matrix)
    lgb_test = lgb.LGBMRegressor(
        n_estimators=best["n_est"], learning_rate=best["lr"],
        num_leaves=best["num_leaves"], reg_alpha=best["reg_alpha"],
        random_state=42, verbose=-1, n_jobs=-1,
    ).fit(oof_matrix, y_train).predict(test_matrix)

    caruana_test = test_matrix @ caruana_w
    total = best["w_caruana"] + best["w_ridge"] + best["w_lgb"]
    final_test = (best["w_caruana"] * caruana_test + best["w_ridge"] * ridge_test + best["w_lgb"] * lgb_test) / total

    np.save(settings.DATA_DIR / "final_test.npy", final_test)
    logger.success("final_test.npy saved.")


@app.command()
def predict() -> None:
    """Generate submission.csv."""
    console.rule("[bold magenta]PREDICT[/bold magenta]")
    df = _load_processed()
    test_df = df.filter(pl.col("is_train") == 0)

    final_test_path = settings.DATA_DIR / "final_test.npy"
    if not final_test_path.exists():
        logger.error("final_test.npy not found. Run stack first.")
        raise typer.Exit(1)

    final_test = np.load(final_test_path)
    final_clipped = np.clip(final_test, 0.0, 10.0)
    n_clipped = int((final_test != final_clipped).sum())

    test_ids = test_df.get_column(settings.ID_COL)
    assert final_clipped.shape[0] == len(test_ids)
    assert np.all(np.isfinite(final_clipped))

    sub = pl.DataFrame({settings.ID_COL: test_ids, settings.TARGET_COL: final_clipped})
    sub.write_csv("submission.csv")

    logger.info(f"Range: [{final_clipped.min():.2f}, {final_clipped.max():.2f}]  mean: {final_clipped.mean():.2f}")
    if n_clipped > 0:
        logger.warning(f"{n_clipped} predictions clipped")
    logger.success(f"submission.csv written ({len(sub)} rows)")

    final_oof_path = settings.DATA_DIR / "final_oof.npy"
    if final_oof_path.exists():
        from sklearn.metrics import mean_squared_error
        y_train = df.filter(pl.col("is_train") == 1).get_column(settings.TARGET_COL).to_numpy()
        cv_rmse = float(np.sqrt(mean_squared_error(y_train, np.load(final_oof_path))))
        logger.info(f"CV RMSE: {cv_rmse:.5f}")


if __name__ == "__main__":
    app()
