
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import config as cfg
from generate_scenarios import get_scenarios

def generate_thermal_images(latent_df, obs_df):

    scenarios = get_scenarios()
    sc_lookup = {s["scenario_id"]: s for s in scenarios}
    gs = cfg.GRID_SIZE
    meta_rows = []

    for sid in latent_df["scenario_id"].unique():
        sc = sc_lookup[sid]
        rows_lat = latent_df[latent_df["scenario_id"] == sid].reset_index(drop=True)
        rows_obs = obs_df[obs_df["scenario_id"] == sid].reset_index(drop=True)
        rng = np.random.default_rng(sc["random_seed"] + 200)

        img_dir = cfg.DATA_THERMAL / sid
        img_dir.mkdir(parents=True, exist_ok=True)

        n = len(rows_lat)
        n_skip = max(1, int(n * cfg.MISSING_THERMAL_FRACTION))
        skip_indices = set(rng.choice(n, size=n_skip, replace=False))

        for idx in range(n):
            r_lat = rows_lat.iloc[idx]
            r_obs = rows_obs.iloc[idx]
            t = int(r_lat["timestamp_min"])

            amb = r_lat["_ambient"] if "_ambient" in r_lat.index else r_obs["ambient_temperature"]
            near_surf = r_lat["near_surface_temperature"]
            ox_state = r_lat["oxidation_state"]

            cx = sc["core_x"]
            cy = sc["core_y"]
            strength = max(0.0, near_surf - amb)
            sigma = min(cfg.HOTSPOT_INITIAL_RADIUS
                        + cfg.HOTSPOT_GROWTH_RATE * ox_state * 100,
                        cfg.HOTSPOT_MAX_RADIUS)

            grid = np.full((gs, gs), amb, dtype=np.float64)

            if strength > 0.01:
                yy, xx = np.mgrid[0:gs, 0:gs]
                dist_sq = (xx - cx)**2 + (yy - cy)**2
                hotspot = strength * np.exp(-dist_sq / (2.0 * sigma**2))
                irregularity = rng.normal(0, 0.3, size=(gs, gs))
                grid += hotspot + irregularity

            grid = gaussian_filter(grid, sigma=cfg.THERMAL_BLUR_SIGMA)
            grid += rng.normal(0, cfg.NOISE_PIXEL_TEMP, size=(gs, gs))

            if rng.random() < 0.02:
                bx = rng.integers(0, gs - 5)
                by = rng.integers(0, gs - 5)
                grid[by:by+5, bx:bx+5] = rng.uniform(amb - 2, amb + 2)

            threshold = amb + cfg.HOTSPOT_ANOMALY_MARGIN
            hot_mask = grid > threshold
            h_area = int(hot_mask.sum())
            if h_area > 0:
                ys, xs = np.where(hot_mask)
                h_x = round(float(xs.mean()), 2)
                h_y = round(float(ys.mean()), 2)
                h_present = 1
            else:
                h_x = float("nan")
                h_y = float("nan")
                h_present = 0

            s_mean = round(float(grid.mean()), 2)
            s_max = round(float(grid.max()), 2)

            img_fname = f"t{t:03d}.png"
            rel_path = f"data/thermal/{sid}/{img_fname}"

            if idx in skip_indices:
                meta_rows.append({
                    "scenario_id": sid,
                    "timestamp_min": t,
                    "image_path": "MISSING",
                    "surface_temperature_mean": s_mean,
                    "surface_temperature_max": s_max,
                    "hotspot_area": h_area,
                    "hotspot_x": h_x,
                    "hotspot_y": h_y,
                    "hotspot_present": h_present,
                })
                continue

            vmin = amb - 2
            vmax = max(amb + 20, grid.max() + 1)
            plt.imsave(str(cfg.DATA_THERMAL / sid / img_fname),
                       grid, cmap=cfg.THERMAL_CMAP, vmin=vmin, vmax=vmax)

            meta_rows.append({
                "scenario_id": sid,
                "timestamp_min": t,
                "image_path": rel_path,
                "surface_temperature_mean": s_mean,
                "surface_temperature_max": s_max,
                "hotspot_area": h_area,
                "hotspot_x": h_x,
                "hotspot_y": h_y,
                "hotspot_present": h_present,
            })

        print(f"  [THERMAL] {sid} — {n - len(skip_indices)} images saved, "
              f"{len(skip_indices)} deliberately missing")

    return pd.DataFrame(meta_rows)

if __name__ == "__main__":
    from generate_latent_state import generate_all_latent
    from generate_observations import generate_observations
    lat = generate_all_latent()
    obs, _ = generate_observations(lat)
    tm = generate_thermal_images(lat, obs)
    print(tm.head())
