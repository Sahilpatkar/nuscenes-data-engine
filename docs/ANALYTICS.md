# Dataset analytics — DuckDB over the Parquet tables

The processed tables are plain Parquet; [DuckDB](https://duckdb.org/) queries them
directly — no warehouse, no table-format migration. The `query` CLI registers views
over `data/processed/` (`samples`, `annotations`, `availability`):

```bash
uv run nuscenes-data-engine query "SELECT ..."
```

`annotations.parquet` and `availability.parquet` are produced on the GPU server —
rsync them here with the rest of `data/processed/` first.

## 1. Boxes per category group × location

```sql
SELECT category_group, location, count(*) AS boxes
FROM annotations GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 8
```

```
│ car            │ boston-seaport       │ 313895 │
│ NULL           │ boston-seaport       │ 203557 │
│ pedestrian     │ boston-seaport       │ 101889 │
│ NULL           │ singapore-onenorth   │  69516 │
│ truck          │ boston-seaport       │  67974 │
│ car            │ singapore-onenorth   │  51305 │
│ pedestrian     │ singapore-onenorth   │  49621 │
│ car            │ singapore-queenstown │  29060 │
```

(`NULL` = categories outside the 10-class detection taxonomy, e.g. static objects.)

## 2. Scenes and images by location × conditions

```sql
SELECT location, is_night, is_rain, count(DISTINCT scene_token) AS scenes, count(*) AS images
FROM samples GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
```

```
│ boston-seaport           │ false │ false │ 318 │ 76542 │
│ boston-seaport           │ false │ true  │ 149 │ 36168 │
│ singapore-hollandvillage │ false │ false │  19 │  4620 │
│ singapore-hollandvillage │ true  │ false │  50 │ 12090 │
│ singapore-hollandvillage │ true  │ true  │  16 │  3852 │
│ singapore-onenorth       │ false │ false │ 183 │ 43848 │
│ singapore-queenstown     │ false │ false │  82 │ 19794 │
│ singapore-queenstown     │ true  │ false │  33 │  7980 │
```

Night driving exists only in the Singapore logs; Boston contributes all the rain.

## 3. Average boxes per image, day vs night

```sql
SELECT channel, is_night, round(avg(n_boxes), 2) AS avg_boxes_per_image
FROM samples GROUP BY 1, 2 ORDER BY 1, 2
```

```
│ CAM_BACK        │ false │ 8.47 │        │ CAM_FRONT       │ false │ 7.60 │
│ CAM_BACK        │ true  │ 4.42 │        │ CAM_FRONT       │ true  │ 4.44 │
│ CAM_BACK_LEFT   │ false │ 3.71 │        │ CAM_FRONT_LEFT  │ false │ 4.45 │
│ CAM_BACK_LEFT   │ true  │ 0.84 │        │ CAM_FRONT_LEFT  │ true  │ 1.19 │
│ CAM_BACK_RIGHT  │ false │ 3.24 │        │ CAM_FRONT_RIGHT │ false │ 3.95 │
│ CAM_BACK_RIGHT  │ true  │ 1.34 │        │ CAM_FRONT_RIGHT │ true  │ 1.96 │
```

Night frames carry roughly half the annotated objects of day frames — the same signal
the Phase 5 drift monitor keys on (`n_boxes`), independently confirmed here.

## 4. Referenced vs present files per channel (availability manifest)

```sql
SELECT channel, is_key_frame, count(*) AS referenced, sum(present::int) AS present
FROM availability GROUP BY 1, 2 ORDER BY 1, 2
```

```
│ CAM_* (x6)        │ true  │ 34149 each  │ 34149 each │   all keyframes present
│ CAM_* (x6)        │ false │ ~160-164K   │ all        │   all sweeps present
│ LIDAR_TOP         │ true  │ 34149       │ 34149      │
│ LIDAR_TOP         │ false │ 297737      │ 0          │   sweeps absent
│ RADAR_* (x5)      │ both  │ ~220K each  │ 0          │   radar entirely absent
```

Radar is referenced by the metadata but entirely absent on this server, and LiDAR has
keyframes only — exactly why the pipeline filters on the manifest instead of trusting
metadata paths (see [DATA.md](DATA.md)).
