-- mart_pitcher_arsenal: pitcher x season x pitch_type
--
-- Grain is one row per pitch a pitcher throws in a season. Percentile ranks are
-- computed WITHIN (season, pitch_type) so a slider's whiff rate is graded against
-- other sliders rather than against fastballs -- ranking across pitch types would
-- make every breaking ball look elite on whiffs and every fastball look elite on
-- velocity.
--
-- Only tracked, competitive pitches count. Pitch-clock/ABS automatic calls carry
-- no tracking data, and pitchouts/intentional balls distort usage rates.

CREATE OR REPLACE TABLE mart_pitcher_arsenal AS
WITH base AS (
    SELECT
        pitcher                        AS mlbam_id,
        season,
        pitch_type,
        pitch_name,
        p_throws,
        release_speed,
        release_spin_rate,
        release_extension,
        ivb_in,
        hb_arm_in,
        arm_angle,
        spin_axis,
        is_swing::INT                  AS swing,
        is_whiff::INT                  AS whiff,
        is_csw::INT                    AS csw,
        is_in_zone::INT                AS in_zone,
        is_chase::INT                  AS chase,
        is_in_play::INT                AS in_play,
        estimated_woba_using_speedangle AS xwoba,
        launch_speed,
        -- delta_run_exp is signed from the batting team's perspective, so a
        -- negative value is a good outcome for the pitcher. Flip it so higher is
        -- better for the pitcher, matching how RV/100 is normally published.
        -delta_run_exp                 AS pitcher_rv
    FROM fact_pitch
    WHERE is_tracked_pitch
      AND is_competitive
      AND pitch_type IS NOT NULL
      AND game_type = 'R'
),
agg AS (
    SELECT
        mlbam_id,
        season,
        pitch_type,
        any_value(pitch_name)                            AS pitch_name,
        any_value(p_throws)                              AS p_throws,
        count(*)                                         AS pitches,
        avg(release_speed)                               AS velo_avg,
        max(release_speed)                               AS velo_max,
        stddev_samp(release_speed)                       AS velo_sd,
        avg(release_spin_rate)                           AS spin_avg,
        avg(release_extension)                           AS extension_avg,
        avg(ivb_in)                                      AS ivb_in,
        avg(hb_arm_in)                                   AS hb_arm_in,
        avg(arm_angle)                                   AS arm_angle,
        avg(spin_axis)                                   AS spin_axis,
        sum(swing)                                       AS swings,
        sum(whiff)                                       AS whiffs,
        100.0 * sum(whiff) / nullif(sum(swing), 0)       AS whiff_pct,
        100.0 * sum(csw)   / count(*)                    AS csw_pct,
        100.0 * sum(in_zone) / count(*)                  AS zone_pct,
        100.0 * sum(chase) / count(*)                    AS chase_pct,
        avg(xwoba)                                       AS xwoba,
        avg(launch_speed)                                AS exit_velo_avg,
        100.0 * sum(pitcher_rv) / count(*)               AS rv_per_100
    FROM base
    GROUP BY mlbam_id, season, pitch_type
),
with_usage AS (
    SELECT
        *,
        100.0 * pitches / sum(pitches) OVER (PARTITION BY mlbam_id, season) AS usage_pct,
        sum(pitches) OVER (PARTITION BY mlbam_id, season)                   AS season_pitches
    FROM agg
),
-- The pitcher's primary fastball anchors the velo/movement deltas that make a
-- stuff model work: a changeup is judged by its separation from the heater, not
-- by its absolute speed.
--
-- A cutter is deliberately ranked below four-seams and sinkers rather than
-- competing on volume alone. It is its own pitch class -- closer in shape to a
-- slider than to a heater -- so anchoring a high-cutter-usage pitcher's deltas on
-- FC reports his sinker as +7mph "offspeed separation", which is meaningless. FC
-- is used only when the pitcher throws no true fastball at all.
primary_fb AS (
    SELECT mlbam_id, season, velo_avg AS fb_velo, ivb_in AS fb_ivb, hb_arm_in AS fb_hb
    FROM (
        SELECT *, row_number() OVER (
            PARTITION BY mlbam_id, season
            ORDER BY CASE WHEN pitch_type IN ('FF', 'SI', 'FA') THEN 0 ELSE 1 END,
                     pitches DESC
        ) AS rn
        FROM with_usage
        WHERE pitch_type IN ('FF', 'SI', 'FC', 'FA')
    ) WHERE rn = 1
)
SELECT
    u.*,
    u.velo_avg  - f.fb_velo AS velo_diff_fb,
    u.ivb_in    - f.fb_ivb  AS ivb_diff_fb,
    u.hb_arm_in - f.fb_hb   AS hb_diff_fb,
    -- Percentiles only among pitches with enough volume to mean anything;
    -- a 6-pitch sample ranking in the 99th percentile is noise, not a finding.
    CASE WHEN u.pitches >= 50 THEN
        percent_rank() OVER (
            PARTITION BY u.season, u.pitch_type
            ORDER BY u.velo_avg
        ) END AS pct_velo,
    CASE WHEN u.pitches >= 50 THEN
        percent_rank() OVER (
            PARTITION BY u.season, u.pitch_type
            ORDER BY u.whiff_pct
        ) END AS pct_whiff,
    CASE WHEN u.pitches >= 50 THEN
        percent_rank() OVER (
            PARTITION BY u.season, u.pitch_type
            ORDER BY u.csw_pct
        ) END AS pct_csw,
    CASE WHEN u.pitches >= 50 THEN
        percent_rank() OVER (
            PARTITION BY u.season, u.pitch_type
            ORDER BY u.rv_per_100
        ) END AS pct_rv
FROM with_usage u
LEFT JOIN primary_fb f USING (mlbam_id, season);
