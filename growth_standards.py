"""
UltraMeasure — Growth Standards
Hadlock (1984) fetal AC reference values by gestational age.
Used to interpret measured AC against expected normal range.

Reference:
    Hadlock FP et al. (1984) Sonographic estimation of fetal age and weight.
    Radiology 150:535-540.
    WHO Fetal Growth Charts (Intergrowth-21 crosswalk where noted).
"""

# ── Hadlock AC Reference Table ─────────────────────────────────────────────
# Gestational age in weeks → (5th percentile, 50th percentile, 95th percentile)
# All values in millimetres.

HADLOCK_AC = {
    14: ( 73,  88, 103),
    15: ( 85, 100, 116),
    16: ( 96, 113, 130),
    17: (109, 126, 144),
    18: (121, 140, 158),
    19: (133, 153, 173),
    20: (146, 166, 187),
    21: (158, 179, 201),
    22: (170, 192, 215),
    23: (182, 206, 229),
    24: (194, 219, 244),
    25: (207, 232, 258),
    26: (219, 245, 272),
    27: (231, 258, 286),
    28: (243, 271, 300),
    29: (255, 284, 314),
    30: (267, 297, 328),
    31: (279, 310, 341),
    32: (290, 322, 354),
    33: (301, 334, 368),
    34: (311, 346, 381),
    35: (321, 357, 393),
    36: (330, 368, 405),
    37: (339, 378, 417),
    38: (347, 388, 428),
    39: (354, 397, 440),
    40: (360, 406, 451),
    41: (365, 414, 462),
    42: (369, 421, 472),
}


# ── LMP → Gestational Age ──────────────────────────────────────────────────

def lmp_to_ga_weeks(lmp_date, scan_date=None) -> float:
    """
    Calculate gestational age in weeks from LMP date.
    If scan_date is None, uses today's date.

    Args:
        lmp_date  : datetime.date object
        scan_date : datetime.date object or None

    Returns:
        Gestational age in weeks (float)
    """
    from datetime import date
    if scan_date is None:
        scan_date = date.today()
    delta = scan_date - lmp_date
    return delta.days / 7.0


# ── Interpolated reference ─────────────────────────────────────────────────

def get_reference(ga_weeks: float) -> dict:
    """
    Return interpolated Hadlock AC reference values for a given GA.

    Args:
        ga_weeks : gestational age in weeks (float, e.g. 24.5)

    Returns:
        dict with keys: ga_weeks, p5, p50, p95, in_table (bool)
    """
    ga_floor = int(ga_weeks)
    ga_ceil  = ga_floor + 1
    frac     = ga_weeks - ga_floor

    if ga_floor not in HADLOCK_AC and ga_ceil not in HADLOCK_AC:
        return {
            'ga_weeks' : ga_weeks,
            'p5'       : None,
            'p50'      : None,
            'p95'      : None,
            'in_table' : False,
        }

    # Clamp to table bounds
    ga_floor = max(14, min(ga_floor, 42))
    ga_ceil  = max(14, min(ga_ceil,  42))

    p5_floor,  p50_floor,  p95_floor  = HADLOCK_AC[ga_floor]
    p5_ceil,   p50_ceil,   p95_ceil   = HADLOCK_AC[ga_ceil]

    # Linear interpolation
    p5  = p5_floor  + frac * (p5_ceil  - p5_floor)
    p50 = p50_floor + frac * (p50_ceil - p50_floor)
    p95 = p95_floor + frac * (p95_ceil - p95_floor)

    return {
        'ga_weeks' : ga_weeks,
        'p5'       : round(p5,  1),
        'p50'      : round(p50, 1),
        'p95'      : round(p95, 1),
        'in_table' : True,
    }


# ── Clinical Interpretation ────────────────────────────────────────────────

def interpret_ac(ac_mm: float, ga_weeks: float) -> dict:
    """
    Compare measured AC to Hadlock reference and return clinical interpretation.

    Returns:
        dict with keys:
            status  : 'normal' | 'small' | 'large' | 'unknown'
            label   : short display label
            message : full clinical suggestion string
            colour  : hex colour for UI badge
            ref     : reference dict from get_reference()
    """
    ref = get_reference(ga_weeks)

    if not ref['in_table']:
        return {
            'status'  : 'unknown',
            'label'   : 'Out of range',
            'message' : (
                f"Gestational age {ga_weeks:.1f} weeks is outside the "
                f"Hadlock (1984) reference range (14–42 weeks). "
                f"Clinical interpretation is not available."
            ),
            'colour'  : '#6B7280',
            'ref'     : ref,
        }

    p5, p50, p95 = ref['p5'], ref['p50'], ref['p95']

    # Compute approximate percentile position
    if ac_mm < p5:
        pct_label = "below the 5th percentile"
        status    = 'small'
        label     = 'Small for Gestational Age'
        colour    = '#DC2626'    # red
        message   = (
            f"The measured AC of {ac_mm:.1f} mm is {pct_label} for "
            f"{ga_weeks:.1f} weeks gestation (Hadlock reference: "
            f"5th={p5} mm, 50th={p50} mm, 95th={p95} mm). "
            f"This finding is consistent with fetal growth restriction (FGR) "
            f"and warrants further clinical assessment, including Doppler "
            f"velocimetry and serial biometry."
        )
    elif ac_mm > p95:
        pct_label = "above the 95th percentile"
        status    = 'large'
        label     = 'Large for Gestational Age'
        colour    = '#D97706'    # amber
        message   = (
            f"The measured AC of {ac_mm:.1f} mm is {pct_label} for "
            f"{ga_weeks:.1f} weeks gestation (Hadlock reference: "
            f"5th={p5} mm, 50th={p50} mm, 95th={p95} mm). "
            f"This finding may indicate macrosomia. Consider screening for "
            f"gestational diabetes and further clinical evaluation."
        )
    else:
        pct_label = "within the normal range"
        status    = 'normal'
        label     = 'Appropriate for Gestational Age'
        colour    = '#059669'    # green
        message   = (
            f"The measured AC of {ac_mm:.1f} mm is {pct_label} "
            f"(between the 5th and 95th percentile) for "
            f"{ga_weeks:.1f} weeks gestation (Hadlock reference: "
            f"5th={p5} mm, 50th={p50} mm, 95th={p95} mm). "
            f"Fetal abdominal growth appears appropriate."
        )

    return {
        'status'  : status,
        'label'   : label,
        'message' : message,
        'colour'  : colour,
        'ref'     : ref,
    }


# ── Chart data for plotting ────────────────────────────────────────────────

def get_chart_data() -> dict:
    """Return p5, p50, p95 arrays for all GA weeks for Plotly chart."""
    weeks = sorted(HADLOCK_AC.keys())
    return {
        'weeks' : weeks,
        'p5'    : [HADLOCK_AC[w][0] for w in weeks],
        'p50'   : [HADLOCK_AC[w][1] for w in weeks],
        'p95'   : [HADLOCK_AC[w][2] for w in weeks],
    }
